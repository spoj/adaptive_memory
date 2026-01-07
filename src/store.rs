//! MemoryStore - the main API for the adaptive memory system.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

use crate::db::{get_max_memory_id, init_schema};
use crate::error::MemoryError;
use crate::memory::{
    AddMemoryResult, GraphStats, Memory, Stats, Timeline, TimelineBucket, TimelineSummary,
};
use crate::relationship::{
    StrengthenResult, add_relationship_event, canonicalize, get_relationship,
};
use crate::search::{RelatedResult, SearchResult, find_related, surface_candidates};
use crate::{MAX_STRENGTHEN_SET, SearchParams};

/// Payload for an "add" operation (for undo).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AddPayload {
    memory_id: i64,
    text: String,
}

/// Payload for a "strengthen" operation (for undo).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StrengthenPayload {
    /// The relationship event IDs that were created.
    event_ids: Vec<i64>,
    /// The memory IDs that were strengthened (for display).
    memory_ids: Vec<i64>,
}

/// Result of an undo operation.
#[derive(Debug, Serialize)]
pub struct UndoResult {
    /// Type of operation that was undone ("add" or "strengthen").
    pub op_type: String,
    /// Human-readable description of what was undone.
    pub description: String,
}

/// The main interface for the adaptive memory system.
///
/// Wraps a SQLite connection and provides methods for adding, searching,
/// and strengthening memories. Caches the maximum memory ID for relationship
/// event timestamps.
pub struct MemoryStore {
    conn: Connection,
    cached_max_mem: i64,
}

impl MemoryStore {
    /// Open or create a memory store at the given path.
    ///
    /// Automatically initializes the schema if needed and caches the
    /// current maximum memory ID.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// Create an in-memory store (useful for testing).
    pub fn open_in_memory() -> Result<Self, MemoryError> {
        let conn = Connection::open_in_memory()?;
        Self::init(conn)
    }

    /// Common initialization logic.
    fn init(conn: Connection) -> Result<Self, MemoryError> {
        init_schema(&conn)?;
        let cached_max_mem = get_max_memory_id(&conn)?;
        Ok(Self {
            conn,
            cached_max_mem,
        })
    }

    /// Get the current maximum memory ID (cached).
    pub fn max_memory_id(&self) -> i64 {
        self.cached_max_mem
    }

    /// Add a new memory.
    ///
    /// Note: No temporal relationships are created. Use strengthen() to create
    /// explicit relationships, or --context flag at search time for temporal context.
    pub fn add(
        &mut self,
        text: &str,
        source: Option<&str>,
    ) -> Result<AddMemoryResult, MemoryError> {
        self.add_with_options(text, source, None)
    }

    /// Add a new memory with optional datetime override.
    ///
    /// If `datetime_str` is provided, it must be in RFC3339 format (e.g. "2024-01-15T10:30:00Z").
    /// If not provided, the current time is used.
    pub fn add_with_options(
        &mut self,
        text: &str,
        source: Option<&str>,
        datetime_str: Option<&str>,
    ) -> Result<AddMemoryResult, MemoryError> {
        let datetime = if let Some(dt_str) = datetime_str {
            DateTime::parse_from_rfc3339(dt_str)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| {
                    MemoryError::InvalidInput(format!(
                        "Invalid datetime format (expected RFC3339, e.g. '2024-01-15T10:30:00Z'): {}",
                        e
                    ))
                })?
        } else {
            Utc::now()
        };

        let datetime_str_to_store = datetime.to_rfc3339();

        let tx = self.conn.transaction()?;

        // Insert the new memory (no temporal relationships created)
        tx.execute(
            "INSERT INTO memories (datetime, text, source) VALUES (?1, ?2, ?3)",
            params![datetime_str_to_store, text, source],
        )?;

        let new_id = tx.last_insert_rowid();

        // Log the operation for undo
        let payload = AddPayload {
            memory_id: new_id,
            text: text.to_string(),
        };
        let payload_json = serde_json::to_string(&payload).map_err(|e| {
            MemoryError::InvalidInput(format!("Failed to serialize payload: {}", e))
        })?;

        tx.execute(
            "INSERT INTO operations (op_type, payload, created_at) VALUES (?1, ?2, ?3)",
            params!["add", payload_json, Utc::now().to_rfc3339()],
        )?;

        tx.commit()?;

        // Update cache
        self.cached_max_mem = new_id;

        let memory = Memory {
            id: new_id,
            datetime,
            text: text.to_string(),
            source: source.map(|s| s.to_string()),
        };

        Ok(AddMemoryResult { memory })
    }

    /// Search for memories using text query and spreading activation.
    ///
    /// If the query is empty, returns the most recent memories.
    pub fn search(&self, query: &str, params: &SearchParams) -> Result<SearchResult, MemoryError> {
        surface_candidates(&self.conn, query, params)
    }

    /// Find memories related to the given seed IDs using PPR (no text search).
    ///
    /// This is like search but skips the FTS5 keyword step - you provide
    /// seed memory IDs directly and get related memories via graph traversal.
    pub fn related(
        &self,
        seed_ids: &[i64],
        params: &SearchParams,
    ) -> Result<RelatedResult, MemoryError> {
        find_related(&self.conn, seed_ids, params)
    }

    /// Strengthen relationships between a set of memory IDs.
    ///
    /// Adds a new explicit relationship event for each pair with strength 1.0.
    /// Repeated calls accumulate strength.
    ///
    /// This operation is wrapped in a transaction for atomicity.
    pub fn strengthen(&mut self, ids: &[i64]) -> Result<StrengthenResult, MemoryError> {
        if ids.len() > MAX_STRENGTHEN_SET {
            return Err(MemoryError::InvalidInput(format!(
                "Cannot strengthen more than {} memories at once (got {})",
                MAX_STRENGTHEN_SET,
                ids.len()
            )));
        }

        if ids.is_empty() {
            return Err(MemoryError::InvalidInput(
                "At least one memory ID is required".to_string(),
            ));
        }

        if ids.len() == 1 {
            return Err(MemoryError::InvalidInput(
                "At least two memory IDs are required to create relationships".to_string(),
            ));
        }

        let tx = self.conn.transaction()?;

        let mut relationships = Vec::new();
        let mut event_ids = Vec::new();

        // Generate all pairs and add a new event for each (1.0 strength per pair)
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (from_mem, to_mem) = canonicalize(ids[i], ids[j]);

                // Add new relationship event with 1.0 strength
                let event_id = add_relationship_event(&tx, from_mem, to_mem, 1.0)?;
                event_ids.push(event_id);

                // Get the aggregated relationship (including the new event)
                if let Some(rel) = get_relationship(&tx, from_mem, to_mem)? {
                    relationships.push(rel);
                }
            }
        }

        // Log the operation for undo
        let payload = StrengthenPayload {
            event_ids,
            memory_ids: ids.to_vec(),
        };
        let payload_json = serde_json::to_string(&payload).map_err(|e| {
            MemoryError::InvalidInput(format!("Failed to serialize payload: {}", e))
        })?;

        tx.execute(
            "INSERT INTO operations (op_type, payload, created_at) VALUES (?1, ?2, ?3)",
            params!["strengthen", payload_json, Utc::now().to_rfc3339()],
        )?;

        tx.commit()?;

        Ok(StrengthenResult { relationships })
    }

    /// Undo the last operation (add or strengthen).
    ///
    /// This pops the most recent operation from the stack and reverses it:
    /// - For "add": deletes the memory and any relationships pointing to it
    /// - For "strengthen": deletes the relationship events that were created
    ///
    /// Returns an error if there are no operations to undo.
    pub fn undo(&mut self) -> Result<UndoResult, MemoryError> {
        // Get the last operation
        let (op_id, op_type, payload_json): (i64, String, String) = self
            .conn
            .query_row(
                "SELECT id, op_type, payload FROM operations ORDER BY id DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(|_| MemoryError::InvalidInput("No operations to undo".to_string()))?;

        let tx = self.conn.transaction()?;

        let description = match op_type.as_str() {
            "add" => {
                let payload: AddPayload = serde_json::from_str(&payload_json).map_err(|e| {
                    MemoryError::InvalidInput(format!("Failed to parse add payload: {}", e))
                })?;

                // Delete any relationships pointing to this memory
                tx.execute(
                    "DELETE FROM relationships WHERE from_mem = ?1 OR to_mem = ?1",
                    params![payload.memory_id],
                )?;

                // Delete the memory (FTS trigger will handle cleanup)
                tx.execute(
                    "DELETE FROM memories WHERE id = ?1",
                    params![payload.memory_id],
                )?;

                // Truncate text for display
                let display_text = if payload.text.len() > 50 {
                    format!("{}...", &payload.text[..50])
                } else {
                    payload.text.clone()
                };

                format!("add memory #{}: \"{}\"", payload.memory_id, display_text)
            }
            "strengthen" => {
                let payload: StrengthenPayload =
                    serde_json::from_str(&payload_json).map_err(|e| {
                        MemoryError::InvalidInput(format!(
                            "Failed to parse strengthen payload: {}",
                            e
                        ))
                    })?;

                // Delete the specific relationship events
                for event_id in &payload.event_ids {
                    tx.execute("DELETE FROM relationships WHERE id = ?1", params![event_id])?;
                }

                let ids_str: Vec<String> =
                    payload.memory_ids.iter().map(|id| id.to_string()).collect();
                format!(
                    "strengthen {} relationships between memories [{}]",
                    payload.event_ids.len(),
                    ids_str.join(", ")
                )
            }
            _ => {
                return Err(MemoryError::InvalidInput(format!(
                    "Unknown operation type: {}",
                    op_type
                )));
            }
        };

        // Remove the operation from the log
        tx.execute("DELETE FROM operations WHERE id = ?1", params![op_id])?;

        tx.commit()?;

        // Update cached max memory ID
        self.cached_max_mem = get_max_memory_id(&self.conn)?;

        Ok(UndoResult {
            op_type,
            description,
        })
    }

    /// Get a memory by ID.
    pub fn get(&self, id: i64) -> Result<Option<Memory>, MemoryError> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, datetime, text, source FROM memories WHERE id = ?1")?;

        let mut rows = stmt.query(params![id])?;

        if let Some(row) = rows.next()? {
            Ok(Some(Self::row_to_memory(row)?))
        } else {
            Ok(None)
        }
    }

    /// Get the latest N memories, ordered by ID descending (most recent first).
    /// Shorthand for `list(None, None, Some(n))`.
    pub fn tail(&self, n: usize) -> Result<Vec<Memory>, MemoryError> {
        self.list(None, None, Some(n))
    }

    /// List memories by ID range.
    ///
    /// - `from_id`: Start ID (inclusive). If None, starts from the beginning.
    /// - `to_id`: End ID (inclusive). If None, goes to the end.
    /// - `limit`: Maximum number of results. If None, returns all in range.
    ///
    /// Results are ordered by ID descending (most recent first).
    pub fn list(
        &self,
        from_id: Option<i64>,
        to_id: Option<i64>,
        limit: Option<usize>,
    ) -> Result<Vec<Memory>, MemoryError> {
        let mut conditions = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(from) = from_id {
            conditions.push("id >= ?".to_string());
            params_vec.push(Box::new(from));
        }

        if let Some(to) = to_id {
            conditions.push("id <= ?".to_string());
            params_vec.push(Box::new(to));
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", conditions.join(" AND "))
        };

        let limit_clause = if let Some(n) = limit {
            params_vec.push(Box::new(n as i64));
            " LIMIT ?".to_string()
        } else {
            String::new()
        };

        let query = format!(
            "SELECT id, datetime, text, source FROM memories{} ORDER BY id DESC{}",
            where_clause, limit_clause
        );

        let mut stmt = self.conn.prepare(&query)?;

        let param_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), Self::row_to_memory)?;

        let memories: Result<Vec<_>, _> = rows.collect();
        Ok(memories?)
    }

    /// Sample unconnected (stray) memories - memories with no relationships.
    ///
    /// Returns up to `limit` memories that have no connections, ordered randomly.
    pub fn stray(&self, limit: usize) -> Result<Vec<Memory>, MemoryError> {
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.datetime, m.text, m.source
             FROM memories m
             WHERE NOT EXISTS (
                 SELECT 1 FROM relationships r
                 WHERE r.from_mem = m.id OR r.to_mem = m.id
             )
             ORDER BY RANDOM()
             LIMIT ?1",
        )?;

        let rows = stmt.query_map(params![limit as i64], Self::row_to_memory)?;

        let memories: Result<Vec<_>, _> = rows.collect();
        Ok(memories?)
    }

    /// Get database statistics.
    pub fn stats(&self) -> Result<Stats, MemoryError> {
        let memory_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

        let min_memory_id: Option<i64> =
            self.conn
                .query_row("SELECT MIN(id) FROM memories", [], |row| row.get(0))?;

        let max_memory_id: Option<i64> =
            self.conn
                .query_row("SELECT MAX(id) FROM memories", [], |row| row.get(0))?;

        // Count unique relationship pairs
        let relationship_count: i64 = self.conn.query_row(
            "SELECT COUNT(DISTINCT from_mem || '-' || to_mem) FROM relationships",
            [],
            |row| row.get(0),
        )?;

        // Count total relationship events
        let relationship_event_count: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))?;

        // Get unique sources
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT source FROM memories WHERE source IS NOT NULL ORDER BY source",
        )?;
        let sources: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        // Graph metrics
        let graph = self.compute_graph_stats()?;

        Ok(Stats {
            memory_count,
            min_memory_id,
            max_memory_id,
            relationship_count,
            relationship_event_count,
            unique_sources: sources,
            graph,
        })
    }

    /// Compute graph-oriented statistics.
    fn compute_graph_stats(&self) -> Result<GraphStats, MemoryError> {
        use std::collections::{HashMap, HashSet};

        // Count stray memories (no connections)
        let stray_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories m
             WHERE NOT EXISTS (
                 SELECT 1 FROM relationships r
                 WHERE r.from_mem = m.id OR r.to_mem = m.id
             )",
            [],
            |row| row.get(0),
        )?;

        // Build adjacency list for connected component analysis
        let mut adj: HashMap<i64, HashSet<i64>> = HashMap::new();

        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT from_mem, to_mem FROM relationships")?;
        let edges = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;

        for edge in edges {
            let (from, to) = edge?;
            adj.entry(from).or_default().insert(to);
            adj.entry(to).or_default().insert(from);
        }

        // Compute degree stats
        let degrees: Vec<i64> = adj
            .values()
            .map(|neighbors| neighbors.len() as i64)
            .collect();
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        let avg_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<i64>() as f64 / degrees.len() as f64
        };

        // Count leaves (degree == 1)
        let leaf_count = degrees.iter().filter(|&&d| d == 1).count() as i64;

        // Find connected components using BFS
        let mut visited: HashSet<i64> = HashSet::new();
        let mut island_sizes: Vec<i64> = Vec::new();

        for &node in adj.keys() {
            if visited.contains(&node) {
                continue;
            }

            // BFS to find component size
            let mut queue = vec![node];
            let mut component_size = 0i64;

            while let Some(current) = queue.pop() {
                if visited.contains(&current) {
                    continue;
                }
                visited.insert(current);
                component_size += 1;

                if let Some(neighbors) = adj.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            queue.push(neighbor);
                        }
                    }
                }
            }

            island_sizes.push(component_size);
        }

        let island_count = island_sizes.len() as i64;
        let largest_island_size = island_sizes.iter().copied().max().unwrap_or(0);

        Ok(GraphStats {
            stray_count,
            island_count,
            largest_island_size,
            leaf_count,
            max_degree,
            avg_degree,
        })
    }

    /// Get the timeline showing memory ID distribution by date.
    ///
    /// Returns daily buckets with ID ranges, useful for finding the ID range
    /// for a specific date when using ranged search.
    pub fn timeline(&self) -> Result<Timeline, MemoryError> {
        // Query to get daily aggregates
        let mut stmt = self.conn.prepare(
            "SELECT 
                DATE(datetime) as date,
                MIN(id) as min_id,
                MAX(id) as max_id,
                COUNT(*) as count
             FROM memories
             GROUP BY DATE(datetime)
             ORDER BY date DESC",
        )?;

        let buckets: Vec<TimelineBucket> = stmt
            .query_map([], |row| {
                Ok(TimelineBucket {
                    date: row.get(0)?,
                    min_id: row.get(1)?,
                    max_id: row.get(2)?,
                    count: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Compute summary
        let total_memories: i64 = buckets.iter().map(|b| b.count).sum();
        let total_days = buckets.len();
        let avg_per_day = if total_days > 0 {
            total_memories as f64 / total_days as f64
        } else {
            0.0
        };

        let newest = buckets.first();
        let oldest = buckets.last();

        let summary = TimelineSummary {
            total_memories,
            oldest_date: oldest.map(|b| b.date.clone()),
            newest_date: newest.map(|b| b.date.clone()),
            oldest_id: oldest.map(|b| b.min_id),
            newest_id: newest.map(|b| b.max_id),
            total_days,
            avg_per_day,
        };

        Ok(Timeline { buckets, summary })
    }

    /// Get multiple memories by their IDs.
    pub fn get_many(&self, ids: &[i64]) -> Result<Vec<Memory>, MemoryError> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let query = format!(
            "SELECT id, datetime, text, source FROM memories WHERE id IN ({})",
            placeholders
        );

        let mut stmt = self.conn.prepare(&query)?;

        let params: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

        let rows = stmt.query_map(params.as_slice(), |row| {
            let datetime_str: String = row.get(1)?;
            let datetime = DateTime::parse_from_rfc3339(&datetime_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok(Memory {
                id: row.get(0)?,
                datetime,
                text: row.get(2)?,
                source: row.get(3)?,
            })
        })?;

        let memories: Result<Vec<_>, _> = rows.collect();
        Ok(memories?)
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Convert a row to a Memory struct.
    fn row_to_memory(row: &rusqlite::Row) -> Result<Memory, rusqlite::Error> {
        let datetime_str: String = row.get(1)?;
        let datetime = DateTime::parse_from_rfc3339(&datetime_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(Memory {
            id: row.get(0)?,
            datetime,
            text: row.get(2)?,
            source: row.get(3)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory() {
        let store = MemoryStore::open_in_memory().unwrap();
        assert_eq!(store.max_memory_id(), 0);
    }

    #[test]
    fn test_add_memory() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        let result = store.add("Test memory", Some("test")).unwrap();
        assert_eq!(result.memory.text, "Test memory");
        assert_eq!(result.memory.source, Some("test".to_string()));
        assert_eq!(store.max_memory_id(), 1);

        // Add another memory (no temporal relationships created)
        let result2 = store.add("Second memory", None).unwrap();
        assert_eq!(result2.memory.id, 2);
        assert_eq!(store.max_memory_id(), 2);
    }

    #[test]
    fn test_get_memory() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("Test memory", Some("test")).unwrap();

        let mem = store.get(1).unwrap().unwrap();
        assert_eq!(mem.text, "Test memory");
        assert_eq!(mem.source, Some("test".to_string()));

        // Non-existent
        let none = store.get(999).unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn test_get_many() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("First", None).unwrap();
        store.add("Second", None).unwrap();
        store.add("Third", None).unwrap();

        let memories = store.get_many(&[1, 3]).unwrap();
        assert_eq!(memories.len(), 2);

        // Empty
        let empty = store.get_many(&[]).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_strengthen() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Add some memories (no temporal relationships created)
        store.add("mem1", None).unwrap();
        store.add("mem2", None).unwrap();
        store.add("mem3", None).unwrap();

        // Strengthen memories 1, 2, 3 - creates 1.0 strength for each pair
        let result = store.strengthen(&[1, 2, 3]).unwrap();
        assert_eq!(result.relationships.len(), 3);

        // Each relationship should have 1 event with strength 1.0
        for rel in &result.relationships {
            assert!((rel.effective_strength - 1.0).abs() < 0.001);
        }

        // Strengthen just 2 memories again - adds another 1.0 event
        let result = store.strengthen(&[1, 2]).unwrap();
        assert_eq!(result.relationships.len(), 1);
        assert!((result.relationships[0].effective_strength - 2.0).abs() < 0.001);
        // 2.0 total
    }

    #[test]
    fn test_strengthen_validation() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Empty list
        let err = store.strengthen(&[]).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));

        // Single ID
        let err = store.strengthen(&[1]).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));

        // Too many IDs
        let ids: Vec<i64> = (1..=15).collect();
        let err = store.strengthen(&ids).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }

    #[test]
    fn test_search() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("First memory about cats", None).unwrap();
        store.add("Second memory about dogs", None).unwrap();
        store.add("Third memory about birds", None).unwrap();

        let params = SearchParams::default();

        // Empty search should return error
        let err = store.search("", &params).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));

        // Search for specific term
        let result = store.search("cats", &params).unwrap();
        assert!(!result.memories.is_empty());
        assert!(result.memories[0].memory.text.contains("cats"));
    }

    #[test]
    fn test_list() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("First", Some("src1")).unwrap();
        store.add("Second", Some("src1")).unwrap();
        store.add("Third", Some("src2")).unwrap();
        store.add("Fourth", None).unwrap();
        store.add("Fifth", Some("src2")).unwrap();

        // List all (no filters)
        let all = store.list(None, None, None).unwrap();
        assert_eq!(all.len(), 5);
        // Should be ordered by ID descending
        assert_eq!(all[0].id, 5);
        assert_eq!(all[4].id, 1);

        // List with from_id
        let from_3 = store.list(Some(3), None, None).unwrap();
        assert_eq!(from_3.len(), 3);
        assert_eq!(from_3[0].id, 5);
        assert_eq!(from_3[2].id, 3);

        // List with to_id
        let to_3 = store.list(None, Some(3), None).unwrap();
        assert_eq!(to_3.len(), 3);
        assert_eq!(to_3[0].id, 3);
        assert_eq!(to_3[2].id, 1);

        // List with range
        let range = store.list(Some(2), Some(4), None).unwrap();
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].id, 4);
        assert_eq!(range[2].id, 2);

        // List with limit
        let limited = store.list(None, None, Some(2)).unwrap();
        assert_eq!(limited.len(), 2);
        assert_eq!(limited[0].id, 5);
        assert_eq!(limited[1].id, 4);

        // List with range and limit
        let range_limited = store.list(Some(1), Some(5), Some(2)).unwrap();
        assert_eq!(range_limited.len(), 2);
    }

    #[test]
    fn test_tail_uses_list() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("First", None).unwrap();
        store.add("Second", None).unwrap();
        store.add("Third", None).unwrap();

        let tail = store.tail(2).unwrap();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].id, 3);
        assert_eq!(tail[1].id, 2);
    }

    #[test]
    fn test_stats() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Empty db stats
        let stats = store.stats().unwrap();
        assert_eq!(stats.memory_count, 0);
        assert_eq!(stats.min_memory_id, None);
        assert_eq!(stats.max_memory_id, None);
        assert_eq!(stats.relationship_count, 0);
        assert_eq!(stats.relationship_event_count, 0);
        assert!(stats.unique_sources.is_empty());

        // Add some memories
        store.add("First", Some("src1")).unwrap();
        store.add("Second", Some("src2")).unwrap();
        store.add("Third", Some("src1")).unwrap();
        store.add("Fourth", None).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.memory_count, 4);
        assert_eq!(stats.min_memory_id, Some(1));
        assert_eq!(stats.max_memory_id, Some(4));
        assert_eq!(stats.unique_sources, vec!["src1", "src2"]);

        // Check graph stats before relationships
        assert_eq!(stats.graph.stray_count, 4); // all memories are stray
        assert_eq!(stats.graph.island_count, 0); // no islands yet
        assert_eq!(stats.graph.largest_island_size, 0);

        // Add relationships
        store.strengthen(&[1, 2]).unwrap();
        store.strengthen(&[1, 2, 3]).unwrap(); // adds events for (1,2), (1,3), (2,3)

        let stats = store.stats().unwrap();
        assert_eq!(stats.relationship_count, 3); // unique pairs: (1,2), (1,3), (2,3)
        assert_eq!(stats.relationship_event_count, 4); // 1 + 3 events

        // Check graph stats after relationships
        assert_eq!(stats.graph.stray_count, 1); // only memory 4 is stray
        assert_eq!(stats.graph.island_count, 1); // one island (1,2,3)
        assert_eq!(stats.graph.largest_island_size, 3);
        assert_eq!(stats.graph.leaf_count, 0); // all have degree 2
        assert_eq!(stats.graph.max_degree, 2);
    }

    #[test]
    fn test_undo_add() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Add a memory
        let result = store.add("Test memory to undo", Some("test")).unwrap();
        assert_eq!(result.memory.id, 1);
        assert_eq!(store.max_memory_id(), 1);

        // Verify it exists
        let mem = store.get(1).unwrap();
        assert!(mem.is_some());

        // Undo the add
        let undo_result = store.undo().unwrap();
        assert_eq!(undo_result.op_type, "add");
        assert!(undo_result.description.contains("memory #1"));

        // Verify it's gone
        let mem = store.get(1).unwrap();
        assert!(mem.is_none());
        assert_eq!(store.max_memory_id(), 0);

        // Undo again should fail (no operations left)
        let err = store.undo().unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }

    #[test]
    fn test_undo_strengthen() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Add memories
        store.add("mem1", None).unwrap();
        store.add("mem2", None).unwrap();
        store.add("mem3", None).unwrap();

        // Strengthen creates relationships
        let result = store.strengthen(&[1, 2, 3]).unwrap();
        assert_eq!(result.relationships.len(), 3);

        // Verify relationships exist
        let stats = store.stats().unwrap();
        assert_eq!(stats.relationship_count, 3);
        assert_eq!(stats.relationship_event_count, 3);

        // Undo the strengthen
        let undo_result = store.undo().unwrap();
        assert_eq!(undo_result.op_type, "strengthen");
        assert!(undo_result.description.contains("3 relationships"));

        // Verify relationships are gone
        let stats = store.stats().unwrap();
        assert_eq!(stats.relationship_count, 0);
        assert_eq!(stats.relationship_event_count, 0);

        // Memories should still exist
        assert!(store.get(1).unwrap().is_some());
        assert!(store.get(2).unwrap().is_some());
        assert!(store.get(3).unwrap().is_some());
    }

    #[test]
    fn test_undo_sequence() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Add memory
        store.add("First memory", None).unwrap();
        store.add("Second memory", None).unwrap();

        // Strengthen
        store.strengthen(&[1, 2]).unwrap();

        // Undo strengthen first
        let undo_result = store.undo().unwrap();
        assert_eq!(undo_result.op_type, "strengthen");

        // Undo second add
        let undo_result = store.undo().unwrap();
        assert_eq!(undo_result.op_type, "add");
        assert!(undo_result.description.contains("memory #2"));

        // Undo first add
        let undo_result = store.undo().unwrap();
        assert_eq!(undo_result.op_type, "add");
        assert!(undo_result.description.contains("memory #1"));

        // No more operations
        let err = store.undo().unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }

    #[test]
    fn test_undo_add_with_relationships() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Add memories
        store.add("mem1", None).unwrap();
        store.add("mem2", None).unwrap();

        // Strengthen to create relationship
        store.strengthen(&[1, 2]).unwrap();

        // Add another memory
        store.add("mem3", None).unwrap();

        // Strengthen 2 and 3
        store.strengthen(&[2, 3]).unwrap();

        // Undo last strengthen
        store.undo().unwrap();

        // Undo add of mem3
        store.undo().unwrap();

        // Now undo the first strengthen
        store.undo().unwrap();

        // Undo add of mem2 - should also remove any dangling relationship refs
        store.undo().unwrap();

        // Verify only mem1 exists with no relationships
        let stats = store.stats().unwrap();
        assert_eq!(stats.memory_count, 1);
        assert_eq!(stats.relationship_count, 0);
    }
}
