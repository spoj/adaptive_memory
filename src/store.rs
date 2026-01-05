//! MemoryStore - the main API for the adaptive memory system.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};

use crate::db::{get_max_memory_id, init_schema};
use crate::error::MemoryError;
use crate::memory::{AddMemoryResult, Memory, Stats};
use crate::relationship::{
    add_relationship_event, canonicalize, get_relationship, relationship_exists, ConnectResult,
    StrengthenResult,
};
use crate::search::{surface_candidates, SearchResult};
use crate::{SearchParams, MAX_STRENGTHEN_SET};

/// The main interface for the adaptive memory system.
///
/// Wraps a SQLite connection and provides methods for adding, searching,
/// and strengthening memories. Caches the maximum memory ID for efficient
/// decay calculations.
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

        // Insert the new memory (no temporal relationships created)
        self.conn.execute(
            "INSERT INTO memories (datetime, text, source) VALUES (?1, ?2, ?3)",
            params![datetime_str_to_store, text, source],
        )?;

        let new_id = self.conn.last_insert_rowid();

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
        surface_candidates(&self.conn, query, params, self.cached_max_mem)
    }

    /// Strengthen relationships between a set of memory IDs.
    ///
    /// Adds a new explicit relationship event for each pair.
    /// The strength per pair is distributed as 1.0 / num_pairs,
    /// so strengthening more memories at once distributes the same total strength.
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
        let mut event_count = 0;

        // Use default params for reading back relationships
        // (the effective_strength shown is just for display, doesn't affect storage)
        let default_params = SearchParams::default();

        // Generate all pairs and add a new event for each (1.0 strength per pair)
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (from_mem, to_mem) = canonicalize(ids[i], ids[j]);

                // Add new relationship event with 1.0 strength
                add_relationship_event(&tx, from_mem, to_mem, self.cached_max_mem, 1.0)?;
                event_count += 1;

                // Get the aggregated relationship (including the new event)
                if let Some(rel) =
                    get_relationship(&tx, from_mem, to_mem, self.cached_max_mem, &default_params)?
                {
                    relationships.push(rel);
                }
            }
        }

        tx.commit()?;

        Ok(StrengthenResult {
            relationships,
            event_count,
        })
    }

    /// Connect memories that don't already have a relationship.
    ///
    /// Unlike strengthen(), this only creates a relationship if none exists.
    /// Each new connection gets strength 1.0. Existing connections are skipped.
    ///
    /// This operation is wrapped in a transaction for atomicity.
    pub fn connect(&mut self, ids: &[i64]) -> Result<ConnectResult, MemoryError> {
        if ids.len() > MAX_STRENGTHEN_SET {
            return Err(MemoryError::InvalidInput(format!(
                "Cannot connect more than {} memories at once (got {})",
                MAX_STRENGTHEN_SET,
                ids.len()
            )));
        }

        if ids.len() < 2 {
            return Err(MemoryError::InvalidInput(
                "At least two memory IDs are required to create connections".to_string(),
            ));
        }

        let tx = self.conn.transaction()?;

        let mut created = Vec::new();
        let mut skipped = Vec::new();

        let default_params = SearchParams::default();

        // Generate all pairs and only add if no existing connection
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (from_mem, to_mem) = canonicalize(ids[i], ids[j]);

                if relationship_exists(&tx, from_mem, to_mem)? {
                    skipped.push((from_mem, to_mem));
                } else {
                    // Create new relationship with strength 1.0
                    add_relationship_event(&tx, from_mem, to_mem, self.cached_max_mem, 1.0)?;

                    if let Some(rel) = get_relationship(
                        &tx,
                        from_mem,
                        to_mem,
                        self.cached_max_mem,
                        &default_params,
                    )? {
                        created.push(rel);
                    }
                }
            }
        }

        tx.commit()?;

        Ok(ConnectResult { created, skipped })
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
        let rows = stmt.query_map(param_refs.as_slice(), |row| Self::row_to_memory(row))?;

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

        let rows = stmt.query_map(params![limit as i64], |row| Self::row_to_memory(row))?;

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

        Ok(Stats {
            memory_count,
            min_memory_id,
            max_memory_id,
            relationship_count,
            relationship_event_count,
            unique_sources: sources,
        })
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
        assert_eq!(result.event_count, 3); // 3 pairs: (1,2), (1,3), (2,3)
        assert_eq!(result.relationships.len(), 3);

        // Each relationship should have 1 event with strength 1.0
        for rel in &result.relationships {
            assert_eq!(rel.event_count, 1);
            assert!((rel.effective_strength - 1.0).abs() < 0.001);
        }

        // Strengthen just 2 memories again - adds another 1.0 event
        let result = store.strengthen(&[1, 2]).unwrap();
        assert_eq!(result.event_count, 1);
        assert_eq!(result.relationships.len(), 1);
        assert_eq!(result.relationships[0].event_count, 2); // 2 events now
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

        // Add relationships
        store.strengthen(&[1, 2]).unwrap();
        store.strengthen(&[1, 2, 3]).unwrap(); // adds events for (1,2), (1,3), (2,3)

        let stats = store.stats().unwrap();
        assert_eq!(stats.relationship_count, 3); // unique pairs: (1,2), (1,3), (2,3)
        assert_eq!(stats.relationship_event_count, 4); // 1 + 3 events
    }

    #[test]
    fn test_connect() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        store.add("mem1", None).unwrap();
        store.add("mem2", None).unwrap();
        store.add("mem3", None).unwrap();
        store.add("mem4", None).unwrap();

        // Connect 1 and 2 - should create relationship
        let result = store.connect(&[1, 2]).unwrap();
        assert_eq!(result.created.len(), 1);
        assert!(result.skipped.is_empty());
        assert_eq!(result.created[0].from_mem, 1);
        assert_eq!(result.created[0].to_mem, 2);

        // Try to connect 1 and 2 again - should skip
        let result = store.connect(&[1, 2]).unwrap();
        assert!(result.created.is_empty());
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(result.skipped[0], (1, 2));

        // Connect 1, 2, 3 - (1,2) should skip, (1,3) and (2,3) should create
        let result = store.connect(&[1, 2, 3]).unwrap();
        assert_eq!(result.created.len(), 2);
        assert_eq!(result.skipped.len(), 1);
        assert_eq!(result.skipped[0], (1, 2));

        // Verify stats
        let stats = store.stats().unwrap();
        assert_eq!(stats.relationship_count, 3); // (1,2), (1,3), (2,3)
        assert_eq!(stats.relationship_event_count, 3); // only 1 event each (unlike strengthen)
    }

    #[test]
    fn test_connect_validation() {
        let mut store = MemoryStore::open_in_memory().unwrap();

        // Single ID should fail
        let err = store.connect(&[1]).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));

        // Empty list should fail
        let err = store.connect(&[]).unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }
}
