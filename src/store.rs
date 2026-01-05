//! MemoryStore - the main API for the adaptive memory system.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};

use crate::db::{get_max_memory_id, init_schema};
use crate::error::MemoryError;
use crate::memory::{AddMemoryResult, Memory};
use crate::relationship::{
    add_relationship_event, canonicalize, get_relationship, StrengthenResult,
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
}
