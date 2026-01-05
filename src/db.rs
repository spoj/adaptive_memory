//! Database initialization and low-level helpers.

use rusqlite::{Connection, Result};
use std::path::PathBuf;

/// Get the default database path (~/.adaptive_memory.db).
pub fn default_db_path() -> PathBuf {
    dirs::home_dir()
        .expect("Could not find home directory")
        .join(".adaptive_memory.db")
}

/// Initialize the database schema.
pub(crate) fn init_schema(conn: &Connection) -> Result<()> {
    // Enable foreign keys
    conn.execute_batch("PRAGMA foreign_keys = ON;")?;

    // Create memories table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            datetime TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT
        )",
        [],
    )?;

    // Create FTS5 virtual table for full-text search
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            text,
            content=memories,
            content_rowid=id
        )",
        [],
    )?;

    // Create triggers to keep FTS in sync
    conn.execute_batch(
        "
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
        END;
        ",
    )?;

    // Create relationships table (event log style - multiple rows per pair allowed)
    // from_mem < to_mem enforces canonical ordering for symmetric relationships
    // Each row represents a strengthening event; effective strength is sum of decayed rows
    conn.execute(
        "CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY,
            from_mem INTEGER NOT NULL,
            to_mem INTEGER NOT NULL,
            created_at_mem INTEGER NOT NULL,
            strength REAL NOT NULL,
            CHECK (from_mem < to_mem),
            FOREIGN KEY (from_mem) REFERENCES memories(id),
            FOREIGN KEY (to_mem) REFERENCES memories(id)
        )",
        [],
    )?;

    // Create indexes for efficient neighbor lookups
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_mem)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_mem)",
        [],
    )?;

    Ok(())
}

/// Get the current maximum memory ID (used for decay calculations).
pub(crate) fn get_max_memory_id(conn: &Connection) -> Result<i64> {
    conn.query_row("SELECT COALESCE(MAX(id), 0) FROM memories", [], |row| {
        row.get(0)
    })
}
