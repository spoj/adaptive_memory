//! Custom error types for the adaptive memory system.

use std::fmt;

/// Errors that can occur in the adaptive memory system.
#[derive(Debug)]
pub enum MemoryError {
    /// Database-related errors (SQLite).
    Database(rusqlite::Error),
    /// Invalid input provided by the caller.
    InvalidInput(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::Database(e) => write!(f, "Database error: {}", e),
            MemoryError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MemoryError::Database(e) => Some(e),
            MemoryError::InvalidInput(_) => None,
        }
    }
}

impl From<rusqlite::Error> for MemoryError {
    fn from(e: rusqlite::Error) -> Self {
        MemoryError::Database(e)
    }
}
