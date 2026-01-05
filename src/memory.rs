//! Memory data structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: i64,
    pub datetime: DateTime<Utc>,
    pub text: String,
    pub source: Option<String>,
}

/// Result of adding a memory.
#[derive(Debug, Serialize)]
pub struct AddMemoryResult {
    pub memory: Memory,
}
