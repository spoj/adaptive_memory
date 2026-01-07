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

/// Database statistics.
#[derive(Debug, Serialize)]
pub struct Stats {
    pub memory_count: i64,
    pub min_memory_id: Option<i64>,
    pub max_memory_id: Option<i64>,
    pub relationship_count: i64,
    pub relationship_event_count: i64,
    pub unique_sources: Vec<String>,
    /// Graph metrics
    pub graph: GraphStats,
}

/// Graph-oriented statistics.
#[derive(Debug, Serialize)]
pub struct GraphStats {
    /// Memories with no connections
    pub stray_count: i64,
    /// Number of connected components (islands)
    pub island_count: i64,
    /// Size of the largest connected component
    pub largest_island_size: i64,
    /// Memories with only one connection
    pub leaf_count: i64,
    /// Maximum degree (most connections on a single memory)
    pub max_degree: i64,
    /// Average degree of connected memories (0 if none)
    pub avg_degree: f64,
}

/// A single day's worth of memories in the timeline.
#[derive(Debug, Serialize)]
pub struct TimelineBucket {
    /// Date in YYYY-MM-DD format
    pub date: String,
    /// First (lowest) memory ID on this date
    pub min_id: i64,
    /// Last (highest) memory ID on this date
    pub max_id: i64,
    /// Number of memories on this date
    pub count: i64,
}

/// Timeline showing memory ID distribution by date.
#[derive(Debug, Serialize)]
pub struct Timeline {
    /// Daily buckets, ordered by date descending (most recent first)
    pub buckets: Vec<TimelineBucket>,
    /// Summary statistics
    pub summary: TimelineSummary,
}

/// Summary statistics for the timeline.
#[derive(Debug, Serialize)]
pub struct TimelineSummary {
    /// Total number of memories
    pub total_memories: i64,
    /// Oldest memory date (YYYY-MM-DD)
    pub oldest_date: Option<String>,
    /// Newest memory date (YYYY-MM-DD)
    pub newest_date: Option<String>,
    /// Oldest memory ID
    pub oldest_id: Option<i64>,
    /// Newest memory ID
    pub newest_id: Option<i64>,
    /// Number of days with memories
    pub total_days: usize,
    /// Average memories per day
    pub avg_per_day: f64,
}
