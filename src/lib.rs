//! Adaptive Memory System
//!
//! A Personalized PageRank (PPR) based memory system with relationship strength.
//!
//! # Example
//!
//! ```no_run
//! use adaptive_memory::{MemoryStore, MemoryError, SearchParams};
//!
//! fn main() -> Result<(), MemoryError> {
//!     let mut store = MemoryStore::open_in_memory()?;
//!
//!     // Add memories
//!     store.add("First memory about cats", Some("test"))?;
//!     store.add("Second memory about dogs", None)?;
//!
//!     // Search with default params
//!     let results = store.search("cats", &SearchParams::default())?;
//!     for mem in results.memories {
//!         println!("{}: {} (energy: {:.2})", mem.memory.id, mem.memory.text, mem.energy);
//!     }
//!
//!     // Strengthen relationships
//!     store.strengthen(&[1, 2])?;
//!
//!     Ok(())
//! }
//! ```

pub mod db;
pub mod error;
pub mod memory;
pub mod relationship;
pub mod search;
pub mod store;

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration Constants
// ============================================================================
//
// ## PPR Configuration
//
// ### PPR_EPSILON (1e-6)
// Convergence threshold for PPR power iteration. When L1 norm of score change
// falls below this, we've converged.
//
// ### PPR_MAX_ITER (100)
// Maximum PPR iterations. Typical convergence is 20-50 iterations.
//
// ### FTS limit (SearchParams.limit, default 10)
// Only top N BM25 matches become seeds. User-configurable via --limit.
//
// ### Context expansion (--context N)
// For each result, optionally fetch N memories before/after by ID.
// This is like grep -B/-A for temporal context.

/// Convergence threshold for PPR power iteration (L1 norm).
pub const PPR_EPSILON: f64 = 1e-6;

/// Maximum iterations for PPR power iteration.
pub const PPR_MAX_ITER: usize = 100;

/// Maximum number of memories that can be strengthened at once.
pub const MAX_STRENGTHEN_SET: usize = 10;

/// Default number of results to return from search.
pub const DEFAULT_LIMIT: usize = 10;

// ============================================================================
// Runtime Configuration
// ============================================================================

/// Runtime search parameters.
///
/// All parameters are configurable at search time,
/// allowing experimentation without recompiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Maximum number of results to return (also used as seed count).
    pub limit: usize,
    /// PPR damping factor (alpha). Classic PageRank uses 0.85.
    /// Higher values = more weight to graph structure, lower = more weight to seeds.
    pub alpha: f64,
    /// Degree penalty exponent (beta). Penalizes high-degree nodes.
    /// 0.0 = no penalty, 0.5 = sqrt penalty, 1.0 = linear penalty.
    /// Higher values boost unique/rare connections over hub connections.
    pub beta: f64,
    /// Context window: fetch N memories before/after each result (like grep -B/-A).
    /// Set to 0 to disable context expansion.
    pub context: usize,
    /// Filter results to memories with ID >= from (inclusive).
    pub from: Option<i64>,
    /// Filter results to memories with ID <= to (inclusive).
    pub to: Option<i64>,
    /// Decay scale for relationship strength (power law).
    /// At age = decay, strength is halved. 0.0 means no decay.
    pub decay: f64,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            alpha: 0.85,
            beta: 0.5,
            context: 0,
            from: None,
            to: None,
            decay: 0.0,
        }
    }
}

// ============================================================================
// Re-exports
// ============================================================================

pub use db::default_db_path;
pub use error::MemoryError;
pub use memory::{
    AddMemoryResult, GraphStats, Memory, Stats, Timeline, TimelineBucket, TimelineSummary,
};
pub use relationship::{Relationship, RelationshipEvent, StrengthenResult};
pub use search::{ActivatedMemory, RelatedResult, SearchResult};
pub use store::{MemoryStore, UndoResult};
