//! Adaptive Memory System
//!
//! A spreading activation memory system with decay-based relationship strength.
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
// ## Retrieval Bounds Documentation
//
// These constants create artificial bounds during retrieval. Analysis of why each is acceptable:
//
// ### ENERGY_THRESHOLD (0.01)
// Energy below 1% is not propagated. With energy_decay=0.5, this means ~7 hops max depth.
// Delta propagation accumulates all incoming energy before checking threshold, so weak paths
// that converge are handled correctly. Natural convergence criterion.
//
// ### MAX_SPREADING_ITERATIONS (5000)
// Safety cap for dense graphs. We process highest-energy items first (max-heap), so most
// relevant are processed even if cap is hit. Output includes `iterations` count so callers
// know if cap was reached. Could truncate results on very dense graphs (80k+ relationships).
//
// ### FTS limit (SearchParams.limit, default 100)
// Only top N BM25 matches become seeds. User-configurable via --limit. Standard practice.
//
// ### BM25 normalization floor (0.1 in search.rs)
// Worst FTS match gets 10% seed energy, best gets 100%. Ensures all matches contribute.
//
// ### Result truncation (SearchParams.limit)
// Final results truncated to limit after sorting by energy. Expected behavior.
//
// ### Bidirectional relationships and loops
// Relationships are symmetric (A↔B). Energy can bounce A→B→A but delta propagation handles
// this correctly - each round trip decays by (energy_decay * strength)^2, converging in
// ~4-5 iterations for a simple loop. ENERGY_THRESHOLD catches convergence naturally.
// MAX_SPREADING_ITERATIONS is safety valve, not loop prevention.
//
// ### Context expansion (--context N)
// Instead of pre-computed temporal relationships, context is fetched at query time.
// For each result, we can optionally fetch N memories before/after by ID.
// This is like grep -B/-A for temporal context.

/// Minimum energy threshold to continue propagation.
/// At 0.01 (1% of initial seed energy), with energy_decay=0.5, this allows ~7 hops depth.
pub const ENERGY_THRESHOLD: f64 = 0.01;

/// Maximum iterations for spreading activation (prevents runaway on dense graphs).
/// With delta propagation, typical searches use 50-300 iterations. This is a safety cap.
/// If hit, highest-energy paths are still processed (max-heap ordering).
pub const MAX_SPREADING_ITERATIONS: usize = 5000;

/// Maximum number of memories that can be strengthened at once.
pub const MAX_STRENGTHEN_SET: usize = 10;

/// Default number of results to return from search.
pub const DEFAULT_LIMIT: usize = 100;

// ============================================================================
// Runtime Configuration
// ============================================================================

/// Runtime search parameters.
///
/// All weights and decay factors are configurable at search time,
/// allowing experimentation without recompiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    /// Maximum number of results to return (also used as seed count).
    pub limit: usize,
    /// Decay factor for relationship strength over memory distance.
    /// Higher = faster decay. With 0.03: ~74% at 10 memories, ~22% at 50.
    pub decay_factor: f64,
    /// Energy decay per hop during spreading activation.
    /// 0.5 means energy halves each hop.
    pub energy_decay: f64,
    /// Context window: fetch N memories before/after each result (like grep -B/-A).
    /// Set to 0 to disable context expansion.
    pub context: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            decay_factor: 0.0,
            energy_decay: 0.5,
            context: 0,
        }
    }
}

// ============================================================================
// Re-exports
// ============================================================================

pub use db::default_db_path;
pub use error::MemoryError;
pub use memory::{AddMemoryResult, Memory};
pub use relationship::{Relationship, RelationshipEvent, StrengthenResult};
pub use search::{ActivatedMemory, SearchResult};
pub use store::MemoryStore;
