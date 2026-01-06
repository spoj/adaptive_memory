//! Search functionality with spreading activation.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rusqlite::{Connection, params};
use serde::Serialize;

use crate::error::MemoryError;
use crate::memory::Memory;
use crate::relationship::get_relationships_for_memory;
use crate::{ENERGY_THRESHOLD, MAX_SPREADING_ITERATIONS, SearchParams};

/// A memory with its activation energy.
#[derive(Debug, Serialize)]
pub struct ActivatedMemory {
    #[serde(flatten)]
    pub memory: Memory,
    /// Energy from spreading activation. 0.0 for context items.
    pub energy: f64,
    /// True if this is a context item (fetched via --context, not by relevance).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_context: bool,
}

/// Result of surface_candidates search.
#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub query: String,
    pub seed_count: usize,
    pub total_activated: usize,
    /// Number of spreading activation iterations performed.
    pub iterations: usize,
    /// Results sorted by energy score (highest first).
    pub memories: Vec<ActivatedMemory>,
}

/// Item for the priority queue (max-heap by energy).
#[derive(Debug, Clone)]
struct ActivationItem {
    energy: f64,
    mem_id: i64,
}

impl PartialEq for ActivationItem {
    fn eq(&self, other: &Self) -> bool {
        self.mem_id == other.mem_id
    }
}

impl Eq for ActivationItem {}

impl PartialOrd for ActivationItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ActivationItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for max-heap (higher energy first)
        self.energy
            .partial_cmp(&other.energy)
            .unwrap_or(Ordering::Equal)
    }
}

/// Perform FTS5 search and return matching memory IDs with BM25 scores.
/// Returns (id, normalized_score) where score is normalized to [0.1, 1.0] range.
/// Results are ordered by BM25 relevance (best match first).
/// Returns error if query is empty.
fn fts_search(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(i64, f64)>, MemoryError> {
    if query.trim().is_empty() {
        return Err(MemoryError::InvalidInput(
            "Search query cannot be empty".to_string(),
        ));
    }

    // FTS5 search with BM25 ranking
    // Note: BM25 returns negative scores (more negative = better match)
    let mut stmt = conn.prepare(
        "SELECT rowid, bm25(memories_fts) FROM memories_fts
         WHERE memories_fts MATCH ?1
         ORDER BY bm25(memories_fts) 
         LIMIT ?2",
    )?;

    let raw_results: Vec<(i64, f64)> = stmt
        .query_map(params![query, limit as i64], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    if raw_results.is_empty() {
        return Ok(vec![]);
    }

    // Normalize BM25 scores to [0.1, 1.0] range
    // BM25 scores are negative, with more negative being better
    let min_score = raw_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::INFINITY, f64::min);
    let max_score = raw_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    let results: Vec<(i64, f64)> = if (max_score - min_score).abs() < 1e-9 {
        // All scores are the same, use 1.0 for all
        raw_results.into_iter().map(|(id, _)| (id, 1.0)).collect()
    } else {
        // Normalize: best (most negative) -> 1.0, worst (least negative) -> 0.1
        // We use 0.1 as floor so even worst matches get some energy
        raw_results
            .into_iter()
            .map(|(id, score)| {
                let normalized = (max_score - score) / (max_score - min_score);
                let scaled = 0.1 + 0.9 * normalized; // Map to [0.1, 1.0]
                (id, scaled)
            })
            .collect()
    };

    Ok(results)
}

/// Get multiple memories by their IDs (internal helper).
fn get_memories_by_ids(conn: &Connection, ids: &[i64]) -> Result<Vec<Memory>, MemoryError> {
    use chrono::{DateTime, Utc};

    if ids.is_empty() {
        return Ok(vec![]);
    }

    let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let query = format!(
        "SELECT id, datetime, text, source FROM memories WHERE id IN ({})",
        placeholders
    );

    let mut stmt = conn.prepare(&query)?;

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

/// Surface candidate memories using text search + spreading activation.
pub(crate) fn surface_candidates(
    conn: &Connection,
    query: &str,
    params: &SearchParams,
) -> Result<SearchResult, MemoryError> {
    let limit = params.limit;

    // Step 1: FTS5 search to get initial candidates with BM25 scores
    // Seeds are selected purely by BM25 relevance (no recency re-ranking)
    let seeds = fts_search(conn, query, limit)?;

    // No padding with recent memories - only use FTS matches as seeds

    let seed_count = seeds.len();

    // Step 3: Spreading activation with delta propagation
    //
    // This implements theoretically correct energy superposition:
    // - Energy accumulates from ALL paths to a node
    // - We only propagate the DELTA (new energy) to avoid redundant work
    // - This gives correct results while being efficient
    let mut energy_map: HashMap<i64, f64> = HashMap::new();
    let mut heap: BinaryHeap<ActivationItem> = BinaryHeap::new();
    // Track how much energy we've already propagated FROM each node
    let mut propagated_energy: HashMap<i64, f64> = HashMap::new();

    // Initialize seeds with BM25-weighted energy
    for (seed_id, bm25_score) in &seeds {
        heap.push(ActivationItem {
            energy: *bm25_score,
            mem_id: *seed_id,
        });
    }

    // Process activation spread with iteration cap
    let mut iterations = 0;
    while let Some(item) = heap.pop() {
        iterations += 1;
        if iterations > MAX_SPREADING_ITERATIONS {
            break;
        }

        // Accumulate energy for this memory
        let total = energy_map.entry(item.mem_id).or_insert(0.0);
        *total += item.energy;

        // Calculate how much NEW energy we have to propagate
        let already_propagated = propagated_energy.get(&item.mem_id).copied().unwrap_or(0.0);
        let to_propagate = *total - already_propagated;

        // Only propagate if there's meaningful new energy
        if to_propagate > ENERGY_THRESHOLD {
            propagated_energy.insert(item.mem_id, *total);

            // Get neighbors with their effective strengths (using runtime weights)
            let neighbors = get_relationships_for_memory(conn, item.mem_id, params)?;

            for (neighbor_id, effective_strength) in neighbors {
                let new_energy = to_propagate * params.energy_decay * effective_strength;

                if new_energy > ENERGY_THRESHOLD {
                    heap.push(ActivationItem {
                        energy: new_energy,
                        mem_id: neighbor_id,
                    });
                }
            }
        }
    }

    // Step 4: Filter by ID range, sort by energy, and take top `limit`
    let mut activated: Vec<(i64, f64)> = energy_map
        .into_iter()
        .filter(|(id, _)| {
            let above_from = params.from.map_or(true, |from| *id >= from);
            let below_to = params.to.map_or(true, |to| *id <= to);
            above_from && below_to
        })
        .collect();
    activated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    activated.truncate(limit);

    let total_activated = activated.len();

    // Step 5: Expand context if requested
    let activated_ids: HashSet<i64> = activated.iter().map(|(id, _)| *id).collect();
    let mut context_ids: HashSet<i64> = HashSet::new();

    if params.context > 0 {
        for &(id, _) in &activated {
            // Add IDs within context window (both before and after)
            let start = (id - params.context as i64).max(1);
            let end = id + params.context as i64;
            for ctx_id in start..=end {
                if ctx_id != id && !activated_ids.contains(&ctx_id) {
                    context_ids.insert(ctx_id);
                }
            }
        }
    }

    // Step 6: Fetch all memory objects (activated + context)
    let mut all_ids: Vec<i64> = activated.iter().map(|(id, _)| *id).collect();
    all_ids.extend(context_ids.iter());
    let memories = get_memories_by_ids(conn, &all_ids)?;

    // Create a map for quick lookup
    let memory_map: HashMap<i64, Memory> = memories.into_iter().map(|m| (m.id, m)).collect();
    let energy_map: HashMap<i64, f64> = activated.into_iter().collect();

    // Build result: activated items with energy, context items with energy=0
    let mut results: Vec<ActivatedMemory> = Vec::new();

    for id in all_ids {
        if let Some(mem) = memory_map.get(&id) {
            let (energy, is_context) = if let Some(&e) = energy_map.get(&id) {
                (e, false)
            } else {
                (0.0, true)
            };
            results.push(ActivatedMemory {
                memory: mem.clone(),
                energy,
                is_context,
            });
        }
    }

    // Sort by energy score (highest first)
    results.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(Ordering::Equal));

    Ok(SearchResult {
        query: query.to_string(),
        seed_count,
        total_activated,
        iterations,
        memories: results,
    })
}
