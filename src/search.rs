//! Search functionality with Personalized PageRank (PPR).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, params};
use serde::Serialize;

use crate::error::MemoryError;
use crate::memory::Memory;
use crate::{PPR_EPSILON, PPR_MAX_ITER, SearchParams};

/// A memory with its PPR score.
#[derive(Debug, Serialize)]
pub struct ActivatedMemory {
    #[serde(flatten)]
    pub memory: Memory,
    /// PPR score. 0.0 for context items.
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
    /// Number of PPR iterations performed.
    pub iterations: usize,
    /// Results sorted by PPR score (highest first).
    pub memories: Vec<ActivatedMemory>,
}

/// In-memory graph representation for PPR computation.
/// Loaded once per search, used for all iterations.
struct Graph {
    /// Adjacency list: node_id -> Vec<(neighbor_id, edge_weight)>
    edges: HashMap<i64, Vec<(i64, f64)>>,
    /// Out-degree (sum of edge weights) for each node
    out_weight: HashMap<i64, f64>,
    /// Number of connections (edge count) for each node
    degree: HashMap<i64, usize>,
}

impl Graph {
    /// Load entire graph from database.
    fn load(conn: &Connection) -> Result<Self, MemoryError> {
        let mut edges: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
        let mut out_weight: HashMap<i64, f64> = HashMap::new();
        let mut degree: HashMap<i64, usize> = HashMap::new();

        // Load all relationships, aggregate by (from, to) pair
        let mut stmt = conn.prepare(
            "SELECT from_mem, to_mem, SUM(strength) as total_strength
             FROM relationships
             GROUP BY from_mem, to_mem",
        )?;

        let rows = stmt.query_map([], |row| {
            let from: i64 = row.get(0)?;
            let to: i64 = row.get(1)?;
            let strength: f64 = row.get(2)?;
            Ok((from, to, strength))
        })?;

        for row in rows {
            let (from, to, strength) = row?;

            // Bidirectional: add edge in both directions
            edges.entry(from).or_default().push((to, strength));
            edges.entry(to).or_default().push((from, strength));

            // Track out-weight for both directions
            *out_weight.entry(from).or_default() += strength;
            *out_weight.entry(to).or_default() += strength;

            // Track degree (number of connections)
            *degree.entry(from).or_default() += 1;
            *degree.entry(to).or_default() += 1;
        }

        Ok(Self {
            edges,
            out_weight,
            degree,
        })
    }

    /// Get neighbors and their normalized transition probabilities with degree penalty.
    /// Returns (neighbor_id, weight / total_out_weight, neighbor_degree).
    fn neighbors(&self, node_id: i64) -> impl Iterator<Item = (i64, f64, usize)> + '_ {
        let total = self.out_weight.get(&node_id).copied().unwrap_or(0.0);
        self.edges
            .get(&node_id)
            .into_iter()
            .flatten()
            .map(move |&(neighbor, weight)| {
                let prob = if total > 0.0 { weight / total } else { 0.0 };
                let neighbor_degree = self.degree.get(&neighbor).copied().unwrap_or(1);
                (neighbor, prob, neighbor_degree)
            })
    }

    /// Check if a node is dangling (no outgoing edges).
    fn is_dangling(&self, node_id: i64) -> bool {
        self.out_weight.get(&node_id).copied().unwrap_or(0.0) == 0.0
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

/// Personalized PageRank power iteration with degree penalty.
///
/// Formula: score = (1 - α) * seed + α * P * score
/// Where P is the transition matrix (normalized edge weights).
///
/// Beta controls degree penalty: contribution is divided by neighbor_degree^beta.
/// This boosts unique/rare connections over high-degree hub connections.
///
/// Dangling nodes (no outgoing edges) teleport their score back to seeds.
fn ppr(graph: &Graph, seeds: &[(i64, f64)], alpha: f64, beta: f64) -> (HashMap<i64, f64>, usize) {
    // Normalize seed scores to sum to 1.0
    let seed_total: f64 = seeds.iter().map(|(_, s)| s).sum();
    if seed_total == 0.0 {
        return (HashMap::new(), 0);
    }

    let seed_map: HashMap<i64, f64> = seeds.iter().map(|(id, s)| (*id, s / seed_total)).collect();

    let mut scores = seed_map.clone();
    let mut iterations = 0;

    for iter in 0..PPR_MAX_ITER {
        iterations = iter + 1;
        let mut new_scores: HashMap<i64, f64> = HashMap::new();

        // Teleport component: (1 - α) * seed_score for each seed
        for (id, seed_score) in &seed_map {
            *new_scores.entry(*id).or_insert(0.0) += (1.0 - alpha) * seed_score;
        }

        // Track dangling node score (nodes with no outgoing edges)
        let mut dangling_sum = 0.0;

        // Propagate: α * Σ(score[node] * transition_prob / neighbor_degree^beta)
        for (&node_id, &score) in &scores {
            if graph.is_dangling(node_id) {
                // Dangling node: accumulate score for redistribution to seeds
                dangling_sum += score;
            } else {
                // Normal node: distribute score to neighbors with degree penalty
                for (neighbor_id, prob, neighbor_degree) in graph.neighbors(node_id) {
                    // Penalize high-degree neighbors: divide by degree^beta
                    let degree_penalty = (neighbor_degree as f64).powf(beta);
                    let contribution = alpha * score * prob / degree_penalty;
                    *new_scores.entry(neighbor_id).or_insert(0.0) += contribution;
                }
            }
        }

        // Redistribute dangling score to seeds (proportional to seed weights)
        // This is equivalent to dangling nodes teleporting back to seeds
        for (id, seed_score) in &seed_map {
            *new_scores.entry(*id).or_insert(0.0) += alpha * dangling_sum * seed_score;
        }

        // Renormalize to maintain probability distribution (since degree penalty breaks it)
        let total: f64 = new_scores.values().sum();
        if total > 0.0 {
            for score in new_scores.values_mut() {
                *score /= total;
            }
        }

        // Check convergence (L1 norm of change)
        let diff: f64 = new_scores
            .iter()
            .map(|(id, &s)| (s - scores.get(id).unwrap_or(&0.0)).abs())
            .sum::<f64>()
            + scores
                .iter()
                .filter(|(id, _)| !new_scores.contains_key(id))
                .map(|(_, &s)| s)
                .sum::<f64>();

        scores = new_scores;

        if diff < PPR_EPSILON {
            break;
        }
    }

    (scores, iterations)
}

/// Surface candidate memories using text search + Personalized PageRank.
pub(crate) fn surface_candidates(
    conn: &Connection,
    query: &str,
    params: &SearchParams,
) -> Result<SearchResult, MemoryError> {
    let limit = params.limit;

    // Step 1: FTS5 search to get initial candidates with BM25 scores
    let seeds = fts_search(conn, query, limit)?;
    let seed_count = seeds.len();

    // Step 2: Load graph into memory
    let graph = Graph::load(conn)?;

    // Step 3: Run PPR with degree penalty
    let (scores, iterations) = ppr(&graph, &seeds, params.alpha, params.beta);

    // Step 4: Filter by ID range, sort by score, and take top `limit`
    let mut activated: Vec<(i64, f64)> = scores
        .into_iter()
        .filter(|(id, _)| {
            let above_from = params.from.is_none_or(|from| *id >= from);
            let below_to = params.to.is_none_or(|to| *id <= to);
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
    let score_map: HashMap<i64, f64> = activated.into_iter().collect();

    // Build result: activated items with score, context items with score=0
    let mut results: Vec<ActivatedMemory> = Vec::new();

    for id in all_ids {
        if let Some(mem) = memory_map.get(&id) {
            let (energy, is_context) = if let Some(&e) = score_map.get(&id) {
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

    // Sort by score (highest first)
    results.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(Ordering::Equal));

    Ok(SearchResult {
        query: query.to_string(),
        seed_count,
        total_activated,
        iterations,
        memories: results,
    })
}
