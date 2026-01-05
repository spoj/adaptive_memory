//! Relationship management and decay calculations.

use rusqlite::{params, Connection, Result};
use serde::Serialize;

use crate::SearchParams;

/// A single relationship event between two memories.
#[derive(Debug, Clone, Serialize)]
pub struct RelationshipEvent {
    pub id: i64,
    pub from_mem: i64,
    pub to_mem: i64,
    pub created_at_mem: i64,
    pub strength: f64,
}

/// Aggregated relationship between two memories (sum of all events).
#[derive(Debug, Clone, Serialize)]
pub struct Relationship {
    pub from_mem: i64,
    pub to_mem: i64,
    /// Raw strength (sum of decayed events). During spreading, ln_1p is applied
    /// for diminishing returns, then normalized across all neighbors.
    pub effective_strength: f64,
    /// Number of strengthening events for this pair.
    pub event_count: usize,
}

/// Result of strengthening relationships.
#[derive(Debug, Serialize)]
pub struct StrengthenResult {
    pub relationships: Vec<Relationship>,
    /// Number of new relationship events created.
    pub event_count: usize,
}

/// Calculate effective strength given stored strength, memory distance, and decay factor.
/// All relationships are now explicit (temporal removed), so no weight multiplier needed.
pub fn calculate_effective_strength(
    stored_strength: f64,
    created_at_mem: i64,
    current_max_mem: i64,
    decay_factor: f64,
) -> f64 {
    let distance = (current_max_mem - created_at_mem) as f64;
    stored_strength * (-distance * decay_factor).exp()
}

/// Canonicalize memory IDs for relationship storage (smaller first).
pub(crate) fn canonicalize(a: i64, b: i64) -> (i64, i64) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Add a new relationship event (used internally).
/// Multiple events between the same pair are allowed and will be summed.
pub(crate) fn add_relationship_event<C: std::ops::Deref<Target = Connection>>(
    conn: &C,
    from_mem: i64,
    to_mem: i64,
    created_at_mem: i64,
    strength: f64,
) -> Result<()> {
    let (from_mem, to_mem) = canonicalize(from_mem, to_mem);

    conn.execute(
        "INSERT INTO relationships (from_mem, to_mem, created_at_mem, strength)
         VALUES (?1, ?2, ?3, ?4)",
        params![from_mem, to_mem, created_at_mem, strength],
    )?;

    Ok(())
}

/// Get aggregated relationship between two memories (order doesn't matter).
/// Returns the sum of effective strengths across all events.
pub(crate) fn get_relationship<C: std::ops::Deref<Target = Connection>>(
    conn: &C,
    a: i64,
    b: i64,
    current_max_mem: i64,
    params: &SearchParams,
) -> Result<Option<Relationship>> {
    let (from_mem, to_mem) = canonicalize(a, b);

    let mut stmt = conn.prepare(
        "SELECT created_at_mem, strength
         FROM relationships
         WHERE from_mem = ?1 AND to_mem = ?2",
    )?;

    let rows = stmt.query_map(rusqlite::params![from_mem, to_mem], |row| {
        let created_at_mem: i64 = row.get(0)?;
        let strength: f64 = row.get(1)?;
        Ok((created_at_mem, strength))
    })?;

    let mut total_effective = 0.0;
    let mut event_count = 0;

    for row in rows {
        let (created_at_mem, strength) = row?;
        total_effective += calculate_effective_strength(
            strength,
            created_at_mem,
            current_max_mem,
            params.decay_factor,
        );
        event_count += 1;
    }

    if event_count > 0 {
        Ok(Some(Relationship {
            from_mem,
            to_mem,
            effective_strength: total_effective,
            event_count,
        }))
    } else {
        Ok(None)
    }
}

/// Get all relationships for a memory (as either from_mem or to_mem).
/// Returns normalized strengths per neighbor (sum to 1.0, like probabilities).
/// This ensures energy is distributed proportionally rather than multiplied.
pub(crate) fn get_relationships_for_memory(
    conn: &Connection,
    mem_id: i64,
    current_max_mem: i64,
    params: &SearchParams,
) -> Result<Vec<(i64, f64)>> {
    use std::collections::HashMap;

    // Single query with UNION ALL to get both directions
    let mut stmt = conn.prepare_cached(
        "SELECT to_mem AS neighbor, created_at_mem, strength
         FROM relationships WHERE from_mem = ?1
         UNION ALL
         SELECT from_mem AS neighbor, created_at_mem, strength
         FROM relationships WHERE to_mem = ?1",
    )?;

    let rows = stmt.query_map(rusqlite::params![mem_id], |row| {
        let neighbor_id: i64 = row.get(0)?;
        let created_at_mem: i64 = row.get(1)?;
        let strength: f64 = row.get(2)?;
        Ok((neighbor_id, created_at_mem, strength))
    })?;

    // Aggregate raw strengths per neighbor (sum of decayed events)
    let mut neighbor_raw: HashMap<i64, f64> = HashMap::new();
    for row in rows {
        let (neighbor_id, created_at_mem, strength) = row?;
        let eff = calculate_effective_strength(
            strength,
            created_at_mem,
            current_max_mem,
            params.decay_factor,
        );
        *neighbor_raw.entry(neighbor_id).or_insert(0.0) += eff;
    }

    // Apply ln_1p for diminishing returns, then normalize
    // ln_1p(x) = ln(1+x), numerically stable for small x
    // This compresses: 1→0.69, 10→2.40, 100→4.62
    let mut neighbor_strengths: HashMap<i64, f64> = neighbor_raw
        .into_iter()
        .map(|(id, raw)| (id, raw.ln_1p()))
        .collect();

    // Normalize: strengths sum to 1.0 (PageRank-style distribution)
    let total_strength: f64 = neighbor_strengths.values().sum();
    if total_strength > 0.0 {
        for strength in neighbor_strengths.values_mut() {
            *strength /= total_strength;
        }
    }

    Ok(neighbor_strengths.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_strength() {
        let decay_factor = 0.03;

        // At distance 0, effective = stored
        let eff = calculate_effective_strength(1.0, 100, 100, decay_factor);
        assert!((eff - 1.0).abs() < 0.001);

        // At distance 10 with decay_factor=0.03: e^(-10*0.03) = e^-0.3 ~= 0.741
        let eff = calculate_effective_strength(1.0, 0, 10, decay_factor);
        assert!((eff - 0.7408).abs() < 0.01);

        // At distance 20 with decay_factor=0.03: e^(-20*0.03) = e^-0.6 ~= 0.549
        let eff = calculate_effective_strength(1.0, 0, 20, decay_factor);
        assert!((eff - 0.5488).abs() < 0.01);
    }

    #[test]
    fn test_canonicalize() {
        assert_eq!(canonicalize(5, 3), (3, 5));
        assert_eq!(canonicalize(3, 5), (3, 5));
    }
}
