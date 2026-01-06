//! Relationship management and strength calculations.

use rusqlite::{Connection, Result, params};
use serde::Serialize;

use crate::SearchParams;

/// A single relationship event between two memories.
#[derive(Debug, Clone, Serialize)]
pub struct RelationshipEvent {
    pub id: i64,
    pub from_mem: i64,
    pub to_mem: i64,
    pub strength: f64,
}

/// Aggregated relationship between two memories (sum of all events).
#[derive(Debug, Clone, Serialize)]
pub struct Relationship {
    pub from_mem: i64,
    pub to_mem: i64,
    /// Raw strength (sum of events).
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

/// Result of connecting memories (only creates if no existing connection).
#[derive(Debug, Serialize)]
pub struct ConnectResult {
    /// Relationships that were created (pairs that had no prior connection).
    pub created: Vec<Relationship>,
    /// Pairs that were skipped because they already had a connection.
    pub skipped: Vec<(i64, i64)>,
}

/// Calculate effective strength (simply returns stored strength, no decay).
pub fn calculate_effective_strength(stored_strength: f64) -> f64 {
    stored_strength
}

/// Canonicalize memory IDs for relationship storage (smaller first).
pub(crate) fn canonicalize(a: i64, b: i64) -> (i64, i64) {
    if a < b { (a, b) } else { (b, a) }
}

/// Add a new relationship event (used internally).
/// Multiple events between the same pair are allowed and will be summed.
pub(crate) fn add_relationship_event<C: std::ops::Deref<Target = Connection>>(
    conn: &C,
    from_mem: i64,
    to_mem: i64,
    strength: f64,
) -> Result<()> {
    let (from_mem, to_mem) = canonicalize(from_mem, to_mem);

    conn.execute(
        "INSERT INTO relationships (from_mem, to_mem, strength)
         VALUES (?1, ?2, ?3)",
        params![from_mem, to_mem, strength],
    )?;

    Ok(())
}

/// Check if a relationship exists between two memories (order doesn't matter).
pub(crate) fn relationship_exists<C: std::ops::Deref<Target = Connection>>(
    conn: &C,
    a: i64,
    b: i64,
) -> Result<bool> {
    let (from_mem, to_mem) = canonicalize(a, b);

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM relationships WHERE from_mem = ?1 AND to_mem = ?2",
        rusqlite::params![from_mem, to_mem],
        |row| row.get(0),
    )?;

    Ok(count > 0)
}

/// Get aggregated relationship between two memories (order doesn't matter).
/// Returns the sum of strengths across all events.
pub(crate) fn get_relationship<C: std::ops::Deref<Target = Connection>>(
    conn: &C,
    a: i64,
    b: i64,
) -> Result<Option<Relationship>> {
    let (from_mem, to_mem) = canonicalize(a, b);

    let mut stmt =
        conn.prepare("SELECT strength FROM relationships WHERE from_mem = ?1 AND to_mem = ?2")?;

    let rows = stmt.query_map(rusqlite::params![from_mem, to_mem], |row| {
        let strength: f64 = row.get(0)?;
        Ok(strength)
    })?;

    let mut total_effective = 0.0;
    let mut event_count = 0;

    for row in rows {
        let strength = row?;
        total_effective += calculate_effective_strength(strength);
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
/// Returns strengths transformed via sigmoid: strength / (strength + k).
/// This ensures edges are independent (strengthening one doesn't affect others)
/// and bounded (always < 1.0), preventing energy blowup.
pub(crate) fn get_relationships_for_memory(
    conn: &Connection,
    mem_id: i64,
    params: &SearchParams,
) -> Result<Vec<(i64, f64)>> {
    use std::collections::HashMap;

    // Single query with UNION ALL to get both directions
    let mut stmt = conn.prepare_cached(
        "SELECT to_mem AS neighbor, strength
         FROM relationships WHERE from_mem = ?1
         UNION ALL
         SELECT from_mem AS neighbor, strength
         FROM relationships WHERE to_mem = ?1",
    )?;

    let rows = stmt.query_map(rusqlite::params![mem_id], |row| {
        let neighbor_id: i64 = row.get(0)?;
        let strength: f64 = row.get(1)?;
        Ok((neighbor_id, strength))
    })?;

    // Aggregate raw strengths per neighbor (sum of events)
    let mut neighbor_raw: HashMap<i64, f64> = HashMap::new();
    for row in rows {
        let (neighbor_id, strength) = row?;
        let eff = calculate_effective_strength(strength);
        *neighbor_raw.entry(neighbor_id).or_insert(0.0) += eff;
    }

    // Sigmoid transform: strength / (strength + k)
    // - Edges are independent (strengthening one doesn't affect others)
    // - Bounded: always < 1.0, so decay * sigmoid < decay < 1 (no blowup)
    // - k=1: strength=1 gives 0.5, strength=10 gives 0.91
    let k = params.sigmoid_k;
    Ok(neighbor_raw
        .into_iter()
        .map(|(id, raw)| (id, raw / (raw + k)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_strength() {
        // effective_strength just returns the input (no decay)
        let eff = calculate_effective_strength(1.0);
        assert!((eff - 1.0).abs() < 0.001);

        let eff = calculate_effective_strength(5.0);
        assert!((eff - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_canonicalize() {
        assert_eq!(canonicalize(5, 3), (3, 5));
        assert_eq!(canonicalize(3, 5), (3, 5));
    }
}
