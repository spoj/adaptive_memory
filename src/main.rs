use std::path::PathBuf;
use std::process;

use chrono::{Local, TimeZone};
use clap::{Parser, Subcommand, ValueEnum};

use adaptive_memory::{
    DEFAULT_LIMIT, MAX_STRENGTHEN_SET, MemoryError, MemoryStore, SearchParams, default_db_path,
};

/// Timezone for datetime display
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum TzOption {
    /// Use local timezone
    #[default]
    Local,
    /// Use UTC (Zulu time)
    Utc,
}

#[derive(Parser)]
#[command(name = "adaptive-memory")]
#[command(about = "Adaptive memory system with Personalized PageRank", long_about = None)]
struct Cli {
    /// Path to the database file (default: ~/.adaptive_memory.db)
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    /// Output in JSON format (default is compact text)
    #[arg(long, global = true)]
    json: bool,

    /// Timezone for datetime display (default: local for text, utc for JSON)
    #[arg(long, global = true, value_enum)]
    tz: Option<TzOption>,

    #[command(subcommand)]
    command: Option<Commands>,

    /// Quick access: ID, IDs (1,2,3), range (1-10), or with + suffix for related
    #[arg(value_name = "SELECTOR")]
    selector: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the database (creates if not exists)
    Init,

    /// Add a new memory
    Add {
        /// The memory text
        content: String,

        /// Optional source identifier
        #[arg(short, long)]
        source: Option<String>,

        /// Optional datetime override (RFC3339 format, e.g. "2024-01-15T10:30:00Z")
        #[arg(short, long)]
        datetime: Option<String>,
    },

    /// Undo the last operation (add or strengthen)
    Undo,

    /// Search for memories using text query and Personalized PageRank
    Search {
        /// Search query (required, cannot be empty)
        query: String,

        /// Maximum number of results to return (also used as seed count)
        #[arg(short, long, default_value_t = DEFAULT_LIMIT)]
        limit: usize,

        /// PPR damping factor (0.85 = classic PageRank, lower = more weight to text matches)
        #[arg(short, long, default_value_t = 0.85)]
        alpha: f64,

        /// Degree penalty (0 = none, 0.5 = sqrt, 1.0 = linear). Boosts unique links over hubs.
        #[arg(short, long, default_value_t = 0.5)]
        beta: f64,

        /// Context window: fetch N memories before/after each result (like grep -B/-A)
        #[arg(short, long, default_value_t = 0)]
        context: usize,

        /// Filter results to memories with ID >= from (inclusive)
        #[arg(long)]
        from: Option<i64>,

        /// Filter results to memories with ID <= to (inclusive)
        #[arg(long)]
        to: Option<i64>,

        /// Decay scale for relationship strength (age at which strength halves). 0 = no decay.
        #[arg(long, default_value_t = 0.0)]
        decay: f64,
    },

    /// Find memories related to seed IDs via graph (skips text search)
    Related {
        /// Comma-separated list of seed memory IDs
        ids: String,

        /// Maximum number of results to return
        #[arg(short, long, default_value_t = DEFAULT_LIMIT)]
        limit: usize,

        /// PPR damping factor (0.85 = classic PageRank, lower = more weight to seeds)
        #[arg(short, long, default_value_t = 0.85)]
        alpha: f64,

        /// Degree penalty (0 = none, 0.5 = sqrt, 1.0 = linear). Boosts unique links over hubs.
        #[arg(short, long, default_value_t = 0.5)]
        beta: f64,

        /// Context window: fetch N memories before/after each result (like grep -B/-A)
        #[arg(short, long, default_value_t = 0)]
        context: usize,

        /// Filter results to memories with ID >= from (inclusive)
        #[arg(long)]
        from: Option<i64>,

        /// Filter results to memories with ID <= to (inclusive)
        #[arg(long)]
        to: Option<i64>,

        /// Decay scale for relationship strength (age at which strength halves). 0 = no decay.
        #[arg(long, default_value_t = 0.0)]
        decay: f64,
    },

    /// Strengthen relationships between memories (always adds, use undo to reverse)
    Strengthen {
        /// Comma-separated list of memory IDs (max 10)
        ids: String,
    },

    /// Show the latest N memories (shorthand for list --limit N)
    Tail {
        /// Number of memories to show (default: 10)
        #[arg(default_value_t = 10)]
        n: usize,
    },

    /// List memories by ID range
    List {
        /// Start ID (inclusive)
        #[arg(long)]
        from: Option<i64>,

        /// End ID (inclusive)
        #[arg(long)]
        to: Option<i64>,

        /// Maximum number of results
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Show database statistics
    Stats,

    /// Sample unconnected (stray) memories
    Stray {
        /// Number of stray memories to sample (default: 10)
        #[arg(default_value_t = 10)]
        n: usize,
    },

    /// Show memory ID distribution by date (useful for finding ID ranges)
    Timeline,
}

fn main() {
    let cli = Cli::parse();
    let db_path = cli.db.unwrap_or_else(default_db_path);

    // Determine timezone: explicit --tz wins, else local for text, utc for json
    let use_local = match cli.tz {
        Some(TzOption::Local) => true,
        Some(TzOption::Utc) => false,
        None => !cli.json, // default: local for text, utc for json
    };

    let result = if let Some(command) = cli.command {
        run(command, &db_path, cli.json, use_local)
    } else if let Some(selector) = cli.selector {
        run_selector(&selector, &db_path, cli.json, use_local)
    } else {
        // No command and no selector - show help by running tail
        run(Commands::Tail { n: 10 }, &db_path, cli.json, use_local)
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

/// Parse and execute a selector shorthand.
///
/// Formats:
/// - `5` -> list --from 5 --to 5 (single memory)
/// - `1,3,5,7` -> get memories 1, 3, 5, 7
/// - `1-10` -> list --from 1 --to 10
/// - `5+` -> related 5
/// - `1,3,5+` -> related 1,3,5
/// - `1-10+` -> list --from 1 --to 10, then related on all of them
/// - anything else -> search query
fn run_selector(
    selector: &str,
    db_path: &PathBuf,
    json_output: bool,
    use_local: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (selector_part, is_related) = if selector.ends_with('+') {
        (&selector[..selector.len() - 1], true)
    } else {
        (selector, false)
    };

    // Try to parse as IDs, otherwise treat as search query
    let ids = match parse_selector(selector_part) {
        Some(ids) => ids,
        None => {
            // Not a valid selector, treat as search query
            return run_search(selector, db_path, json_output, use_local);
        }
    };

    if is_related {
        // Run related command with these IDs as seeds
        let store = MemoryStore::open(db_path)?;
        let params = SearchParams::default();
        let result = store.related(&ids, &params)?;
        if json_output {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!(
                "# {} results related to {:?} ({} activated, {} iters)\n",
                result.memories.len(),
                result.seeds,
                result.total_activated,
                result.iterations
            );
            for m in &result.memories {
                let marker = if m.is_context {
                    "~"
                } else if m.is_seed {
                    "*"
                } else {
                    "+"
                };
                print_memory_with_score(&m.memory, m.energy, marker, use_local);
            }
        }
    } else {
        // Just fetch and print these memories
        let store = MemoryStore::open(db_path)?;
        let memories = store.get_many(&ids)?;
        if json_output {
            let result = serde_json::json!({
                "count": memories.len(),
                "memories": memories
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            print_memories(&memories, use_local);
        }
    }

    Ok(())
}

/// Run a search query.
fn run_search(
    query: &str,
    db_path: &PathBuf,
    json_output: bool,
    use_local: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::open(db_path)?;
    let params = SearchParams::default();
    let result = store.search(query, &params)?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!(
            "# {} results for \"{}\" ({} activated, {} iters)\n",
            result.memories.len(),
            result.query,
            result.total_activated,
            result.iterations
        );
        for m in &result.memories {
            let marker = if m.is_context {
                "~"
            } else if m.is_seed {
                "*"
            } else {
                "+"
            };
            print_memory_with_score(&m.memory, m.energy, marker, use_local);
        }
    }
    Ok(())
}

/// Parse a selector string into a list of IDs.
///
/// Formats:
/// - `5` -> [5]
/// - `1,3,5,7` -> [1, 3, 5, 7]
/// - `1-10` -> [1, 2, 3, ..., 10]
///
/// Returns None if the string doesn't look like a valid selector (treated as search query).
fn parse_selector(selector: &str) -> Option<Vec<i64>> {
    // Check for range format (contains exactly one hyphen, not at start)
    if selector.contains('-') && !selector.starts_with('-') {
        let parts: Vec<&str> = selector.splitn(2, '-').collect();
        if parts.len() == 2 {
            let start: i64 = parts[0].trim().parse().ok()?;
            let end: i64 = parts[1].trim().parse().ok()?;
            if start > end {
                return None;
            }
            return Some((start..=end).collect());
        }
    }

    // Check for comma-separated IDs
    if selector.contains(',') {
        let ids: Result<Vec<i64>, _> = selector
            .split(',')
            .map(|s| s.trim().parse::<i64>())
            .collect();
        return ids.ok();
    }

    // Single ID
    let id: i64 = selector.trim().parse().ok()?;
    Some(vec![id])
}

fn run(
    command: Commands,
    db_path: &PathBuf,
    json_output: bool,
    use_local: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Init => {
            let store = MemoryStore::open(db_path)?;
            if json_output {
                let result = serde_json::json!({
                    "success": true,
                    "database": db_path.display().to_string(),
                    "message": "Database initialized successfully",
                    "max_memory_id": store.max_memory_id()
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "SUCCESS: Database initialized at {} (max_id: {})",
                    db_path.display(),
                    store.max_memory_id()
                );
            }
        }

        Commands::Add {
            content,
            source,
            datetime,
        } => {
            let mut store = MemoryStore::open(db_path)?;
            let result =
                store.add_with_options(&content, source.as_deref(), datetime.as_deref())?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_memory(&result.memory, use_local);
            }
        }

        Commands::Undo => {
            let mut store = MemoryStore::open(db_path)?;
            let result = store.undo()?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("UNDONE: {}", result.description);

                // Show full details for re-adding
                if let Some(ref mem) = result.memory {
                    println!("\nTo re-add this memory:");
                    let source_arg = mem
                        .source
                        .as_ref()
                        .map(|s| format!(" --source \"{}\"", s))
                        .unwrap_or_default();
                    println!(
                        "  adaptive-memory add \"{}\" --datetime \"{}\"{}",
                        mem.text.replace("\"", "\\\""),
                        mem.datetime,
                        source_arg
                    );
                }

                if let Some(ref rels) = result.relationships {
                    println!("\nUndone relationships:");
                    for rel in rels {
                        println!(
                            "  {} <-> {} (strength: {:.1})",
                            rel.from_mem, rel.to_mem, rel.strength
                        );
                    }
                    if let Some(ref ids) = result.memory_ids {
                        let ids_str: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
                        println!("\nTo re-add these relationships:");
                        println!("  adaptive-memory strengthen {}", ids_str.join(","));
                    }
                }
            }
        }

        Commands::Search {
            query,
            limit,
            alpha,
            beta,
            context,
            from,
            to,
            decay,
        } => {
            let store = MemoryStore::open(db_path)?;
            let params = SearchParams {
                limit,
                alpha,
                beta,
                context,
                from,
                to,
                decay,
            };
            let result = store.search(&query, &params)?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "# {} results for \"{}\" ({} activated, {} iters)\n",
                    result.memories.len(),
                    result.query,
                    result.total_activated,
                    result.iterations
                );
                for m in &result.memories {
                    let marker = if m.is_context {
                        "~"
                    } else if m.is_seed {
                        "*"
                    } else {
                        "+"
                    };
                    print_memory_with_score(&m.memory, m.energy, marker, use_local);
                }
            }
        }

        Commands::Related {
            ids,
            limit,
            alpha,
            beta,
            context,
            from,
            to,
            decay,
        } => {
            let seed_ids = parse_seed_ids(&ids)?;
            let store = MemoryStore::open(db_path)?;
            let params = SearchParams {
                limit,
                alpha,
                beta,
                context,
                from,
                to,
                decay,
            };
            let result = store.related(&seed_ids, &params)?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!(
                    "# {} results related to {:?} ({} activated, {} iters)\n",
                    result.memories.len(),
                    result.seeds,
                    result.total_activated,
                    result.iterations
                );
                for m in &result.memories {
                    let marker = if m.is_context {
                        "~"
                    } else if m.is_seed {
                        "*"
                    } else {
                        "+"
                    };
                    print_memory_with_score(&m.memory, m.energy, marker, use_local);
                }
            }
        }

        Commands::Strengthen { ids } => {
            let ids = parse_ids(&ids)?;
            let mut store = MemoryStore::open(db_path)?;
            let result = store.strengthen(&ids)?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("STRENGTHENED {} relationships:", result.relationships.len());
                for r in &result.relationships {
                    println!(
                        "  {} <-> {} (strength: {:.1})",
                        r.from_mem, r.to_mem, r.effective_strength
                    );
                }
            }
        }

        Commands::Tail { n } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.tail(n)?;
            if json_output {
                let result = serde_json::json!({
                    "count": memories.len(),
                    "memories": memories
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_memories(&memories, use_local);
            }
        }

        Commands::List { from, to, limit } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.list(from, to, limit)?;
            if json_output {
                let result = serde_json::json!({
                    "count": memories.len(),
                    "memories": memories
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_memories(&memories, use_local);
            }
        }

        Commands::Stats => {
            let store = MemoryStore::open(db_path)?;
            let stats = store.stats()?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("STATS:");
                println!("  Memories:      {}", stats.memory_count);
                println!(
                    "  ID Range:      {} - {}",
                    stats.min_memory_id.unwrap_or(0),
                    stats.max_memory_id.unwrap_or(0)
                );
                println!(
                    "  Relationships: {} (events: {})",
                    stats.relationship_count, stats.relationship_event_count
                );
                println!("  Sources:       {}", stats.unique_sources.join(", "));
                println!("  Graph:");
                println!("    Stray:       {}", stats.graph.stray_count);
                println!("    Islands:     {}", stats.graph.island_count);
                println!("    Largest:     {}", stats.graph.largest_island_size);
                println!("    Leaf:        {}", stats.graph.leaf_count);
                println!("    Max Degree:  {}", stats.graph.max_degree);
                println!("    Avg Degree:  {:.2}", stats.graph.avg_degree);
            }
        }

        Commands::Stray { n } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.stray(n)?;
            if json_output {
                let result = serde_json::json!({
                    "count": memories.len(),
                    "memories": memories
                });
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_memories(&memories, use_local);
            }
        }

        Commands::Timeline => {
            let store = MemoryStore::open(db_path)?;
            let timeline = store.timeline()?;
            if json_output {
                println!("{}", serde_json::to_string_pretty(&timeline)?);
            } else {
                println!("TIMELINE:");
                for b in timeline.buckets {
                    println!(
                        "  {} | IDs: {:>5} - {:<5} | count: {:>4}",
                        b.date, b.min_id, b.max_id, b.count
                    );
                }
                println!("\nSUMMARY:");
                println!(
                    "  Total:   {} memories over {} days",
                    timeline.summary.total_memories, timeline.summary.total_days
                );
                println!(
                    "  Range:   {} to {}",
                    timeline.summary.oldest_date.as_deref().unwrap_or(""),
                    timeline.summary.newest_date.as_deref().unwrap_or("")
                );
                println!(
                    "  IDs:     {} to {}",
                    timeline.summary.oldest_id.unwrap_or(0),
                    timeline.summary.newest_id.unwrap_or(0)
                );
                println!(
                    "  Avg:     {:.1} memories/day",
                    timeline.summary.avg_per_day
                );
            }
        }
    }

    Ok(())
}

fn parse_ids(ids_str: &str) -> Result<Vec<i64>, MemoryError> {
    let ids: Result<Vec<i64>, _> = ids_str
        .split(',')
        .map(|s| s.trim().parse::<i64>())
        .collect();

    let ids = ids.map_err(|e| MemoryError::InvalidInput(format!("Invalid ID format: {}", e)))?;

    if ids.is_empty() {
        return Err(MemoryError::InvalidInput(
            "At least one memory ID is required".to_string(),
        ));
    }

    if ids.len() > MAX_STRENGTHEN_SET {
        return Err(MemoryError::InvalidInput(format!(
            "Cannot strengthen more than {} memories at once (got {})",
            MAX_STRENGTHEN_SET,
            ids.len()
        )));
    }

    Ok(ids)
}

/// Parse seed IDs for the related command (no upper limit on count).
fn parse_seed_ids(ids_str: &str) -> Result<Vec<i64>, MemoryError> {
    let ids: Result<Vec<i64>, _> = ids_str
        .split(',')
        .map(|s| s.trim().parse::<i64>())
        .collect();

    let ids = ids.map_err(|e| MemoryError::InvalidInput(format!("Invalid ID format: {}", e)))?;

    if ids.is_empty() {
        return Err(MemoryError::InvalidInput(
            "At least one seed memory ID is required".to_string(),
        ));
    }

    Ok(ids)
}

/// Format datetime based on timezone option
fn format_datetime(dt: &chrono::DateTime<chrono::Utc>, use_local: bool) -> String {
    if use_local {
        Local
            .from_utc_datetime(&dt.naive_utc())
            .format("%Y-%m-%d %H:%M")
            .to_string()
    } else {
        dt.format("%Y-%m-%d %H:%M").to_string()
    }
}

/// Format a memory for text output (compact, no marker)
fn print_memory(m: &adaptive_memory::Memory, use_local: bool) {
    let source_str = m.source.as_deref().unwrap_or("-");
    println!(
        "--- Memory {} | {} | {} ---\n{}",
        m.id,
        format_datetime(&m.datetime, use_local),
        source_str,
        m.text
    );
}

/// Format a memory with score and marker for search output
fn print_memory_with_score(
    m: &adaptive_memory::Memory,
    energy: f64,
    marker: &str,
    use_local: bool,
) {
    let source_str = m.source.as_deref().unwrap_or("-");
    println!(
        "--- Memory {} | {} | {} | {:.2}{} ---\n{}",
        m.id,
        format_datetime(&m.datetime, use_local),
        source_str,
        energy,
        marker,
        m.text
    );
}

/// Print a list of memories
fn print_memories(memories: &[adaptive_memory::Memory], use_local: bool) {
    for m in memories {
        print_memory(m, use_local);
    }
}
