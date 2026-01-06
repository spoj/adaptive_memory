use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

use adaptive_memory::{
    DEFAULT_LIMIT, MAX_STRENGTHEN_SET, MemoryError, MemoryStore, SearchParams, default_db_path,
};

#[derive(Parser)]
#[command(name = "adaptive-memory")]
#[command(about = "Adaptive memory system with spreading activation", long_about = None)]
struct Cli {
    /// Path to the database file (default: ~/.adaptive_memory.db)
    #[arg(long, global = true)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize the database (creates if not exists)
    Init,

    /// Add a new memory
    Add {
        /// The memory text
        text: String,

        /// Optional source identifier
        #[arg(short, long)]
        source: Option<String>,

        /// Optional datetime override (RFC3339 format, e.g. "2024-01-15T10:30:00Z")
        #[arg(short, long)]
        datetime: Option<String>,
    },

    /// Amend (update) an existing memory's text
    Amend {
        /// Memory ID to amend
        id: i64,

        /// New text for the memory
        text: String,
    },

    /// Search for memories using text query and spreading activation
    Search {
        /// Search query (required, cannot be empty)
        query: String,

        /// Maximum number of results to return (also used as seed count)
        #[arg(short, long, default_value_t = DEFAULT_LIMIT)]
        limit: usize,

        /// Decay factor for relationship strength over memory distance (0 = no decay)
        #[arg(long, default_value_t = 0.0)]
        decay: f64,

        /// Energy decay per hop during spreading activation
        #[arg(long, default_value_t = 0.5)]
        energy_decay: f64,

        /// Context window: fetch N memories before/after each result (like grep -B/-A)
        #[arg(short, long, default_value_t = 0)]
        context: usize,

        /// Filter results to memories with ID >= from (inclusive)
        #[arg(long)]
        from: Option<i64>,

        /// Filter results to memories with ID <= to (inclusive)
        #[arg(long)]
        to: Option<i64>,
    },

    /// Strengthen relationships between memories
    Strengthen {
        /// Comma-separated list of memory IDs (max 10)
        ids: String,
    },

    /// Connect memories (only if no existing relationship)
    Connect {
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
}

fn main() {
    let cli = Cli::parse();
    let db_path = cli.db.unwrap_or_else(default_db_path);

    let result = run(cli.command, &db_path);

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run(command: Commands, db_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Init => {
            let store = MemoryStore::open(db_path)?;
            let result = serde_json::json!({
                "success": true,
                "database": db_path.display().to_string(),
                "message": "Database initialized successfully",
                "max_memory_id": store.max_memory_id()
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Add {
            text,
            source,
            datetime,
        } => {
            let mut store = MemoryStore::open(db_path)?;
            let result = store.add_with_options(&text, source.as_deref(), datetime.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Amend { id, text } => {
            let mut store = MemoryStore::open(db_path)?;
            let result = store.amend(id, &text)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Search {
            query,
            limit,
            decay,
            energy_decay,
            context,
            from,
            to,
        } => {
            let store = MemoryStore::open(db_path)?;
            let params = SearchParams {
                limit,
                decay_factor: decay,
                energy_decay,
                context,
                from,
                to,
            };
            let result = store.search(&query, &params)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Strengthen { ids } => {
            let ids = parse_ids(&ids)?;
            let mut store = MemoryStore::open(db_path)?;
            let result = store.strengthen(&ids)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Connect { ids } => {
            let ids = parse_ids(&ids)?;
            let mut store = MemoryStore::open(db_path)?;
            let result = store.connect(&ids)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Tail { n } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.tail(n)?;
            let result = serde_json::json!({
                "count": memories.len(),
                "memories": memories
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::List { from, to, limit } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.list(from, to, limit)?;
            let result = serde_json::json!({
                "count": memories.len(),
                "memories": memories
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
        }

        Commands::Stats => {
            let store = MemoryStore::open(db_path)?;
            let stats = store.stats()?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }

        Commands::Stray { n } => {
            let store = MemoryStore::open(db_path)?;
            let memories = store.stray(n)?;
            let result = serde_json::json!({
                "count": memories.len(),
                "memories": memories
            });
            println!("{}", serde_json::to_string_pretty(&result)?);
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
