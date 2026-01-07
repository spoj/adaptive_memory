# Adaptive Memory

An associative memory system using Personalized PageRank (PPR). Memories are stored in SQLite with FTS5 full-text search, and retrieved using BM25 text matching combined with PPR graph-based ranking through explicit relationships.

## Core Concepts

### Memories
Entries with `id`, `datetime`, `text`, and optional `source`. Stored in SQLite with FTS5 full-text indexing. IDs are sequential integers assigned on insertion.

### Relationships
Symmetric connections between memories. Created only via explicit `strengthen` calls - no auto-generated relationships. Multiple strengthen events accumulate; effective strength is the sum of all events.

Relationships are stored canonically (`from_mem < to_mem`) as an event log. This allows strength to build up over time through repeated strengthening.

### Personalized PageRank (PPR)
Search works by:
1. **FTS5 Search**: Find memories matching query using BM25 ranking
2. **Seed Selection**: Top BM25 results become seeds (normalized to sum to 1.0)
3. **PPR Power Iteration**: Classic PageRank with personalization
   - `score = (1 - α) * seed + α * P * score` where P is transition matrix
   - Edge weights (relationship strength) determine transition probabilities
   - Dangling nodes (no connections) teleport back to seeds
   - Converges in ~20-50 iterations (max 100)
4. **Results**: Memories sorted by PPR score (highest first)

### Context Expansion
Instead of pre-computed temporal relationships, use `--context N` to fetch N memories before/after each result by ID. This is like `grep -B/-A` for temporal context.

## Installation

### From crates.io

```bash
cargo install adaptive_memory
```

### From source

```bash
git clone https://github.com/spoj/adaptive_memory
cd adaptive_memory
cargo build --release
# Binary at: target/release/adaptive-memory
```

## CLI Usage

```
adaptive-memory [OPTIONS] <COMMAND>

Commands:
  init        Initialize the database
  add         Add a new memory
  undo        Undo the last operation (add or strengthen)
  search      Search for memories
  related     Find memories related to seed IDs via graph
  strengthen  Strengthen relationships between memories
  tail        Show the latest N memories
  list        List memories by ID range
  stats       Show database statistics
  stray       Sample unconnected (stray) memories
  timeline    Show memory ID distribution by date

Global Options:
  --db <PATH>  Database path (default: ~/.adaptive_memory.db)
  --json       Output in JSON format
```

### Initialize Database

```bash
adaptive-memory init
```

### Add Memory

```bash
adaptive-memory add [OPTIONS] <TEXT>

Options:
  -s, --source <SOURCE>      Source identifier (e.g., "journal", "slack")
  -d, --datetime <DATETIME>  Override datetime (RFC3339 format)
```

**Examples:**
```bash
# Simple memory
adaptive-memory add "Had coffee with Sarah, discussed the new project"

# With source
adaptive-memory add "Reviewed PR #123" -s "github"

# Historical entry
adaptive-memory add "Started learning Rust" -d "2023-06-15T10:00:00Z"
```

**Output:**
```json
{
  "memory": {
    "id": 42,
    "datetime": "2026-01-05T12:30:00Z",
    "text": "Had coffee with Sarah, discussed the new project",
    "source": null
  }
}
```

### Search Memories

```bash
adaptive-memory search [OPTIONS] <QUERY>

Options:
  -l, --limit <N>      Maximum results (default: 10)
  -c, --context <N>    Fetch N memories before/after each result (default: 0)
  -a, --alpha <VALUE>  PPR damping factor (default: 0.85, lower = more weight to text matches)
  -b, --beta <VALUE>   Degree penalty (default: 0.5, higher = boost unique links over hubs)
  --from <ID>          Filter results to memories with ID >= from
  --to <ID>            Filter results to memories with ID <= to
  --decay <SCALE>      Decay scale for relationship strength (default: 0 = no decay)
```

**Decay**: Older relationships contribute less to search. At age = scale, strength halves.
Formula: `effective_strength = strength / (1 + age / scale)` where age is measured in relationship event IDs.

| Scale | Effect |
|-------|--------|
| 0     | No decay (default) |
| 100   | Aggressive - 50% at 100 events ago |
| 1000  | Moderate - 50% at 1000 events ago |
| 5000  | Gentle - old connections retain influence longer |

**Examples:**
```bash
# Basic search
adaptive-memory search "project meeting"

# With temporal context (like grep -B2 -A2)
adaptive-memory search "rust" --context 2

# Limit results
adaptive-memory search "database" --limit 10

# More weight to text matches (less graph influence)
adaptive-memory search "ideas" --alpha 0.5
```

**Output:**
```json
{
  "query": "project meeting",
  "seed_count": 15,
  "total_activated": 47,
  "iterations": 234,
  "memories": [
    {
      "id": 38,
      "datetime": "2026-01-04T09:00:00Z",
      "text": "Project kickoff meeting with the team",
      "source": "calendar",
      "energy": 1.87
    },
    {
      "id": 42,
      "datetime": "2026-01-05T12:30:00Z",
      "text": "Had coffee with Sarah, discussed the new project",
      "source": null,
      "energy": 2.45
    }
  ]
}
```

Results are sorted by PPR score (highest first). The `energy` field indicates relevance (PPR scores sum to 1.0 across all activated nodes).

Context items (from `--context`) have `energy: 0.0` and `is_context: true`.

### Related (Graph Search)

Find memories related to specific seed IDs via graph traversal (no text search).

```bash
adaptive-memory related [OPTIONS] <IDS>

Arguments:
  <IDS>  Comma-separated seed memory IDs

Options:
  -l, --limit <N>      Maximum results (default: 10)
  -c, --context <N>    Fetch N memories before/after each result (default: 0)
  -a, --alpha <VALUE>  PPR damping factor (default: 0.85)
  -b, --beta <VALUE>   Degree penalty (default: 0.5)
  --from <ID>          Filter results to memories with ID >= from
  --to <ID>            Filter results to memories with ID <= to
  --decay <SCALE>      Decay scale for relationship strength (default: 0)
```

**Example:**
```bash
# Find memories related to memories 42 and 38
adaptive-memory related 42,38

# With decay - older relationships matter less
adaptive-memory related 42,38 --decay 1000
```

### Strengthen Relationships

Create explicit associations between memories. Always adds strength (use `undo` to reverse).

```bash
adaptive-memory strengthen <IDS>

Arguments:
  <IDS>  Comma-separated memory IDs (max 10)
```

**Examples:**
```bash
# Link two related memories (adds 1.0 strength)
adaptive-memory strengthen 42,38

# Link multiple (creates all pairs, 1.0 each)
# 4 IDs = 6 pairs, each gets 1.0 strength
adaptive-memory strengthen 1,5,12,34

# Repeated calls accumulate strength
adaptive-memory strengthen 42,38  # Now strength = 2.0
```

**Output:**
```json
{
  "relationships": [
    {
      "from_mem": 38,
      "to_mem": 42,
      "effective_strength": 2.0
    }
  ]
}
```

### Undo

Undo the last operation (`add` or `strengthen`). Works like a stack - can only undo the most recent operation.

```bash
adaptive-memory undo
```

**Examples:**
```bash
# Add a memory
adaptive-memory add "Test memory"
# Output: Memory 42 | ...

# Oops, undo it
adaptive-memory undo
# Output: UNDONE: add memory #42: "Test memory"

# Strengthen some memories
adaptive-memory strengthen 1,2,3
# Output: STRENGTHENED 3 relationships

# Undo the strengthen
adaptive-memory undo
# Output: UNDONE: strengthen 3 relationships between memories [1, 2, 3]
```

**Note**: Undo is append-only at the rear. You cannot undo operations from the middle of history - only the most recent operation can be undone.

### List Memories

List memories by ID range.

```bash
adaptive-memory list [OPTIONS]

Options:
  --from <FROM>    Start ID (inclusive)
  --to <TO>        End ID (inclusive)
  -l, --limit <N>  Maximum number of results
```

**Examples:**
```bash
# List memories 10-20
adaptive-memory list --from 10 --to 20

# List last 50 memories
adaptive-memory list --limit 50
```

### Tail

Show the latest N memories (shorthand for `list --limit N`).

```bash
adaptive-memory tail [N]

Arguments:
  [N]  Number of memories to show (default: 10)
```

**Example:**
```bash
# Show last 5 memories
adaptive-memory tail 5
```

### Stats

Show database statistics including memory count, relationship count, and graph metrics.

```bash
adaptive-memory stats
```

**Output:**
```json
{
  "memory_count": 1234,
  "relationship_count": 567,
  "connected_memories": 890,
  "stray_memories": 344,
  "avg_connections": 1.27
}
```

### Stray

Sample unconnected (stray) memories - useful for finding memories that could benefit from being linked to others.

```bash
adaptive-memory stray [N]

Arguments:
  [N]  Number of stray memories to sample (default: 10)
```

**Example:**
```bash
# Find 5 unconnected memories to review
adaptive-memory stray 5
```

## FTS5 Query Syntax

The search query uses SQLite FTS5 syntax, which supports powerful search operators:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `word` | Match word | `meeting` |
| `word1 word2` | Match both (implicit AND) | `project meeting` |
| `word1 OR word2` | Match either | `cat OR dog` |
| `"phrase"` | Exact phrase | `"weekly standup"` |
| `word*` | Prefix match | `meet*` matches meeting, meetings |
| `NOT word` | Exclude | `meeting NOT standup` |
| `NEAR(w1 w2, N)` | Words within N tokens | `NEAR(rust memory, 5)` |
| `^word` | Match at start of field | `^TODO` |

**Special characters**: Characters like `+`, `-`, `@` have special meaning in FTS5. To search for literal special characters, quote them: `"2024-01-15"` or `"email@example.com"`.

**Examples:**
```bash
# All memories with "rust" AND "async"
adaptive-memory search "rust async"

# Either term
adaptive-memory search "rust OR python"

# Exact phrase
adaptive-memory search '"weekly standup"'

# Prefix matching
adaptive-memory search "meet*"

# Exclude term
adaptive-memory search "project NOT cancelled"

# Words near each other
adaptive-memory search "NEAR(database migration, 10)"
```

## Library Usage

```rust
use adaptive_memory::{MemoryStore, MemoryError, SearchParams};

fn main() -> Result<(), MemoryError> {
    let mut store = MemoryStore::open("~/.adaptive_memory.db")?;

    // Add memories
    let result = store.add("Learning about spreading activation", Some("research"))?;
    println!("Added memory {}", result.memory.id);

    // Search with default params
    let results = store.search("activation", &SearchParams::default())?;
    for mem in results.memories {
        println!("{}: {} (energy: {:.2})", mem.memory.id, mem.memory.text, mem.energy);
    }

    // Search with context expansion
    let params = SearchParams {
        limit: 50,
        context: 2,
        ..SearchParams::default()
    };
    let results = store.search("activation", &params)?;

    // Strengthen relationships
    store.strengthen(&[1, 2, 3])?;

    Ok(())
}
```

## Configuration

### Compile-time Constants (`src/lib.rs`)

| Constant | Default | Description |
|----------|---------|-------------|
| `PPR_EPSILON` | 1e-6 | Convergence threshold for PPR iteration |
| `PPR_MAX_ITER` | 100 | Maximum PPR iterations |
| `MAX_STRENGTHEN_SET` | 10 | Max memories per strengthen call |
| `DEFAULT_LIMIT` | 10 | Default result limit |

### Runtime Parameters (`SearchParams`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 10 | Max results (also seed count for FTS) |
| `alpha` | 0.85 | PPR damping factor (classic PageRank value) |
| `beta` | 0.5 | Degree penalty (0=none, 0.5=sqrt, 1.0=linear) |
| `context` | 0 | Fetch N memories before/after each result |
| `from` | None | Filter by minimum memory ID |
| `to` | None | Filter by maximum memory ID |
| `decay` | 0.0 | Decay scale (0 = no decay) |

### Tuning `alpha` (Damping Factor)

Controls the balance between text matches (seeds) and graph structure:

| Value | Behavior |
|-------|----------|
| 0.5 | More weight to text matches, less graph exploration |
| 0.85 | Classic PageRank balance (default) |
| 0.95 | More weight to graph structure, deeper exploration |

The PPR formula is: `score = (1 - α) * seed + α * P * score`
- Lower α → higher `(1 - α)` → seeds dominate
- Higher α → more propagation through graph

## Database Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    datetime TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT
);

CREATE VIRTUAL TABLE memories_fts USING fts5(text, content=memories, content_rowid=id);

-- Event-log style: multiple rows per pair allowed, summed for effective strength
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    from_mem INTEGER NOT NULL,
    to_mem INTEGER NOT NULL,
    strength REAL NOT NULL,
    CHECK (from_mem < to_mem)
);

-- Operation log for undo support
CREATE TABLE operations (
    id INTEGER PRIMARY KEY,
    op_type TEXT NOT NULL,     -- 'add' or 'strengthen'
    payload TEXT NOT NULL,     -- JSON with IDs for undo
    created_at TEXT NOT NULL
);
```

## How It Works

### Adding a Memory
1. Insert into `memories` table
2. FTS5 trigger auto-indexes the text
3. No relationships created (use `strengthen` or `--context` for associations)

### Searching
1. **FTS5**: BM25-ranked text matches become seeds (normalized to sum to 1.0)
2. **PPR Power Iteration**:
   - `score = (1 - α) * seed + α * P * score`
   - P is transition matrix from relationship strengths (normalized by out-degree)
   - Dangling nodes teleport back to seeds
   - Iterates until convergence (L1 diff < 1e-6) or max 100 iterations
3. **Context Expansion**: Optionally fetch surrounding memories by ID
4. **Results**: Sorted by PPR score (highest first)

### Strengthening
1. For each pair of IDs, add relationship event with strength 1.0
2. Events accumulate - the pair's effective strength grows with repeated strengthening
3. Higher strength → higher transition probability in PPR (normalized by out-degree)

## Tips

- **Source field**: Tag memories for filtering/identification (e.g., "slack", "journal", "calendar")
- **Strengthen after retrieval**: If a search surfaces related memories, strengthen them to reinforce the association
- **Context for temporal**: Use `--context N` instead of pre-computed temporal links
- **Batch import**: Use `-d` to preserve original timestamps when importing historical data
- **Quote special chars**: FTS5 special characters (`+`, `-`, `*`, etc.) should be quoted for literal matching

## License

MIT
