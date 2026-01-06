# Adaptive Memory

An associative memory system using spreading activation. Memories are stored in SQLite with FTS5 full-text search, and retrieved using BM25 text matching combined with graph-based activation spreading through explicit relationships.

## Core Concepts

### Memories
Entries with `id`, `datetime`, `text`, and optional `source`. Stored in SQLite with FTS5 full-text indexing. IDs are sequential integers assigned on insertion.

### Relationships
Symmetric connections between memories. Created only via explicit `strengthen` calls - no auto-generated relationships. Multiple strengthen events accumulate; effective strength is the sum of all events.

Relationships are stored canonically (`from_mem < to_mem`) as an event log. This allows strength to build up over time through repeated strengthening.

### Spreading Activation
Search works by:
1. **FTS5 Search**: Find memories matching query using BM25 ranking
2. **Seed Selection**: Top BM25 results become seeds with energy 0.1-1.0 (proportional to relevance)
3. **Energy Propagation**: Energy spreads through relationship graph
   - Energy is *distributed* across neighbors (PageRank-style normalization)
   - Each hop multiplies by `energy_decay` (default 0.5)
   - Propagation stops when energy < 0.01 threshold
4. **Results**: Memories sorted by energy score (highest first)

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
  amend       Amend (update) an existing memory's text
  search      Search for memories
  strengthen  Strengthen relationships between memories
  connect     Connect memories (only if no existing relationship)
  tail        Show the latest N memories
  list        List memories by ID range
  stats       Show database statistics
  stray       Sample unconnected (stray) memories

Global Options:
  --db <PATH>  Database path (default: ~/.adaptive_memory.db)
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
  -l, --limit <N>          Maximum results (default: 10)
  -c, --context <N>        Fetch N memories before/after each result (default: 0)
  --energy-decay <FACTOR>  Energy multiplier per hop (default: 0.5)
  -k <VALUE>               Sigmoid k for edge strength: strength/(strength+k) (default: 1.0)
```

**Examples:**
```bash
# Basic search
adaptive-memory search "project meeting"

# With temporal context (like grep -B2 -A2)
adaptive-memory search "rust" --context 2

# Limit results
adaptive-memory search "database" --limit 10

# Deeper activation spread (reach more distant associations)
adaptive-memory search "ideas" --energy-decay 0.7
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

Results are sorted by energy score (highest first). The `energy` field indicates relevance:
- ~1.0 = direct BM25 match
- ~0.5 = one hop from a seed
- < 0.1 = reached via multi-hop spreading

Context items (from `--context`) have `energy: 0.0` and `is_context: true`.

### Strengthen Relationships

Create explicit associations between memories.

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
```

**Output:**
```json
{
  "relationships": [
    {
      "from_mem": 38,
      "to_mem": 42,
      "effective_strength": 2.0,
      "event_count": 2
    }
  ],
  "event_count": 1
}
```

### Connect Memories

Like `strengthen`, but only creates relationships if none exist between the pair.

```bash
adaptive-memory connect <IDS>

Arguments:
  <IDS>  Comma-separated memory IDs (max 10)
```

**Example:**
```bash
# Connect memories only if not already related
adaptive-memory connect 42,38,15
```

### Amend Memory

Update the text of an existing memory. Only allowed if the memory has no relationships to later memories (preserves integrity of memories that later entries depend on).

```bash
adaptive-memory amend <ID> <TEXT>

Arguments:
  <ID>    Memory ID to amend
  <TEXT>  New text for the memory
```

**Example:**
```bash
# Fix a typo in memory 42
adaptive-memory amend 42 "Had coffee with Sarah, discussed the new project timeline"
```

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
| `ENERGY_THRESHOLD` | 0.01 | Stop propagation below this energy |
| `MAX_SPREADING_ITERATIONS` | 5000 | Safety limit on activation iterations |
| `MAX_STRENGTHEN_SET` | 10 | Max memories per strengthen call |
| `DEFAULT_LIMIT` | 10 | Default result limit |

### Runtime Parameters (`SearchParams`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 10 | Max results (also seed count for FTS) |
| `energy_decay` | 0.5 | Energy multiplier per hop (0.5 = 50% retained each hop) |
| `sigmoid_k` | 1.0 | Sigmoid k for edge strength: strength/(strength+k) |
| `context` | 0 | Fetch N memories before/after each result |

### Tuning `energy_decay`

Controls how far activation spreads through the graph:

| Value | Behavior | Max Depth |
|-------|----------|-----------|
| 0.3 | Shallow spread, stick close to seeds | ~4 hops |
| 0.5 | Balanced (default) | ~7 hops |
| 0.7 | Deep spread, reach distant associations | ~12 hops |

Energy at each hop (starting from seed with energy 1.0):

```
Hop:    0     1      2       3        4
0.5:   1.0   0.50   0.25    0.125    0.0625
0.7:   1.0   0.70   0.49    0.343    0.240
```

### Tuning `sigmoid_k`

Controls how relationship strength translates to propagation factor:

```
propagation_factor = strength / (strength + k)
```

| k | strength=1 | strength=2 | strength=5 | strength=10 |
|---|------------|------------|------------|-------------|
| 0.5 | 0.67 | 0.80 | 0.91 | 0.95 |
| 1.0 | 0.50 | 0.67 | 0.83 | 0.91 |
| 2.0 | 0.33 | 0.50 | 0.71 | 0.83 |
| 5.0 | 0.17 | 0.29 | 0.50 | 0.67 |

Lower k = stronger edges propagate more energy. Higher k = requires more strengthening to achieve high propagation.

## Database Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    datetime TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT
);

CREATE VIRTUAL TABLE memories_fts USING fts5(text, content=memories, content_rowid=id);

CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    from_mem INTEGER NOT NULL,
    to_mem INTEGER NOT NULL,
    created_at_mem INTEGER NOT NULL,
    strength REAL NOT NULL,
    CHECK (from_mem < to_mem)
);
```

## How It Works

### Adding a Memory
1. Insert into `memories` table
2. FTS5 trigger auto-indexes the text
3. No relationships created (use `strengthen` or `--context` for associations)

### Searching
1. **FTS5**: BM25-ranked text matches become seeds
2. **Spreading Activation**:
   - Seeds get energy proportional to BM25 score
   - Energy spreads through relationships (delta propagation)
   - Neighbors' strengths are normalized (sum to 1.0) - energy is distributed, not amplified
   - Raw strength is compressed via `ln(1+x)` for diminishing returns
3. **Context Expansion**: Optionally fetch surrounding memories by ID
4. **Results**: Sorted by memory ID (timeline order)

### Strengthening
1. For each pair of IDs, add relationship event with strength 1.0
2. Events accumulate - the pair's effective strength grows with repeated strengthening
3. ln_1p compression means: 1st event → 0.69 effective, 10 events → 2.40, 100 events → 4.62

## Tips

- **Source field**: Tag memories for filtering/identification (e.g., "slack", "journal", "calendar")
- **Strengthen after retrieval**: If a search surfaces related memories, strengthen them to reinforce the association
- **Context for temporal**: Use `--context N` instead of pre-computed temporal links
- **Batch import**: Use `-d` to preserve original timestamps when importing historical data
- **Quote special chars**: FTS5 special characters (`+`, `-`, `*`, etc.) should be quoted for literal matching

## License

MIT
