#!/usr/bin/env python3
"""LoCoMo benchmark harness for adaptive_memory.

Evaluates adaptive_memory on the LoCoMo long-term conversational memory benchmark.

Modes:
  - fts5_only: Pure FTS5 search, return top match (no LLM reasoning)
  - fts5_llm: FTS5 search + LLM reasoning (no relationships)
  - spreading_llm: FTS5 + spreading activation + LLM reasoning (with pre-tagged relationships)

Usage:
  uv run python locomo_bench.py --mode all
  uv run python locomo_bench.py --mode fts5_llm --samples 1
"""

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from eval_metrics import f1_multi, f1_score

# --- Constants ---

DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
LLM_MODEL = "gemini-3-flash-preview"
PROXY = "socks5://dosg:1080"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RETRIEVAL_LIMIT = 20
CONTEXT_WINDOW = 0
ENERGY_DECAY = 0.7  # Default spreading decay

# --- Category names ---

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

# --- Prompts ---

PRETAG_PROMPT = """You are analyzing a conversation to find semantically related memories.

Here are all the memories (ID: text):
{memories}

For each memory, list other memory IDs that are semantically related (same topic, person, event, or continuation of discussion).
Output as JSON: {{"id": [related_ids], ...}}
Only include memories that have at least one strong semantic relationship. Be selective.
Output only valid JSON, no other text."""

QA_PROMPT = """Based on the conversation excerpts below, answer the question in a short phrase.
Use exact words from the context when possible. If no information is available, say "No information available".

Context:
{context}

Question: {question}
Short answer:"""


# --- Subprocess helpers ---


def adaptive_memory(db_path: str, *args: str) -> str:
    """Call adaptive-memory CLI."""
    cmd = ["adaptive-memory", "--db", db_path] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and result.stderr:
        print(f"  Warning: {result.stderr.strip()}")
    return result.stdout


def llm_call(prompt: str, model: str = None) -> str:
    """Call LLM via llm CLI with proxy and retry logic."""
    model = model or LLM_MODEL

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["llm", "-n", "-m", model, "-o", "thinking_level", "minimal", prompt],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "HTTPS_PROXY": PROXY},
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"  LLM error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  LLM timeout")

        if attempt < MAX_RETRIES - 1:
            print(f"  Retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(RETRY_DELAY)

    return "Error: LLM call failed"


# --- Data loading ---


def download_data(data_dir: Path) -> list:
    """Download locomo10.json if not present."""
    data_file = data_dir / "locomo10.json"
    if not data_file.exists():
        print("Downloading locomo10.json...")
        subprocess.run(
            ["curl", "-sL", "-o", str(data_file), DATA_URL],
            env={**os.environ, "HTTPS_PROXY": PROXY},
        )
    return json.loads(data_file.read_text())


def parse_datetime(locomo_dt: str) -> str:
    """Parse LoCoMo datetime string like '1:56 pm on 8 May, 2023' to ISO format."""
    # Pattern: "H:MM am/pm on D Month, YYYY"
    pattern = r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})"
    match = re.match(pattern, locomo_dt, re.IGNORECASE)

    if not match:
        # Fallback: return a default datetime
        return "2023-01-01T00:00:00Z"

    hour, minute, ampm, day, month_name, year = match.groups()
    hour = int(hour)
    minute = int(minute)
    day = int(day)
    year = int(year)

    # Convert 12-hour to 24-hour
    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    elif ampm.lower() == "am" and hour == 12:
        hour = 0

    # Parse month
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = months.get(month_name.lower(), 1)

    dt = datetime(year, month, day, hour, minute)
    return dt.isoformat() + "Z"


# --- Ingestion ---


def ingest_conversation(db_path: str, sample: dict) -> int:
    """Load all dialog turns into adaptive_memory. Returns memory count."""
    # Delete existing DB to avoid duplicates
    if os.path.exists(db_path):
        os.remove(db_path)
    adaptive_memory(db_path, "init")
    conv = sample["conversation"]
    memory_count = 0

    for sess_num in range(1, 100):
        key = f"session_{sess_num}"
        if key not in conv:
            break

        dt_key = f"{key}_date_time"
        dt = parse_datetime(conv.get(dt_key, ""))

        for turn in conv[key]:
            text = f"{turn['speaker']}: {turn['text']}"
            # Include image caption if present
            if "blip_caption" in turn:
                text += f" [shared image: {turn['blip_caption']}]"
            adaptive_memory(db_path, "add", text, "-d", dt)
            memory_count += 1

    return memory_count


# --- Pre-tagging (spreading_llm mode) via Stray Chasing ---


PARALLEL_QA = 30  # Number of parallel QA calls
STRAY_CHASE_MAX_ROUNDS = 20  # Max rounds of stray chasing
STRAY_CHASE_PER_ROUND = 15  # Strays to process per round
STRAY_CHASE_PARALLEL = 15  # Parallel LLM calls for stray chasing

STRAY_RELATE_PROMPT = """Memory {stray_id}: {stray_text}

Candidates:
{candidates}

Which candidate IDs are semantically related to memory {stray_id}?
(Same topic, person, event, or continuation of discussion)
Output ONLY comma-separated IDs, or "none" if no strong relationships."""


def stray_query_from_text(text: str) -> str:
    """Generate simple OR query from memory text."""
    words = re.findall(r"\b\w+\b", text.lower())
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 3][:5]
    return " OR ".join(content_words) if content_words else "memory"


def chase_one_stray(db_path: str, stray: dict, model: str = None) -> tuple:
    """Process one stray memory. Returns (stray_id, [related_ids])."""
    # Search for candidates using stray's text
    query = stray_query_from_text(stray["text"])
    result = adaptive_memory(db_path, "search", query, "-l", "8")

    try:
        candidates = json.loads(result).get("memories", [])
    except json.JSONDecodeError:
        return (stray["id"], [])

    # Filter out self
    candidates = [c for c in candidates if c["id"] != stray["id"]][:6]
    if not candidates:
        return (stray["id"], [])

    # Format candidates for LLM
    cand_text = "\n".join([f"{c['id']}: {c['text'][:100]}" for c in candidates])
    prompt = STRAY_RELATE_PROMPT.format(
        stray_id=stray["id"], stray_text=stray["text"][:150], candidates=cand_text
    )

    # Ask LLM which are related
    response = llm_call(prompt, model)

    # Parse response
    if "none" in response.lower():
        return (stray["id"], [])

    # Extract IDs from response
    ids = re.findall(r"\b\d+\b", response)
    related = [int(i) for i in ids if int(i) != stray["id"]]
    return (stray["id"], related[:4])  # Max 4 relationships per stray


def pretag_relationships(db_path: str, model: str = None) -> int:
    """Use stray-chasing to create semantic relationships.

    Procedure:
    1. Get unconnected memories (strays)
    2. For each stray, search for similar memories
    3. Ask LLM to confirm semantic relationships
    4. Create connections
    5. Repeat until no strays or max rounds
    """
    total_relationships = 0

    # Get initial stats
    stats = json.loads(adaptive_memory(db_path, "stats"))
    initial_strays = stats["graph"]["stray_count"]
    print(f"    Starting stray chase: {initial_strays} strays")

    for round_num in range(1, STRAY_CHASE_MAX_ROUNDS + 1):
        # Get stray memories
        result = adaptive_memory(db_path, "stray", str(STRAY_CHASE_PER_ROUND))
        try:
            strays = json.loads(result).get("memories", [])
        except json.JSONDecodeError:
            print(f"    Round {round_num}: Error getting strays")
            break

        if not strays:
            print(f"    Round {round_num}: No more strays - done!")
            break

        round_start = time.time()
        round_rels = 0

        # Process strays in parallel
        with ThreadPoolExecutor(max_workers=STRAY_CHASE_PARALLEL) as executor:
            futures = {
                executor.submit(chase_one_stray, db_path, stray, model): stray["id"]
                for stray in strays
            }
            for future in as_completed(futures):
                stray_id, related_ids = future.result()
                if related_ids:
                    # Create connections (ensure from < to for constraint)
                    ids_str = ",".join([str(stray_id)] + [str(r) for r in related_ids])
                    adaptive_memory(db_path, "connect", ids_str)
                    round_rels += len(related_ids)

        total_relationships += round_rels
        elapsed = time.time() - round_start

        # Get updated stray count
        stats = json.loads(adaptive_memory(db_path, "stats"))
        remaining = stats["graph"]["stray_count"]

        print(
            f"    Round {round_num}: +{round_rels} rels in {elapsed:.1f}s | {remaining} strays left | total: {total_relationships}"
        )

        # Early exit if few strays remain
        if remaining <= 5:
            print(f"    Few strays left ({remaining}) - stopping early")
            break

    return total_relationships


# --- QA Evaluation ---


# Stopwords for deterministic query cleaning (mode: fts5_only)
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "if",
    "because",
    "about",
    "against",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
}

QUERY_EXPAND_PROMPT = """Generate 3 different search queries to find memories that could answer this question.
Use different phrasings, synonyms, and related terms.
Output one query per line, nothing else.

Question: {question}
Queries:"""

QUERY_REWRITE_PROMPT = """Extract ALL important search terms from this question.
Include names, topics, events, activities - everything that might appear in relevant memories.
Use OR between terms for maximum recall.
Output ONLY the search terms separated by OR, nothing else.

Question: {question}
Terms:"""


def clean_query_deterministic(query: str) -> str:
    """Clean query by removing stopwords (deterministic, no LLM).

    Simple approach: just OR all content words together.
    """
    words = re.findall(r"\b\w+\b", query.lower())
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if not content_words:
        content_words = words[:3]

    # Simple OR of all content words
    return " OR ".join(content_words)


def sanitize_fts5_query(query: str) -> str:
    """Sanitize query for FTS5 - handle special characters."""
    # Replace hyphens with spaces (FTS5 treats - as NOT operator)
    query = query.replace("-", " ")
    # Remove problematic characters (keep AND/OR operators)
    query = re.sub(r'["\'\(\)\[\]\{\},;:.!/\\@#$%^&*+=<>?`~]', " ", query)
    # Remove possessives
    query = re.sub(r"'s\b", "", query)
    # Clean up multiple spaces
    query = re.sub(r"\s+", " ", query).strip()
    # Remove trailing/leading AND/OR
    query = re.sub(r"^(AND|OR)\s+", "", query)
    query = re.sub(r"\s+(AND|OR)$", "", query)
    # Remove empty AND/OR patterns like "word AND AND word" or "OR OR"
    query = re.sub(r"\s+(AND|OR)\s+(AND|OR)\s+", " OR ", query)
    query = re.sub(r"^(AND|OR)\s+", "", query)
    # Handle "word OR OR word" -> "word OR word"
    query = re.sub(r"\s+OR\s+OR\s+", " OR ", query)
    return query


def rewrite_query_llm(question: str, model: str = None) -> str:
    """Use LLM to rewrite question as FTS5 query."""
    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    result = llm_call(prompt, model).strip()

    # LLM outputs valid FTS5 directly, just clean up any extra text
    # Remove any leading/trailing quotes or explanation
    result = result.strip("\"'")

    if not result or len(result) < 2:
        # Fallback to deterministic
        return clean_query_deterministic(question)

    # Sanitize for FTS5
    return sanitize_fts5_query(result)


def expand_query_llm(question: str, model: str = None) -> list:
    """Use LLM to generate multiple search queries for better recall."""
    prompt = QUERY_EXPAND_PROMPT.format(question=question)
    result = llm_call(prompt, model)

    # Parse lines as separate queries
    lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
    queries = []
    for line in lines[:3]:  # Max 3 queries
        # Clean each query - extract content words
        words = re.findall(r"\b\w+\b", line.lower())
        content = [w for w in words if w not in STOPWORDS and len(w) > 2]
        if content:
            queries.append(" OR ".join(content[:5]))

    # Always include deterministic query as fallback
    det_query = clean_query_deterministic(question)
    if det_query not in queries:
        queries.insert(0, det_query)

    return queries[:3]


def multi_search(db_path: str, queries: list, limit: int = 15) -> list:
    """Run multiple queries and combine results, deduplicating by ID."""
    seen_ids = set()
    all_memories = []

    for query in queries:
        result = adaptive_memory(
            db_path,
            "search",
            query,
            "-l",
            str(limit),
            "-c",
            str(CONTEXT_WINDOW),
            "--energy-decay",
            str(ENERGY_DECAY),
        )
        try:
            memories = json.loads(result).get("memories", [])
        except json.JSONDecodeError:
            continue

        for m in memories:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                all_memories.append(m)

    # Sort by energy (highest first) and limit
    all_memories.sort(key=lambda x: x.get("energy", 0), reverse=True)
    return all_memories[:limit]


def answer_question(db_path: str, question: str, mode: str, model: str = None) -> str:
    """Search and optionally use LLM to answer."""

    # Use deterministic query for all modes (simpler and more reliable)
    # LLM query expansion was tested but reduced F1 from 0.50 to 0.40
    search_query = clean_query_deterministic(question)

    result = adaptive_memory(
        db_path,
        "search",
        search_query,
        "-l",
        str(RETRIEVAL_LIMIT),
        "-c",
        str(CONTEXT_WINDOW),
        "--energy-decay",
        str(ENERGY_DECAY),
    )
    try:
        memories = json.loads(result).get("memories", [])
    except json.JSONDecodeError:
        memories = []

    if not memories:
        return "No information available"

    if mode == "fts5_only":
        # Just return top match text (baseline, no LLM)
        # Extract just the relevant part after speaker name
        text = memories[0]["text"]
        # Try to extract the answer part
        if ": " in text:
            text = text.split(": ", 1)[1]
        return text

    # Format context for LLM
    context = "\n".join([f"[{m['datetime']}] {m['text']}" for m in memories])
    return llm_call(QA_PROMPT.format(context=context, question=question), model)


def compute_f1(prediction: str, qa: dict) -> float:
    """Compute F1 score based on question category."""
    category = qa["category"]

    # Category 5 (adversarial) uses 'adversarial_answer' and correct response is "no info"
    if category == 5:
        if (
            "no information" in prediction.lower()
            or "not mentioned" in prediction.lower()
            or "no info" in prediction.lower()
        ):
            return 1.0
        else:
            return 0.0

    # Get answer (use 'answer' key, fallback to 'adversarial_answer')
    answer = str(qa.get("answer", qa.get("adversarial_answer", "")))

    # For open-domain questions, take first part before semicolon
    if category == 3:
        answer = answer.split(";")[0].strip()

    if category == 1:  # multi-hop
        return f1_multi(prediction, answer)
    else:  # single-hop, temporal, open-domain
        return f1_score(prediction, answer)


# --- Main evaluation loop ---


def evaluate_sample(
    sample: dict,
    mode: str,
    results_dir: Path,
    model: str = None,
    use_cached_db: bool = False,
) -> dict:
    """Evaluate all QAs for one conversation sample.

    If use_cached_db is True:
      - fts5_only, fts5_llm: use results/dbs/{sample_id}_no_rels.db
      - spreading_llm: use results/dbs/{sample_id}_with_rels.db
    """
    sample_id = sample["sample_id"]

    if use_cached_db:
        # Use pre-built DBs (read-only)
        dbs_dir = results_dir / "dbs"
        if mode == "spreading_llm":
            db_path = str(dbs_dir / f"{sample_id}_with_rels.db")
        else:
            db_path = str(dbs_dir / f"{sample_id}_no_rels.db")

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"Cached DB not found: {db_path}. Run without --use-cached-db first."
            )

        stats = json.loads(adaptive_memory(db_path, "stats"))
        memory_count = stats["memory_count"]
        relationship_count = stats["relationship_count"]
        print(
            f"  Using cached DB: {db_path} ({memory_count} memories, {relationship_count} rels)"
        )
    else:
        db_path = str(results_dir / f"conv_{sample_id}.db")

        # 1. Ingest
        memory_count = ingest_conversation(db_path, sample)
        print(f"  Ingested {memory_count} memories")

        # 2. Pre-tag (only for spreading_llm mode)
        relationship_count = 0
        if mode == "spreading_llm":
            print("  Pre-tagging relationships...")
            relationship_count = pretag_relationships(db_path, model)
            print(f"  Created {relationship_count} relationships")

    # 3. Answer all questions
    qa_count = len(sample["qa"])
    print(f"  Answering {qa_count} questions (parallel={PARALLEL_QA})...")

    def process_qa(qa):
        pred = answer_question(db_path, qa["question"], mode, model)
        score = compute_f1(pred, qa)
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        return {
            "question": qa["question"],
            "answer": str(answer),
            "prediction": pred,
            "category": qa["category"],
            "f1": round(score, 4),
        }

    predictions = []
    with ThreadPoolExecutor(max_workers=PARALLEL_QA) as executor:
        futures = {
            executor.submit(process_qa, qa): i for i, qa in enumerate(sample["qa"])
        }
        done = 0
        for future in as_completed(futures):
            predictions.append((futures[future], future.result()))
            done += 1
            if done % 50 == 0:
                print(f"    Progress: {done}/{qa_count}")

    # Sort by original order
    predictions = [p for _, p in sorted(predictions)]

    # Compute sample-level F1
    sample_f1 = (
        sum(p["f1"] for p in predictions) / len(predictions) if predictions else 0
    )

    return {
        "sample_id": sample_id,
        "memory_count": memory_count,
        "relationship_count": relationship_count,
        "f1": round(sample_f1, 4),
        "predictions": predictions,
    }


def run_benchmark(
    mode: str,
    samples: int = None,
    data_dir: Path = None,
    results_dir: Path = None,
    model: str = None,
    use_cached_db: bool = False,
) -> dict:
    """Run full benchmark."""
    data_dir = Path(data_dir or "data")
    results_dir = Path(results_dir or "results")
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = download_data(data_dir)
    if samples:
        data = data[:samples]

    # Evaluate each sample
    all_results = []
    for i, sample in enumerate(data):
        print(f"[{i + 1}/{len(data)}] Evaluating {sample['sample_id']}...")
        result = evaluate_sample(sample, mode, results_dir, model, use_cached_db)
        all_results.append(result)
        print(f"  Sample F1: {result['f1']:.3f}")

    # Aggregate metrics
    all_preds = [p for r in all_results for p in r["predictions"]]

    by_category = {}
    for cat, name in CATEGORY_NAMES.items():
        cat_preds = [p for p in all_preds if p["category"] == cat]
        if cat_preds:
            by_category[f"{cat}_{name}"] = {
                "f1": round(sum(p["f1"] for p in cat_preds) / len(cat_preds), 4),
                "count": len(cat_preds),
            }

    overall_f1 = sum(p["f1"] for p in all_preds) / len(all_preds) if all_preds else 0

    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": model or LLM_MODEL if mode != "fts5_only" else None,
        "retrieval_limit": RETRIEVAL_LIMIT,
        "context_window": CONTEXT_WINDOW,
        "overall_f1": round(overall_f1, 4),
        "by_category": by_category,
        "per_sample": [
            {k: v for k, v in r.items() if k != "predictions"} for r in all_results
        ],
        "predictions": all_preds,
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = results_dir / f"{timestamp}_{mode}.json"
    out_file.write_text(json.dumps(output, indent=2))

    # Print summary
    print(f"\nResults saved to {out_file}")
    print(f"Overall F1: {output['overall_f1']:.4f}")
    for cat_key, cat_data in by_category.items():
        print(f"  {cat_key}: {cat_data['f1']:.4f} ({cat_data['count']} questions)")

    return output


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="LoCoMo benchmark for adaptive_memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  fts5_only      Pure FTS5 search, return top match (no LLM)
  fts5_llm       FTS5 search + LLM reasoning (no relationships)
  spreading_llm  FTS5 + spreading activation + LLM (with pre-tagged relationships)
  all            Run all three modes sequentially

Examples:
  uv run python locomo_bench.py --mode all
  uv run python locomo_bench.py --mode fts5_llm --samples 1
  uv run python locomo_bench.py --mode spreading_llm --model gpt-4o
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["fts5_only", "fts5_llm", "spreading_llm", "all"],
        default="all",
        help="Evaluation mode (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Limit to N samples (for testing)",
    )
    parser.add_argument(
        "--model",
        default=LLM_MODEL,
        help=f"LLM model for reasoning (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--use-cached-db",
        action="store_true",
        help="Use pre-built DBs from results/dbs/ (avoids re-ingestion and re-tagging)",
    )
    args = parser.parse_args()

    if args.mode == "all":
        results = {}
        for mode in ["fts5_only", "fts5_llm", "spreading_llm"]:
            print(f"\n{'=' * 60}")
            print(f"Running mode: {mode}")
            print("=" * 60)
            results[mode] = run_benchmark(
                mode,
                samples=args.samples,
                model=args.model,
                use_cached_db=args.use_cached_db,
            )

        # Print comparison
        print(f"\n{'=' * 60}")
        print("COMPARISON")
        print("=" * 60)
        for mode, result in results.items():
            print(f"{mode:20s} F1: {result['overall_f1']:.4f}")
    else:
        run_benchmark(
            args.mode,
            samples=args.samples,
            model=args.model,
            use_cached_db=args.use_cached_db,
        )


if __name__ == "__main__":
    main()
