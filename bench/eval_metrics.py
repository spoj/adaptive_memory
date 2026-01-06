"""F1 evaluation metrics (ported from LoCoMo)."""

import re
import string
from collections import Counter

from nltk.stem import PorterStemmer

ps = PorterStemmer()


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = str(s).replace(",", "")
    # Remove articles
    s = re.sub(r"\b(a|an|the|and)\b", " ", s, flags=re.IGNORECASE)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Lowercase and normalize whitespace
    s = " ".join(s.lower().split())
    return s


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming."""
    pred_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    """F1 for multi-answer questions (comma-separated)."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]

    if not ground_truths:
        return 0.0

    scores = []
    for gt in ground_truths:
        best = max(f1_score(pred, gt) for pred in predictions) if predictions else 0.0
        scores.append(best)

    return sum(scores) / len(scores)
