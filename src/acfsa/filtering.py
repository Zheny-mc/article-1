from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    median: float
    mad: float
    robust_sigma: float
    kept_indices: list[int]


def adaptive_threshold(
    scores: Sequence[float],
    *,
    beta: float = 1.0,
    min_threshold: float = 0.35,
) -> ThresholdResult:
    """Compute a robust, query-specific relevance threshold.

    T(Q) = max(T_min, median(S) - beta * 1.4826 * MAD(S)).
    Scores are assumed to be cosine similarities from one fixed embedding model.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("scores must be a one-dimensional sequence")
    if len(values) == 0:
        return ThresholdResult(float("inf"), float("nan"), 0.0, 0.0, [])
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    if not -1.0 <= min_threshold <= 1.0:
        raise ValueError("min_threshold must be in [-1, 1] for cosine similarity")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(float(min_threshold), median - beta * robust_sigma)
    kept = np.flatnonzero(values >= threshold).astype(int).tolist()
    return ThresholdResult(threshold, median, mad, robust_sigma, kept)


def remove_exact_duplicates(candidates):
    """Remove exact text duplicates, keeping the highest-relevance occurrence."""
    unique = {}
    for candidate in sorted(candidates, key=lambda item: item.relevance, reverse=True):
        normalized = " ".join(candidate.chunk.text.casefold().split())
        if normalized and normalized not in unique:
            unique[normalized] = candidate
    return list(unique.values())


def filter_candidates(candidates, *, beta: float = 1.0, min_threshold: float = 0.35):
    """Return filtered candidates and a serializable audit record."""
    scores = [candidate.relevance for candidate in candidates]
    result = adaptive_threshold(scores, beta=beta, min_threshold=min_threshold)
    kept = [candidates[i] for i in result.kept_indices]
    kept = remove_exact_duplicates(kept)
    audit = {
        "threshold": result.threshold,
        "median": result.median,
        "mad": result.mad,
        "robust_sigma": result.robust_sigma,
        "input_count": len(candidates),
        "after_threshold_count": len(result.kept_indices),
        "after_exact_dedup_count": len(kept),
    }
    return kept, audit
