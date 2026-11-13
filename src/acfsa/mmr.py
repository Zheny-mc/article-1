from __future__ import annotations

import numpy as np
from .models import Candidate


def _normalized_relevance(candidates: list[Candidate]) -> np.ndarray:
    # Cosine similarity is in [-1, 1]; clip to [0, 1] without min-max
    # rescaling so the least-relevant candidate is not forced to zero.
    return np.clip(np.asarray([c.relevance for c in candidates], dtype=float), 0.0, 1.0)


def mmr_select(candidates: list[Candidate], *, k: int = 5, lambda_param: float = 0.7) -> list[Candidate]:
    """Greedy, deterministic Maximal Marginal Relevance selection."""
    if k <= 0 or not candidates:
        return []
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be in [0, 1]")

    relevance = _normalized_relevance(candidates)
    selected_indices: list[int] = []
    # A list preserves Retriever order for deterministic tie-breaking.
    remaining = list(range(len(candidates)))
    selected_matrix = None

    while remaining and len(selected_indices) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            if selected_matrix is not None:
                similarities = selected_matrix @ candidates[idx].embedding
                redundancy = max(0.0, float(np.max(similarities)))
            else:
                redundancy = 0.0
            score = lambda_param * float(relevance[idx]) - (1.0 - lambda_param) * redundancy
            # Strict comparison preserves the first candidate in Retriever order on ties.
            if score > best_score:
                best_score, best_idx = score, idx

        if best_idx is None:
            break
        candidates[best_idx].mmr_score = float(best_score)
        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        selected_matrix = np.vstack([candidates[j].embedding for j in selected_indices])

    return [candidates[i] for i in selected_indices]
