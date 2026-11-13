from __future__ import annotations

import numpy as np
from .models import Candidate


def _normalized_relevance(candidates: list[Candidate]) -> np.ndarray:
    # Cosine similarity is in [-1, 1]; clip to [0, 1] without min-max
    # rescaling so the least-relevant candidate does not get forced to zero.
    return np.clip(np.asarray([c.relevance for c in candidates], dtype=float), 0.0, 1.0)


def mmr_select(candidates: list[Candidate], *, k: int = 5, lambda_param: float = 0.7) -> list[Candidate]:
    """Greedy Maximal Marginal Relevance selection.

    MMR(d) = lambda * relevance(d,Q)
             - (1-lambda) * max_similarity(d, already_selected).
    Embeddings must be L2-normalized. Negative pair similarities are clipped to 0
    because negative cosine should not create a reward larger than the relevance term.
    """
    if k <= 0 or not candidates:
        return []
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be in [0, 1]")

    relevance = _normalized_relevance(candidates)
    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected_indices) < k:
        best_idx = None
        best_score = -float("inf")
        for idx in remaining:
            if selected_indices:
                selected_embeddings = np.vstack([candidates[j].embedding for j in selected_indices])
                similarities = selected_embeddings @ candidates[idx].embedding
                redundancy = max(0.0, float(np.max(similarities)))
            else:
                redundancy = 0.0
            score = lambda_param * float(relevance[idx]) - (1.0 - lambda_param) * redundancy
            if score > best_score:
                best_score, best_idx = score, idx
        assert best_idx is not None
        candidates[best_idx].mmr_score = float(best_score)
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]
