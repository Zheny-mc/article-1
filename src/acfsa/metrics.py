from __future__ import annotations
import re
import math


def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Transparent approximation if tokenizer package/encoding is unavailable.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _unique_documents(candidates, k: int):
    """Keep the first-ranked occurrence of each document ID."""
    unique = []
    seen = set()
    for candidate in candidates:
        doc_id = candidate.chunk.doc_id
        if doc_id not in seen:
            unique.append(candidate)
            seen.add(doc_id)
            if len(unique) >= k:
                break
    return unique


def retrieval_metrics(candidates, relevant_doc_ids: set[str], k: int) -> dict[str, float]:
    """Document-level metrics; multiple chunks from one source count once."""
    if k <= 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

    ranked = _unique_documents(candidates, k)
    retrieved = [candidate.chunk.doc_id for candidate in ranked]
    rel_hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in retrieved]
    hits = sum(rel_hits)
    precision = hits / k
    recall = len(set(retrieved) & relevant_doc_ids) / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    reciprocal_rank = 0.0
    dcg = 0.0
    for rank, is_relevant in enumerate(rel_hits, start=1):
        if is_relevant and reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / rank
        dcg += is_relevant / math.log2(rank + 1)

    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def mean_pairwise_similarity(candidates) -> float:
    if len(candidates) < 2:
        return 0.0
    import numpy as np
    vectors = np.vstack([candidate.embedding for candidate in candidates])
    matrix = vectors @ vectors.T
    upper = matrix[np.triu_indices(len(candidates), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0
