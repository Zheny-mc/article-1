from __future__ import annotations
import numpy as np
from .models import Chunk, Candidate


def retrieve(query: str, chunks: list[Chunk], chunk_embeddings: np.ndarray, embedder, top_n: int = 20) -> list[Candidate]:
    query_vector = embedder.encode_queries([query])[0]
    scores = chunk_embeddings @ query_vector
    order = np.argsort(-scores)[:max(0, top_n)]
    return [Candidate(chunks[int(i)], float(scores[int(i)]), chunk_embeddings[int(i)]) for i in order]
