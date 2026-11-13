from __future__ import annotations
from .models import Candidate


def build_context(candidates: list[Candidate]) -> str:
    blocks = []
    for rank, candidate in enumerate(candidates, start=1):
        chunk = candidate.chunk
        blocks.append(
            f"[Фрагмент {rank} | источник: {chunk.source} | doc_id: {chunk.doc_id} "
            f"| chunk_id: {chunk.chunk_id} | relevance: {candidate.relevance:.4f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)
