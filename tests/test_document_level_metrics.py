import numpy as np
from acfsa.metrics import retrieval_metrics
from acfsa.models import Chunk, Candidate


def candidate(chunk_id, doc_id):
    chunk = Chunk(chunk_id, doc_id, f"text {chunk_id}", f"{doc_id}.md")
    return Candidate(chunk, 0.9, np.array([1.0, 0.0]))


def test_multiple_chunks_from_one_document_count_once():
    ranked = [
        candidate("a-1", "doc-a"),
        candidate("a-2", "doc-a"),
        candidate("b-1", "doc-b"),
    ]
    metrics = retrieval_metrics(ranked, {"doc-a", "doc-b"}, k=2)
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_document_metrics_preserve_rank_of_unique_sources():
    ranked = [
        candidate("x-1", "irrelevant"),
        candidate("a-1", "relevant"),
        candidate("a-2", "relevant"),
    ]
    metrics = retrieval_metrics(ranked, {"relevant"}, k=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
