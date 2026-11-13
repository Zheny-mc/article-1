from pathlib import Path
from acfsa.loader import load_documents
from acfsa.metrics import retrieval_metrics, token_count
from acfsa.models import Chunk, Candidate
import numpy as np


def test_loader_reads_markdown_and_assigns_stable_ids(tmp_path: Path):
    (tmp_path / "example.md").write_text("# Title\n\nFirst paragraph.\n\nSecond paragraph.", encoding="utf-8")
    chunks = load_documents(tmp_path, max_chars=100, overlap_chars=10)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "example"
    assert chunks[0].chunk_id == "example::0001"


def test_token_count_is_positive():
    assert token_count("A short test sentence.") > 0


def test_retrieval_metrics_handles_empty_relevant_set():
    c = Candidate(Chunk("x", "doc", "text", "doc.md"), 0.8, np.array([1.0, 0.0]))
    metrics = retrieval_metrics([c], set(), 5)
    assert metrics["precision_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0
