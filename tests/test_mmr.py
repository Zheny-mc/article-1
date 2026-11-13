import numpy as np
from acfsa.models import Chunk, Candidate
from acfsa.mmr import mmr_select


def make_candidate(cid, relevance, embedding):
    chunk = Chunk(cid, cid, f"text {cid}", f"{cid}.md")
    vector = np.asarray(embedding, dtype=float)
    vector /= np.linalg.norm(vector)
    return Candidate(chunk, relevance, vector)


def test_mmr_prefers_less_redundant_candidate():
    candidates = [
        make_candidate("a", 0.92, [1.0, 0.0]),
        make_candidate("b", 0.90, [0.99, 0.10]),
        make_candidate("c", 0.87, [0.0, 1.0]),
    ]
    selected = mmr_select(candidates, k=2, lambda_param=0.7)
    assert [item.chunk.chunk_id for item in selected] == ["a", "c"]


def test_mmr_empty_and_limits():
    assert mmr_select([], k=3) == []
    candidates = [make_candidate("a", 0.8, [1, 0])]
    assert len(mmr_select(candidates, k=10)) == 1
