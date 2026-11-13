from __future__ import annotations
import numpy as np


class E5Embedder:
    """Thin adapter for multilingual E5 embeddings."""
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = [f"query: {query}" for query in queries]
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        texts = [f"passage: {text}" for text in passages]
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
