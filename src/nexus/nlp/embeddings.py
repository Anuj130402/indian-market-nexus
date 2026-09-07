"""
Text embedding models — turn text into meaning-vectors.

Two behind one interface (same two-tier pattern as sentiment):
  LexicalEmbedder  - instant, offline, stateless (hashed TF vectors). For plumbing
                     tests and a lexical baseline. Cosine similarity = word overlap.
  TransformerEmbedder - real SEMANTIC vectors via sentence-transformers
                     (default all-MiniLM-L6-v2). Needs a one-time model download;
                     GPU-friendly (the Colab job). Cosine similarity = meaning overlap.

Both return L2-normalised vectors, so cosine similarity = a plain dot product.
"""
from __future__ import annotations
import numpy as np


class EmbeddingModel:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LexicalEmbedder(EmbeddingModel):
    def __init__(self, n_features: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=n_features, alternate_sign=False,
                                     norm="l2", stop_words="english")

    def embed(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform([str(t) for t in texts])
        return np.asarray(X.todense())


class TransformerEmbedder(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([str(t) for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=True)