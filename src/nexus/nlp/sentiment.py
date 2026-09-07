"""
Sentiment scoring for announcement text — the temporal "Eyes" signal.

Two scorers behind one interface, so you can validate the whole pipeline fast
and then swap in the real model without changing anything downstream:

  LexiconScorer  - instant, rule-based. For plumbing tests and offline dev.
  FinBERTScorer  - ProsusAI/finbert (a BERT fine-tuned on financial text).
                   The real signal. Needs a one-time ~440MB model download
                   (internet), and is much faster on a GPU (this is the Colab job).

Both return a POLARITY in [-1, 1] per text: +1 very positive, -1 very negative.
"""
from __future__ import annotations
import numpy as np

# --- a small finance sentiment lexicon (for the fast fallback scorer) ---
_POS = {"gain","gains","profit","growth","surge","rise","rises","up","beat","beats",
        "record","strong","approval","approved","award","awarded","order","orders","win","wins",
        "won","expansion","dividend","bonus","upgrade","upgraded","high","boost","robust",
        "outperform","rally","jump","secures","secured","positive","higher"}
_NEG = {"loss","losses","decline","declines","fall","falls","drop","drops","down","weak",
        "miss","misses","cut","cuts","downgrade","downgraded","resign","resigns","resigned",
        "fraud","probe","penalty","fine","fined","default","delay","delays","litigation",
        "warning","warn","concern","concerns","low","plunge","slump","layoff","layoffs",
        "lawsuit","negative","lower","halt","suspended"}


class SentimentScorer:
    def score(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class LexiconScorer(SentimentScorer):
    """Fast, dependency-free. polarity = (pos-neg)/(pos+neg) over matched words."""
    def score(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts))
        for i, t in enumerate(texts):
            toks = "".join(c.lower() if c.isalnum() else " " for c in str(t)).split()
            p = sum(w in _POS for w in toks)
            n = sum(w in _NEG for w in toks)
            out[i] = (p - n) / (p + n) if (p + n) else 0.0
        return out


class FinBERTScorer(SentimentScorer):
    """
    ProsusAI/finbert via transformers. polarity = P(positive) - P(negative).
    Install: pip install transformers torch
    First run downloads the model (~440MB). Use a GPU (Colab) for the full 69k.
    """
    def __init__(self, batch_size: int = 32, max_length: int = 64, device: int = -1):
        from transformers import pipeline  # lazy import
        self.clf = pipeline("text-classification", model="ProsusAI/finbert",
                            top_k=None, truncation=True, max_length=max_length, device=device)
        self.batch_size = batch_size

    def score(self, texts: list[str]) -> np.ndarray:
        texts = [str(t) if t else "" for t in texts]
        results = self.clf(texts, batch_size=self.batch_size)
        out = np.zeros(len(results))
        for i, r in enumerate(results):
            # r is a list of {label, score} for positive/negative/neutral
            scores = {d["label"].lower(): d["score"] for d in r}
            out[i] = scores.get("positive", 0.0) - scores.get("negative", 0.0)
        return out