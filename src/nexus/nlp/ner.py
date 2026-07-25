"""
Named Entity Recognition — find company mentions in text (Section 3c, rank 1).

STATUS: STUB. Recommended v1 path:
  - Start with a gazetteer/dictionary matcher over the universe names (high
    precision, trivially explainable) via nexus.data.entity_resolution.resolve.
  - Then layer a finance-tuned NER model (spaCy, or a HF token-classifier) to
    catch mentions the dictionary misses.
"""
from __future__ import annotations

from nexus.data.entity_resolution import resolve


def extract_company_mentions(text: str) -> list[str]:
    """
    Return the tickers of companies mentioned in `text`.

    STUB: a naive dictionary pass to prove the interface end-to-end. Replace with
    real NER + entity linking. Kept intentionally simple so the pipeline runs.
    """
    found = set()
    for token in _candidate_spans(text):
        ticker = resolve(token)
        if ticker:
            found.add(ticker)
    return sorted(found)


def _candidate_spans(text: str, max_words: int = 4) -> list[str]:
    """Yield capitalised n-grams as crude company-name candidates."""
    words = text.split()
    spans = []
    for n in range(1, max_words + 1):
        for i in range(len(words) - n + 1):
            span = " ".join(words[i:i + n])
            if span[:1].isupper():
                spans.append(span)
    return spans
