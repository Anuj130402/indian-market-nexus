"""
Information-content quantification (Section 3c, rank 2 — the FinBERT idea).

Not every news item matters. This scores how *surprising* a piece of text is
relative to prior consensus — the "information content" that actually moves
markets. High-surprise items are the shock triggers worth propagating.

STATUS: STUB. Recommended path: FinBERT (or an Indian-finance-tuned model) for
sentiment/polarity, then measure deviation from a rolling consensus baseline.
"""
from __future__ import annotations


def information_content(text: str, consensus_context: str | None = None) -> float:
    """STUB: return a surprise score in [0, 1]. 0 = fully expected, 1 = shock."""
    raise NotImplementedError("information_content — v2 milestone (needs FinBERT).")
