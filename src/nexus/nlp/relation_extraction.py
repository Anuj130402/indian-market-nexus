"""
Relation Extraction — turn text into typed, directed edges (Section 3c, rank 3).

Output feeds nexus.graph.builder.add_nlp_edges as tuples:
    (src_ticker, dst_ticker, relation_type, confidence)
where relation_type is one of the five from Section 3b:
    supplier_customer | parent_subsidiary | competitor | same_sector | lender_borrower

STATUS: STUB. Recommended v1 path:
  - Dependency-pattern / rule-based extraction over sentences containing >=2
    resolved company mentions (high precision, explainable — matches your
    'understand the reasoning' preference).
  - v2: fine-tune a relation-classification model (e.g. a HF sentence classifier)
    on a small hand-labelled set of Indian filings.
"""
from __future__ import annotations

RELATION_TYPES = (
    "supplier_customer",
    "parent_subsidiary",
    "competitor",
    "same_sector",
    "lender_borrower",
)


def extract_relations(text: str) -> list[tuple[str, str, str, float]]:
    """
    STUB: return typed directed relations found in `text`.
    Implement rule-based extraction first; return [] for now so the graph
    builder can call this safely today.
    """
    return []
