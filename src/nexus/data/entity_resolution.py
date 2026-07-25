"""
Entity resolution: messy text mentions -> canonical ticker.

"Tata Motors", "TATAMOTORS", "Tata Motors Ltd", "Tata Motors Limited" must all
collapse to TATAMOTORS.NS. Section 3d chose: hand-built map for the small v1
universe, fuzzy fallback for when the universe grows to Nifty 500.

STATUS: functional (exact + fuzzy). Relation extraction (nlp/) will call
`resolve()` on every company span it pulls out of text.
"""
from __future__ import annotations

import re
import sys
from difflib import get_close_matches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import universe  # noqa: E402

_SUFFIXES = (" limited", " ltd", " ltd.", " corp", " corporation", " inc", " plc")


def _normalise(name: str) -> str:
    name = name.lower().strip()
    for suf in _SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)].strip()
    return re.sub(r"[^a-z0-9& ]", "", name)


# Pre-normalise the known display names once.
_LOOKUP = {_normalise(n): t for n, t in universe.name_to_ticker().items()}


def resolve(mention: str, fuzzy_cutoff: float = 0.85) -> str | None:
    """
    Map a raw company mention to a ticker, or None if we can't confidently.

    fuzzy_cutoff is intentionally strict: a wrong edge is worse than a missing
    one, because a bad edge silently corrupts every downstream propagation path.
    """
    key = _normalise(mention)
    if key in _LOOKUP:
        return _LOOKUP[key]
    match = get_close_matches(key, _LOOKUP.keys(), n=1, cutoff=fuzzy_cutoff)
    return _LOOKUP[match[0]] if match else None
