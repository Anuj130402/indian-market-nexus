#!/usr/bin/env python
"""
Build the SEMANTIC EYES from the announcement corpus.

Embeds every announcement once, then uses it twice (the two embeddings that
empower each other):
  - company identity  = centroid of a company's announcement vectors  (news -> company)
  - daily novelty     = how far a day's news sits from that identity   (company -> news)
  - candidate edges   = company-vector cosine similarity               (Eyes PROPOSE)

Next phase (the Brain) confirms these candidate edges with statistics.

    # fast, offline lexical baseline first (validates on your real 69k in seconds):
    python scripts/09_build_semantic_eyes.py lexical
    # then the real semantic embeddings (MiniLM is small; a few min on CPU):
    pip install sentence-transformers
    python scripts/09_build_semantic_eyes.py transformer

Outputs: data/processed/news_semantic.parquet  (daily novelty signal)
         data/processed/candidate_edges.parquet (Eyes-proposed edges, for the Brain)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from config import settings, universe
from nexus.nlp import semantic as S

KIND = sys.argv[1] if len(sys.argv) > 1 else "lexical"


def get_embedder(kind):
    if kind == "transformer":
        from nexus.nlp.embeddings import TransformerEmbedder
        print("Loading sentence-transformer (first run downloads the model)...")
        return TransformerEmbedder()
    from nexus.nlp.embeddings import LexicalEmbedder
    return LexicalEmbedder()


def main():
    df = pd.read_parquet(settings.RAW_DIR / "announcements.parquet")
    df["text"] = df["text"].fillna("").astype(str)
    print(f"Loaded {len(df):,} announcements. Embedding with: {KIND}")

    emb = S.embed_announcements(df, get_embedder(KIND))
    tickers, cvec = S.company_embeddings(df, emb)
    print(f"Company identity vectors: {cvec.shape}")

    edges, diag = S.propose_edges(tickers, cvec, top_k=3, min_sim=0.30,
                                  sectors=universe.SECTORS)
    print(f"\nProposed {diag['n_edges']} candidate edges "
          f"(same-sector={diag.get('same_sector','?')}, "
          f"cross-sector={diag.get('cross_sector','?')})")
    print("\nTop CROSS-sector candidate links (the potentially non-obvious ones):")
    shown = 0
    for a, b, s in edges:
        if universe.SECTORS.get(a) != universe.SECTORS.get(b):
            print(f"  {a:<14} ~ {b:<14} sim={s:.3f}  "
                  f"[{universe.SECTORS.get(a,'?')} / {universe.SECTORS.get(b,'?')}]")
            shown += 1
        if shown >= 10:
            break

    daily = S.daily_semantic_signal(df, emb, tickers, cvec)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(settings.PROCESSED_DIR / "news_semantic.parquet")
    pd.DataFrame(edges, columns=["a", "b", "similarity"]).to_parquet(
        settings.PROCESSED_DIR / "candidate_edges.parquet")
    print(f"\nSaved daily novelty ({len(daily):,} rows) and {len(edges)} candidate edges.")
    print("Novelty distribution:")
    print(daily["news_novelty"].describe()[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()