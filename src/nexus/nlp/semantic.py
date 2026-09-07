"""
The semantic "Eyes": company identity + news novelty + candidate edge proposal.

Pipeline (embed once, use twice — the two embeddings empower each other):
  1. embed every announcement            -> a vector per announcement
  2. company_embeddings: centroid per co  -> STATIC identity (news -> company)
  3. daily_semantic_signal: per (co,day)  -> NOVELTY vs the company centroid
                                             (company -> news: is today unusual?)
  4. propose_edges: company-vector cosine  -> candidate STRUCTURAL edges
                                             (Eyes propose; Brain confirms later)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def embed_announcements(df: pd.DataFrame, embedder, text_col="text") -> np.ndarray:
    return np.asarray(embedder.embed(df[text_col].fillna("").astype(str).tolist()))


def _normalise(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def company_embeddings(df, emb, ticker_col="ticker") -> tuple[list[str], np.ndarray]:
    """Static identity = L2-normalised centroid of each company's announcement vectors."""
    tickers, vecs = [], []
    for tk, idx in df.groupby(ticker_col).indices.items():
        tickers.append(tk)
        vecs.append(emb[idx].mean(axis=0))
    return tickers, _normalise(np.vstack(vecs))


def daily_semantic_signal(df, emb, tickers, company_emb,
                          ticker_col="ticker", date_col="date") -> pd.DataFrame:
    """
    Per (company, day): news_novelty = 1 - cos(day-centroid, company-centroid).
    High = the day's news is semantically unlike the company's usual output.
    """
    cmap = {t: company_emb[i] for i, t in enumerate(tickers)}
    d = df.copy()
    d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=["date"])
    d["_row"] = np.arange(len(d))
    rows = []
    for (tk, day), grp in d.groupby([ticker_col, d["date"].dt.normalize()]):
        day_vec = emb[grp["_row"].to_numpy()].mean(axis=0)
        day_vec = day_vec / (np.linalg.norm(day_vec) or 1)
        novelty = 1.0 - float(np.dot(day_vec, cmap[tk]))
        rows.append({"ticker": tk, "date": day, "news_count": len(grp),
                     "news_novelty": novelty})
    return pd.DataFrame(rows)


def propose_edges(tickers, company_emb, top_k=3, min_sim=0.30, sectors=None):
    """
    For each company, its top_k most semantically-similar peers (above min_sim)
    become candidate edges. Returns (edges, diagnostics).
    edges: list of (a, b, similarity). diagnostics reports same-sector share so
    we can see whether similarity is finding real links or just re-discovering sectors.
    """
    sim = company_emb @ company_emb.T
    np.fill_diagonal(sim, -1.0)
    edges = []
    n = len(tickers)
    for i in range(n):
        order = np.argsort(sim[i])[::-1][:top_k]
        for j in order:
            if sim[i, j] >= min_sim:
                a, b = tickers[i], tickers[j]
                edges.append((a, b, float(sim[i, j])))
    # dedupe undirected pairs keeping max sim
    best = {}
    for a, b, s in edges:
        key = tuple(sorted((a, b)))
        best[key] = max(best.get(key, 0), s)
    edges = [(a, b, s) for (a, b), s in best.items()]

    diag = {"n_edges": len(edges)}
    if sectors:
        same = sum(1 for a, b, _ in edges if sectors.get(a) == sectors.get(b))
        diag["same_sector"] = same
        diag["cross_sector"] = len(edges) - same
    return sorted(edges, key=lambda e: -e[2]), diag