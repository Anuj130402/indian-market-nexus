"""
The BRAIN: confirm Eyes-proposed edges with statistics. STRICT version.

An edge survives ONLY if it is significant on BOTH cointegration AND transfer
entropy, each corrected for multiple comparisons (Benjamini-Hochberg FDR across
all candidate pairs). This is the honest bar: no lenient OR, no uncorrected
p-value fishing across 100+ tests.
"""
import numpy as np, pandas as pd, networkx as nx
from statsmodels.tsa.stattools import coint


def _bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg: boolean mask of p-values significant at FDR=alpha."""
    p = np.asarray(pvals, float); m = len(p)
    if m == 0: return np.zeros(0, bool)
    order = np.argsort(p)
    passed = p[order] <= alpha * (np.arange(1, m + 1) / m)
    mask = np.zeros(m, bool)
    if passed.any():
        mask[order[: np.max(np.where(passed)[0]) + 1]] = True
    return mask


def _te_p(s, t, bins=4, n_sur=100, seed=0):
    s=np.asarray(s); t=np.asarray(t); n=len(s)
    def H(c):
        tot=c.sum()
        if tot<=0: return 0.0
        p=c/tot; p=p[p>0]; return float(-(p*np.log(p)).sum())
    def te1(s,t):
        def d(x): r=pd.Series(x).rank(method="first").to_numpy(); e=np.linspace(0,len(x),bins+1); return np.digitize(r,e[1:-1])
        x=d(s); y=d(t); yt,yp,xp=y[1:],y[:-1],x[:-1]
        jyy=np.histogram2d(yt,yp,bins=bins)[0]; hyp=np.histogram(yp,bins=bins)[0]
        jyyx=np.bincount((yt*bins*bins)+(yp*bins)+xp,minlength=bins**3).astype(float)
        jyx=np.bincount((yp*bins)+xp,minlength=bins**2).astype(float)
        return max(0.0,(H(jyy)-H(hyp))-(H(jyyx)-H(jyx)))
    if n<20: return 0.0,1.0
    obs=te1(s,t); rng=np.random.default_rng(seed)
    null=np.array([te1(np.roll(s,int(rng.integers(1,n))),t) for _ in range(n_sur)])
    return obs,(np.sum(null>=obs)+1)/(n_sur+1)


def confirm_edges(candidates, prices, returns, alpha=0.05, min_obs=250, require="and"):
    recs = []
    for _, row in candidates.iterrows():
        a, b = row["a"], row["b"]
        if a not in prices.columns or b not in prices.columns: continue
        pa, pb = prices[a].dropna(), prices[b].dropna()
        idx = pa.index.intersection(pb.index)
        if len(idx) < min_obs: continue
        try: _, cp, _ = coint(pa.loc[idx], pb.loc[idx])
        except Exception: cp = 1.0
        ra, rb = returns[a].dropna(), returns[b].dropna()
        ri = ra.index.intersection(rb.index)
        te_ab, pab = _te_p(ra.loc[ri].to_numpy(), rb.loc[ri].to_numpy())
        te_ba, pba = _te_p(rb.loc[ri].to_numpy(), ra.loc[ri].to_numpy())
        if te_ab >= te_ba: src, dst, tep = a, b, pab
        else:              src, dst, tep = b, a, pba
        recs.append({"src": src, "dst": dst, "sim": float(row["similarity"]),
                     "coint_p": float(cp), "te_p": float(tep)})

    if not recs:
        return nx.DiGraph(), 0
    coint_ok = _bh_reject([r["coint_p"] for r in recs], alpha)   # FDR-corrected
    te_ok    = _bh_reject([r["te_p"]    for r in recs], alpha)   # FDR-corrected
    keep = (coint_ok & te_ok) if require == "and" else (coint_ok | te_ok)

    g = nx.DiGraph()
    for r, k in zip(recs, keep):
        if not k: continue
        g.add_edge(r["src"], r["dst"], weight=r["sim"] * (1 - max(r["coint_p"], r["te_p"])),
                   similarity=r["sim"], coint_p=r["coint_p"], te_p=r["te_p"])
    return g, int(keep.sum())