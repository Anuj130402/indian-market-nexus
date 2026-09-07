# The Indian Market Nexus

> A directed **causal graph of the Indian equity market (NSE/BSE)** that models how shocks propagate between stocks - built with the discipline that its evaluation is honest enough to reject its own hypotheses.

**TL;DR of the findings.** Across a full research arc I established that *directional* shock propagation on the Nifty is efficient and **not** predictable (a coin flip, four different ways), but that **volatility spillover is partially predictable** through network structure (AUC ≈ 0.58, out-of-sample). I also showed - with a clean negative result - that regulatory-filing text is too homogeneous to propose economic graph structure. The project is now pivoting to richer data (annual reports + real news) and an event-study methodology; see [Roadmap](#roadmap).

*Research & engineering project. Not investment advice.*

---

## What it is

The market is modeled as a **directed, weighted graph**: nodes are companies (48 Nifty constituents), an edge `A → B` means "a shock in A tends to propagate to B," and every edge carries the *evidence* that created it (so nothing is a black box). Price statistics and NLP each propose and confirm edges; a machine-learning layer then tests whether the graph actually predicts anything **out-of-sample**.

Guiding principle - **"Eyes before Brain":** text/NLP *proposes* candidate edges (economic meaning); statistics *confirm* them (significance). An edge is kept only if it is real on both.

## Architecture

```
 Price data (EOD OHLCV, NSE via yfinance)     Text data (BSE announcements; soon: annual reports + news)
            │                                              │
            ▼                                              ▼
      STATS (the "Brain")                            NLP (the "Eyes")
 cointegration · transfer entropy · Kalman     sentiment · embeddings · relation extraction
            │                                              │
            └──────────────►  GRAPH BUILDER  ◄─────────────┘
                    directed graph, evidence per edge, Eyes-propose → Brain-confirm (AND + FDR)
                                     │
        ┌────────────────────────────┼───────────────────────────┐
        ▼                           ▼                             ▼
  discovery / centrality      propagation → signal          ML models (GBM / logistic)
                                     │                     look-ahead-safe evaluation
                                     ▼
                          EVALUATION (out-of-sample, AUC + permutation importance)
```

Clean `src/`-layout Python package (`config/`, `src/nexus/{data,stats,graph,nlp,propagation,ml}/`, `scripts/`, `tests/`). Every stage is isolated so data sources or the graph store can be swapped without touching the science.

## How it was built (pipeline)

| Stage | What it does | Key methods |
|---|---|---|
| **Data** | EOD prices (48 stocks, 2018–2025, spanning the COVID crash); 69.5k BSE announcements | `yfinance`, `bse`, Parquet cache |
| **Statistical edges** | build the directed graph from prices | Engle–Granger **cointegration**, **transfer entropy** with surrogate-permutation significance, **Kalman** dynamic hedge ratio |
| **Graph** | fuse edges, rank importance | NetworkX, PageRank / in-out-strength / betweenness centrality |
| **NLP (Eyes)** | sentiment + semantic representation | **FinBERT** sentiment; **sentence-transformer** company embeddings + a *news-novelty* signal |
| **Confirmation** | keep only edges that are both proposed and statistically real | strict **AND + Benjamini–Hochberg FDR** |
| **ML** | test whether the graph predicts | Gradient Boosting + logistic baseline, 14 graph/price features (+ news features) |
| **Evaluation** | honest, look-ahead-safe scoring | train pre-2023 / test 2023+, **ROC-AUC**, **permutation importance** |

**Methodological commitments (throughout):** strict temporal train/test split (no look-ahead); point-in-time graphs for any walk-forward; AUC over accuracy (classes are imbalanced); permutation importance to interpret *what* a model uses, not just its score; synthetic ground-truth tests (planted lead-lag / cointegration / independent series) validate every component before it touches real data.

## Results (the intellectual core)

All evaluated **out-of-sample** (fit on pre-2023, tested on 2023–2025).

1. **Direction is efficient - not predictable.** A hand-designed contagion rule scored a **0.50** hit-rate; a Gradient-Boosted model on 14 features reached only **AUC 0.53**, and permutation importance showed it *ignored the graph entirely*, leaning on market volatility. Direction isn't graph-predictable.
2. **Volatility spillover IS partially predictable - the one real positive result.** Reframing the target from *which way* to *how big* (an outsized move) lifted out-of-sample **AUC to ≈ 0.58**. The dominant driver is each stock's own volatility persistence, but graph **centrality features carry real, non-zero importance** (they were exactly zero for direction) - a genuine, if second-order, network effect.
3. **News as a direction feature: no lift (+0.001 AUC).** Adding FinBERT sentiment to the direction model, in a controlled same-rows comparison, changed nothing - the efficient-market hypothesis appearing in my own data (public filings are priced by the time they're disclosed).
4. **Semantic edges from regulatory filings: a clean negative result.** Company embeddings built from BSE announcements do **not** separate companies economically - the corpus is boilerplate-dominated (both lexical and transformer embedders collapsed all 48 companies into a narrow 0.86–0.98 similarity band; same-sector rate ≈ 14%). Strict AND+FDR confirmation collapsed the proposed edges; a lenient filter had surfaced economically absurd links (e.g. Cipla → Hero MotoCorp). **Conclusion: filing text is too homogeneous to propose structure.**

## Where it fell short (honest limitations)

- **Regulatory announcements were the wrong text source** for structure - established empirically above.
- **Beta confounding (open question).** The volatility/centrality result may partly reflect centrality proxying for a stock's **beta** to the Nifty; it needs market-orthogonalized returns to be sure it's a true network effect (planned - see roadmap).
- **Common-factor bias.** Statistical edges can't yet distinguish a genuine A→B spillover from a shared *sector/macro* shock hitting both (e.g. a steel import-duty change moving Tata Steel and JSW together). Disentangling this is a core roadmap item.
- **Cointegration is used strictly** (rarely fires on raw equities); the graph leans more on transfer entropy for direction.
- **Daily granularity is coarse** - shocks digest over multiple days, not one session.

## Roadmap

The next phase upgrades the project from a static mapping exercise into a **dynamic, event-driven engine**: annual reports lay the *structural plumbing* (who is economically linked); news/sentiment is the *water* (the shock) flowing through it. This direction was sharpened by an external methodological review; the additions below are **planned**, with rationale.

**Better data.**
- *Structural (annual reports):* scrape MD&A and Related-Party-Transactions sections (Screener.in / BSE), parse with `pdfplumber` - the sections that actually name customers, suppliers, and competitors.
- *Temporal (news):* historical Indian financial news via open datasets (Kaggle/Hugging Face) and the **GDELT** Global Knowledge Graph (BigQuery free tier), fetched with `newspaper3k`.

**Better Eyes.** Replace embedding-similarity with **LLM zero-shot relation extraction** - a local model (e.g. Llama-3 8B via Ollama) prompted to emit strict JSON `{source, target, relationship}` for supply-chain / competitor / JV links from MD&A. Feed directly into `candidate_edges.parquet`.

**Better statistics (the key methodological fixes).**
- **Orthogonalize returns against the Nifty (and sector) index before measuring propagation** - strips out market/sector beta so a surviving edge reflects a *localized* network effect, not shared systemic exposure. This directly de-risks the beta-confounder above.
- Confirm LLM-proposed edges with **beta-adjusted transfer entropy** under the existing strict AND+FDR gate.

**Better targets & metrics (event-study framing).**
- Shift the target from a daily spike to **Cumulative Abnormal Volatility (CAV)** over multi-day horizons `[t+1, t+h], h ∈ {1,3,5}`, measured *after* subtracting a GARCH/beta baseline.
- Add the **Diebold–Yilmaz spillover index** as a standard network-connectedness benchmark (comparing a purely-statistical market to the text-enhanced graph). *(Noted as a complementary, largely linear/contemporaneous measure - used alongside, not instead of, transfer entropy.)*

**Shock taxonomy.** Tag every event as **endogenous** (firm-specific: earnings, MD&A, governance - propagates along edges) or **exogenous/systemic** (policy, RBI, tariffs - a *broadcast* shock to a whole sub-graph), so systemic co-moves aren't mistaken for peer spillover.

**Validation stays strict.** Baselines (GARCH, event-response) calibrated only on pre-2023; all CAV/spillover predictions scored blindly on the 2023+ partition.

## Tech stack

Python · pandas / NumPy · statsmodels · NetworkX · scikit-learn · transformers / FinBERT · sentence-transformers · yfinance · `bse` · Matplotlib · pytest. *Planned:* Ollama (local LLM), pdfplumber, GDELT/BigQuery, `arch` (GARCH).

## Repo layout

```
config/        settings, universe (tickers, sectors, groups)
src/nexus/     data · stats · graph · nlp · propagation · ml
scripts/       01_fetch_market_data … 10_confirm_edges
data/          raw/ (prices, announcements)  processed/ (signals, candidate edges, graph)  [gitignored]
tests/         synthetic ground-truth tests
```

---

*Built as an exercise in rigorous, honest research - the kind where a negative result is a finding, not a failure. Not an investment advice.*
