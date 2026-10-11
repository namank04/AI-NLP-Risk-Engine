# Signal Risk: AI/NLP Financial Risk Engine

**S&P Global & Crisil Campus Hackathon 2026**

| | |
|---|---|
| **Candidate Name** | Naman Karwa |
| **College Email ID** | namankarwa311@gmail.com |
| **College / Campus** | Indian Institute of Technology Guwahati |
| **Demo Video Link** | [YouTube, unlisted: link to be added] |
| **Slide Deck** | [docs/presentation.pdf](docs/presentation.pdf) |

![Overview](docs/screenshots/overview.png)

---

## 1. Project Overview

### Problem

Markets react to text (news, regulatory filings, social media) long before that information shows up in prices,
ratings or financial statements. Risk teams need that text as **structured, machine-readable signals** they can feed
into portfolio and capital decisions.

### Approach

**The engine** ingests four text sources and, for every *(item, company)* pair, produces:

| Output | What it is |
|---|---|
| **Sentiment** | Score in [−1, 1], **per entity**: "Apple beats, while Tesla misses" gives AAPL +0.71 and TSLA −0.95 |
| **Event class** | Multi-label over 10 classes (geopolitical, macroeconomic, credit event with ISDA-style subtypes, M&A, earnings, product, regulatory/legal, management, operational, none) |
| **Impact** | Score from 1 to 10, validated against realised abnormal stock returns |
| **Scope and channels** | Countries, sectors, commodities, currencies, plus the market channels affected (rates, spreads, equity, FX, commodities) |

Before any scoring, the ingestion layer handles what decides signal quality:
- **entity linking** with ambiguity handling ("apple pie" is not AAPL; "the Dow" is not Dow Inc);
- a **financial-relevance gate** for general news;
- **story clustering**, so syndicated copies count once and breadth of coverage becomes a feature;
- **novelty** and **source credibility**.

Two downstream applications subscribe to the live signal stream:

- **Module A: Sentiment-driven index rebalancer.** Turns per-company sentiment into Black-Litterman views, sizes them
  with Grinold's rule (α = IC × σ × z) and optimises a 20-stock index under a hard 2% ex-ante tracking-error budget,
  with weight caps and a no-trade band.
- **Module B: Event-triggered stress test.** A corroborated high-impact negative event triggers a scenario calibrated
  to a historical analogue (2008, 2020, 2022), scaled by the engine's impact score and targeted by its scope. The
  scenario is applied to a synthetic 408-position **wholesale banking book**:
  - loans are valued through **IFRS 9 expected credit loss**, with stressed PDs from the **Vasicek / Basel IRB** model;
  - bonds, swaps, FX and CDS are marked to market;
  - **CET1** is recomputed against the regulatory minimum and buffers.

  The page then shows factor attribution and the severity at which capital breaks (reverse stress test).

---

## 2. Architecture & Tech Stack

![Architecture](docs/architecture.png)

```
sources ──► ingestion ────────────────► NLP engine ─────────────────► signal store ──► REST + SSE ──► Module A (index)
GDELT GKG   timestamps, entity linking,  entity sentiment (FinBERT)     DuckDB                        Module B (stress)
EDGAR 8-K   relevance gate, MinHash      events (student + zero-shot)                                 React dashboard
FNSPID      story clusters, novelty,     impact (rule, event-study
tweets      source credibility           validated)
```

| Layer | Choices |
|---|---|
| NLP | PyTorch (CUDA optional), Hugging Face Transformers: ProsusAI/FinBERT, DeBERTa-v3 zero-shot NLI, fine-tuned FinBERT event student; datasketch MinHash LSH |
| Quant | pandas, NumPy, SciPy, scikit-learn (Ledoit-Wolf), cvxpy, LightGBM |
| Data & serving | DuckDB, Parquet, FastAPI (REST + server-sent events) |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS, Apache ECharts, TanStack Query, Motion |
| LLM teacher (optional) | Any OpenAI-compatible endpoint; Groq free tier (`openai/gpt-oss-120b`) by default. No paid APIs. |

### API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Score any text: per-entity sentiment, event, impact, scope |
| `GET /api/signals` | Stored signals, filterable by entity, event and minimum impact |
| `GET /api/stream` | Server-sent events: `signal`, `index`, `stress`, `status` |
| `GET /api/index/current`, `/history` | Module A weights, views, tracking error |
| `POST /api/stress/run` | Run a systemic or single-name scenario |
| `GET /api/stress/curve`, `/compare`, `/library` | Reverse stress curve, scenario comparison, calibrated library |
| `POST /api/replay/start`, `/stop` | Replay a historical news stream through the live engine |

Interactive docs: http://localhost:8000/docs once the backend runs.

### Repository layout

```
config/       universe, event taxonomy, stress scenarios (calibrated)
data/         replay streams, synthetic bank book, gold set, evaluation outputs (see data/README.md)
docs/         architecture, presentation, methodology notes, screenshots
frontend/     React dashboard
scripts/      data building, backtest, calibration, teacher and student training
src/riskengine/
  ingest/     GDELT, SEC EDGAR, FNSPID, tweets
  nlp/        entities, clustering, sentiment, events, impact, pipeline
  modules/    index/ (Module A), stress/ (Module B)
  market/     prices, FRED, event study
  api/        FastAPI app
tests/        66 pytest tests
```

---

## 3. Dataset Used

All data is **public or synthetic**. No S&P Global or Crisil proprietary data is used. Full sources, licences and
rebuild steps: [data/README.md](data/README.md).

| Source | Use |
|---|---|
| **GDELT 2.0 GKG** (15-minute global news files) | Live-style news with tone and themes; crisis replays |
| **SEC EDGAR 8-K filings** | Item codes act as authoritative event labels (e.g. 2.04 debt acceleration) |
| **FNSPID** (`Zihan1004/FNSPID`) | Ticker-tagged financial news, 2009–2023; backtest and gold set |
| **Stock tweets** (`mjw/stock_market_tweets`) | Social-media source |
| **twitter-financial-news-topic** | Public labelled data for the event student |
| **yfinance, FRED** | Prices for the event study and backtest; rates, credit spreads, VIX, oil for scenario calibration |
| **Synthetic bank book** (`data/synthetic/`) | 408 positions, 80 counterparties; the 20 universe companies appear as counterparties so company signals hit real exposures |

**Universe:** 20 S&P 100 names covering all 11 GICS sectors (AAPL, MSFT, NVDA, GOOGL, DIS, TSLA, NKE, GS, WFC, XOM, CVX,
MRK, ABBV, BA, CAT, KO, WMT, NEE, DOW, AMT), chosen for dense 2022–2023 news coverage
([docs/data_audit.md](docs/data_audit.md)).

**Assumptions:**
- Historical text is mostly date-stamped, so backtests trade on the next day.
- Benchmark weights use current shares outstanding.
- Rating PDs are rounded long-run averages from public literature.

---

## 4. Quickstart & Installation

**Requirements:** Python 3.11–3.13 and Node.js 20.19+. Tested on Windows 11 with Python 3.13 and Node 20. A GPU is
optional; on CPU the engine is slower, so pick the 1h/s replay speed.

### 1. Backend

```bash
git clone https://github.com/namank04/AI-NLP-Risk-Engine.git
cd AI-NLP-Risk-Engine

python -m venv .venv
.venv\Scripts\activate                     # macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # NVIDIA GPU: .../whl/cu128
pip install -r requirements.txt
pip install -e .

uvicorn riskengine.api.app:app --port 8000
```

### 2. Frontend (second terminal)

```bash
cd frontend
npm install
npm run dev                                # open http://localhost:5173
```

On first use, the backend downloads the NLP models from Hugging Face (about 1.5 GB) and daily prices from Yahoo
Finance. The replay streams, bank book, backtest and evaluation outputs are already in the repository.

### 3. Demo walkthrough

1. **Overview**: the engine card scores sample headlines live through `/api/analyze`. Press **Replay the SVB collapse**.
2. **Desk**: stories stream in grouped by cluster, with per-company tone, event mix and high-impact alerts.
3. **Index**: active weights re-sort with every rebalance, within the tracking-error budget.
4. **Stress**: when the engine fires a scenario, an alert appears and the CET1 gauge drains to the stressed ratio.
   Drag the severity slider to see projected CET1 from the model's reverse-stress curve.
5. **Analyze**: paste any text. Each entity's clause is highlighted with its sentiment.

| Stress (Module B) | Index (Module A) |
|---|---|
| ![Stress](docs/screenshots/stress.png) | ![Index](docs/screenshots/index.png) |

### Optional

```bash
python -m pytest                           # 66 tests
python scripts/train_students.py           # event student (~5 min on a GPU); used automatically once trained
cp .env.example .env                       # add a free Groq key, then:
python scripts/run_teacher.py gold         # LLM teacher labels; resumes daily within the free quota
```

---

## 5. Key Results & Domain Impact

Every number below comes from scripts in this repository; methodology is in `docs/`.

| Area | Result |
|---|---|
| **Impact score** ([impact_model.md](docs/impact_model.md)) | On 2,141 held-out stock-days (H2 2023), impact ranks realised 2-day abnormal moves with Spearman ρ = 0.059 (≈2.7 s.e.). A LightGBM alternative reached 0.072, but the paired-bootstrap gain was not significant (95% CI −0.029 to +0.057), so the transparent rule stays. |
| **Event classification** | Fine-tuned FinBERT student: macro-F1 **0.95** vs **0.48** for zero-shot NLI on 3,105 held-out finance tweets (six covered classes). |
| **Sentiment timing** ([backtest.md](docs/backtest.md)) | Sentiment agrees with the same-day abnormal move (ρ = +0.086) but, traded a day later, mildly reverses (IC −0.024, t −2.0): news is priced within a day. This is why Module A rebalances intraday on 15-minute timestamps. |
| **Module A**, 2022–2023 walk-forward | Realised tracking error 2.1% against a 2% budget; +1.1% p.a. active return, IR 0.52 (t 0.74, not significant), 5 bp costs. |
| **Module B calibration** | At full historical severity, CET1 (13.5% base) falls 3.4 pp (geopolitical), 4.7 pp (2022 rates), 6.4 pp (2008 credit) and 6.0 pp (2020 pandemic): instantaneous shocks, in line with regulatory severely adverse scenarios. |
| **Reverse stress** | The book falls below its 8% CET1 requirement at **84%** of Lehman-2008 severity, 90% of March-2020 and 120% of the 2022 rates shock; the geopolitical scenario never breaches up to 150%. |
| **SVB replay** (Mar 2023) | The systemic credit scenario triggers on the SVB liquidity crisis (10 Mar), then re-runs on Credit Suisse contagion (15 Mar) and the First Republic downgrade (20 Mar). Zero false alarms on a quiet 48-hour stream. |

**Why it matters:** a risk team sees a credible event within minutes, along with its capital impact on the bank and
the exposures driving it, instead of waiting for the next scheduled stress-testing cycle. The same signals rebalance a
portfolio under explicit risk limits.

---

## 6. Limitations & Next Steps

- **Impact is the weakest link.** It is validated but modest (ρ = 0.059). On the Ukraine replay the invasion day
  scored 6, just under the trigger of 7, so the first systemic run came on 27 February. Next: richer event features
  from LLM-teacher labels distilled into student models, measured on the 875-item gold set.
- **Event classes without public training data.** Credit, operational and product events still use zero-shot NLI plus
  lexicons. The Groq teacher (`scripts/run_teacher.py silver`) adds silver labels within the free tier's daily quota,
  then `scripts/train_students.py` retrains.
- **Historical timing.** Date-only timestamps force next-day trading in backtests, where news is already priced. The
  live system uses 15-minute GDELT timestamps.
- **Synthetic book.** Exposures, ratings and LGDs are generated. Scenario market moves are calibrated to history;
  credit-factor shocks and revolver drawdowns are expert-set.
- **Next:** hosted deployment, more sources (earnings-call transcripts, central-bank speeches), and intraday prices to
  test the signal at the speed it is produced.

---

Licensed under the [MIT License](LICENSE).
