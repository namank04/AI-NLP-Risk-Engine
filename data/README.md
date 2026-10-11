# Data

Everything the prototype runs on is either **publicly available** or **synthetic**. No proprietary or
client data is used, and nothing from S&P Global or Crisil.

Large raw downloads live in `data/raw/` (gitignored, ~2 GB) and are rebuilt by the scripts below. The
files committed here are small derived samples that let the dashboard and tests run without them.

## Sources

| Source | What it provides | Access | Licence / terms |
|---|---|---|---|
| [GDELT 2.0 GKG](https://www.gdeltproject.org/) | Global online news every 15 minutes: page title, URL, tone, themes | raw `*.gkg.csv.zip` files, no key | open, free for any use with attribution |
| [SEC EDGAR](https://www.sec.gov/edgar) | 8-K filings with item codes and press-release exhibits | `data.sec.gov` JSON, no key (User-Agent required) | US public domain |
| [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) | Ticker-tagged financial news articles and headlines, 2009-2023 | Hugging Face, HTTP range requests | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); noncommercial use and attribution required |
| [Stock market tweets](https://huggingface.co/datasets/mjw/stock_market_tweets) | 1.7M tweets on large-cap tickers, 2015-2019 | Hugging Face | Apache-2.0 |
| [Yahoo Finance via yfinance](https://pypi.org/project/yfinance/) | Daily adjusted prices, shares outstanding, FX | public endpoints | personal / research use |
| [FRED](https://fred.stlouisfed.org/) | Treasury yields, Moody's Baa/Aaa, ICE BofA OAS (recent years only), VIX, Brent | `fredgraph.csv`, no key | public |

Pretrained models (downloaded on first run, not committed): `ProsusAI/finbert`,
`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`.

### FNSPID attribution and reuse

FNSPID was created by **Zihan Dong, Xinyu Fan, and Zhiyuan Peng**, *FNSPID: A
Comprehensive Financial News Dataset in Time Series* (2024),
[arXiv:2402.06698](https://arxiv.org/abs/2402.06698). The
[dataset card](https://huggingface.co/datasets/Zihan1004/FNSPID) publishes it under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The FNSPID-derived samples in this repository were filtered by ticker and date,
deduplicated, combined with other public sources for crisis replays, and transformed
into model signals for research and demonstration. This attribution does not imply
endorsement by the dataset authors. The repository's MIT license covers original
project code; it does **not** replace the FNSPID dataset's license or any rights in
individual news articles, whose original publishers are identified where available.
Do not treat these samples as licensed for commercial reuse.

## Committed files

| Path | Contents | Built by |
|---|---|---|
| `replay/svb_2023.parquet` | 14,483 items, 6-24 Mar 2023: GDELT news (hourly sample) + FNSPID articles | `scripts/build_replays.py svb_2023` |
| `replay/ukraine_2022.parquet` | 12,206 items, 17 Feb - 10 Mar 2022 | `scripts/build_replays.py ukraine_2022` |
| `replay/covid_2020.parquet` | 19 Feb - 27 Mar 2020 | `scripts/build_replays.py covid_2020` |
| `replay/recent.parquet` | 2 days of live GDELT news + SEC 8-Ks around the build date | `scripts/build_replays.py recent` |
| `replay/smoke_covid_headlines.parquet` | 1,995 FNSPID headlines, Feb-Apr 2020 (fast smoke test) | ad hoc |
| `processed/signals_fnspid_2022_2023.parquet` | 110,737 engine signals over 72,451 FNSPID articles (2022-2023) | `scripts/run_engine_batch.py` |
| `processed/impact_eval.json` | Event-study validation of the impact score | `scripts/train_impact_model.py` |
| `processed/backtest.json` | Module A walk-forward backtest | `scripts/run_backtest.py` |
| `processed/coverage_by_year.csv` | Articles per ticker per year | `scripts/data_audit.py` |
| `synthetic/portfolio.csv` | 408-position wholesale banking book | `riskengine.modules.stress.portfolio` (seed 7) |
| `synthetic/counterparties.csv` | 80 counterparties: the 20 universe companies + 60 fictitious | same |
| `gold/pool.jsonl` | 875 items selected for hand labeling, with baseline suggestions | `scripts/build_gold_pool.py`, `scripts/boost_gold_pool.py` |
| `gold/labels.jsonl` | Hand labels (appears once labeling starts) | dashboard, Evaluation → Labeling |

## Assumptions

- **Timestamps.** Historical news is mostly date-stamped without a time, so backtests trade the day after
  publication and event studies use a two-day reaction window [0, +1].
- **Benchmark weights** are price × *current* shares outstanding (split-adjusted), an approximation of
  historical market caps.
- **Synthetic book.** Exposures, internal ratings and LGDs are generated; rating PDs are rounded long-run
  averages from public literature. The 20 listed companies appear only as hypothetical counterparties.
- **Universe.** FNSPID has no articles for JPM, BAC, JNJ, UNH, PG or META and none for AMZN before 2023, so
  those were replaced by same-sector names with coverage ([docs/data_audit.md](../docs/data_audit.md)).
- **GDELT sampling.** Crisis replays keep one GKG file per hour to bound download size; live use reads every file.

## Rebuilding everything

```bash
python scripts/data_audit.py            # FNSPID headlines + tweets, coverage report
python -c "from riskengine.ingest.historical import download_fnspid_articles; download_fnspid_articles()"
python scripts/build_replays.py recent svb_2023 ukraine_2022 covid_2020 --step 60
python scripts/run_engine_batch.py      # ~1 h on a laptop GPU
python scripts/train_impact_model.py
python scripts/run_backtest.py
python scripts/calibrate_scenarios.py
```
