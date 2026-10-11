# Data audit

## FNSPID articles for the final universe (`data/raw/fnspid_articles_universe.parquet`)

Downloaded by HTTP range requests over the ticker-sorted 23 GB file (only ~6% transferred).

| ticker   |   2016 |   2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |
|:---------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| AAPL     |      0 |      0 |      0 |      0 |      0 |      0 |   3094 |   5771 |
| MSFT     |      0 |      0 |      0 |      0 |      0 |      0 |   2679 |   6058 |
| NVDA     |      0 |      0 |      0 |      0 |      0 |    711 |   2368 |   5637 |
| GOOGL    |      0 |      0 |      0 |   1428 |   1637 |   1472 |   1878 |   2311 |
| DIS      |      0 |      0 |      0 |    851 |   2117 |   1741 |   1922 |   2080 |
| TSLA     |      0 |      0 |      0 |      0 |      0 |      0 |   3169 |   5543 |
| NKE      |    672 |    828 |    673 |    540 |    681 |    730 |    921 |    964 |
| GS       |    746 |    972 |    805 |    563 |   1028 |   1436 |   1429 |   1751 |
| WFC      |   1704 |   1595 |    794 |    563 |    948 |    857 |    850 |   1186 |
| XOM      |      0 |      0 |      0 |    275 |   1303 |   1242 |   1959 |   1978 |
| CVX      |    339 |    933 |    838 |    739 |   1057 |    983 |   1768 |   2031 |
| MRK      |    538 |    866 |    739 |    454 |    610 |    884 |   1078 |    980 |
| ABBV     |    397 |    536 |    589 |    429 |    495 |    551 |    809 |    968 |
| BA       |      0 |      0 |    703 |   1327 |   2015 |    677 |    926 |    417 |
| CAT      |    143 |    368 |    845 |    398 |    278 |    363 |    496 |    918 |
| KO       |    624 |    645 |    588 |    593 |    621 |    556 |    966 |   1295 |
| WMT      |      0 |      0 |    116 |   1340 |   2023 |   1486 |   1883 |   1827 |
| NEE      |    378 |    451 |    462 |    323 |    397 |    317 |    601 |    807 |
| DOW      |      0 |      0 |      0 |    100 |    199 |    164 |    312 |    325 |
| AMT      |    632 |    885 |    873 |    338 |    133 |    189 |    313 |    407 |

## Coverage gaps that shaped the universe

FNSPID contains **no** articles for JPM, BAC, JNJ, UNH, PG or META (verified by binary-searching the
sorted file: e.g. `INTL` is followed directly by `K`), and none for AMZN before 2023. They were replaced
by same-sector S&P 100 names with dense 2022-2023 coverage: WFC (Financials), MRK and ABBV (Health Care),
WMT (Consumer Staples), DIS (Communication Services), NKE (Consumer Discretionary), DOW (Materials) and
AMT (Real Estate, completing all 11 GICS sectors).

**Backtest / impact-training window: 2022-01-01 to 2023-12-15**, where every ticker has coverage.

## Other corpora

- FNSPID headline file (`All_external.csv`, 2009-2020): patchy for this universe; used only for the COVID-2020 replay.
- Stock tweets (2015-2019): 1.7M tweets but only AAPL, MSFT, GOOGL, TSLA (and AMZN); date-only timestamps.
- Timestamp precision: <1% of historical items carry an intraday time, so backtests trade on the next day's open.
