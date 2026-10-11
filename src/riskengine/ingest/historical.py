"""Historical datasets (Hugging Face mirrors) converted to RawItem replay files."""

from __future__ import annotations

import io
from collections.abc import Iterator
from datetime import UTC

import pandas as pd

from riskengine.config import RAW_DIR, universe
from riskengine.schema import RawItem, stable_id

HF = "https://huggingface.co/datasets"
FNSPID_HEADLINES_URL = f"{HF}/Zihan1004/FNSPID/resolve/main/Stock_news/All_external.csv"
STOCK_TWEETS_URL = f"{HF}/mjw/stock_market_tweets/resolve/main/stock_market_tweets.csv"

# Historical tickers that map onto today's universe members.
TICKER_HISTORY = {"GOOG": "GOOGL"}


def canonical_ticker(symbol: str) -> str | None:
    s = str(symbol).strip().upper()
    s = TICKER_HISTORY.get(s, s)
    return s if s in set(universe().tickers) else None


def _stream_csv(url: str, chunksize: int = 200_000, **kw) -> Iterator[pd.DataFrame]:
    """Read a remote CSV in chunks without saving it."""
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as resp:
        resp.raise_for_status()
        raw = _IterStream(resp.iter_bytes(chunk_size=1 << 20))
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        yield from pd.read_csv(text, chunksize=chunksize, **kw)


class _IterStream(io.RawIOBase):
    def __init__(self, it):
        self._it = it
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


FNSPID_ARTICLES_URL = f"{HF}/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"


def download_fnspid_articles(progress: bool = True, max_chars: int = 4000, grid: int = 800,
                             workers: int = 12, tickers: list[str] | None = None,
                             out_name: str = "fnspid_articles_universe") -> pd.DataFrame:
    """Full FNSPID articles (23 GB, ~1999-2023) for the universe only, via HTTP range requests.

    The CSV is sorted by ticker, so a coarse grid of byte offsets (each resolved to
    the ticker of the next record) brackets every wanted ticker; only those byte
    spans are downloaded (~5-10% of the file) and parsed.
    """
    import csv as _csv
    import re
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    out = RAW_DIR / f"{out_name}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    url = FNSPID_ARTICLES_URL
    client = httpx.Client(follow_redirects=True, timeout=120)
    size = int(client.get(url, headers={"Range": "bytes=0-0"}).headers["content-range"].split("/")[1])
    # a record starts with "<index or empty>,<YYYY-MM-DD HH:MM:SS[ UTC]>," on a new line
    row_start = re.compile(rb"\n[0-9.]*,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?: UTC)?,")
    ticker_ok = re.compile(r"^[A-Z][A-Z0-9.\-]{0,6}$")
    end_sym = "￿"   # sorts after every ticker

    def record_at(off: int) -> tuple[int, str] | None:
        span = 1 << 16
        while span <= 8 << 20:                       # never pull more than 8 MB for one probe
            b = client.get(url, headers={"Range": f"bytes={off}-{min(size, off + span) - 1}"}).content
            for m in row_start.finditer(b):
                try:
                    rec = next(_csv.reader(io.StringIO(b[m.start() + 1:m.start() + 20000].decode("utf-8", "replace"))))
                except StopIteration:
                    continue
                if len(rec) > 3 and ticker_ok.match(rec[3].upper()):
                    return off + m.start() + 1, rec[3].upper()
            if off + span >= size:
                return None
            span *= 4
        return None

    with ThreadPoolExecutor(workers) as pool:
        found = [p for p in pool.map(record_at, [int(size * k / grid) for k in range(1, grid)]) if p]
    # keep only probes consistent with the file's ticker sort order (guards against misparsed rows)
    points: list[tuple[int, str]] = [(0, "")]
    for off, sym in sorted(set(found)):
        if sym >= points[-1][1]:
            points.append((off, sym))
    points.append((size, end_sym))
    if progress:
        print(f"  mapped {len(points)} offsets")

    wanted = sorted(set(tickers) if tickers else set(universe().tickers) | set(TICKER_HISTORY))
    spans = []
    for t in wanted:
        lo = max(i for i, (_, sym) in enumerate(points) if sym < t)
        hi = min(i for i, (_, sym) in enumerate(points) if sym > t)
        spans.append([points[lo][0], points[hi][0]])
    spans.sort()
    merged = [spans[0]]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    total = sum(b - a for a, b in merged)
    if progress:
        print(f"  downloading {len(merged)} spans, {total / 1e9:.2f} GB of {size / 1e9:.1f} GB")

    # Spool every range straight to disk (one sparse file per span) so memory stays
    # flat; a .done file of finished pieces makes the download resumable.
    import threading

    spool = RAW_DIR / f"{out_name}_spans"
    spool.mkdir(parents=True, exist_ok=True)
    done_file = spool / "pieces.done"
    finished = set(done_file.read_text().split()) if done_file.exists() else set()
    piece = 8 << 20
    jobs = [(i, a, a + k, min(b, a + k + piece)) for i, (a, b) in enumerate(merged) for k in range(0, b - a, piece)]
    todo = [j for j in jobs if f"{j[2]}" not in finished]
    lock = threading.Lock()
    progress_bytes = [total - sum(j[3] - j[2] for j in todo)]
    for i, (a, b) in enumerate(merged):
        f = spool / f"span_{i}.csv"
        if not f.exists():
            with open(f, "wb") as fh:
                fh.truncate(b - a)

    def fetch(job):
        i, a, lo, hi = job
        for _ in range(6):
            try:
                data = client.get(url, headers={"Range": f"bytes={lo}-{hi - 1}"}).content
                if len(data) == hi - lo:
                    break
            except httpx.HTTPError:
                pass
        else:
            raise RuntimeError(f"range {lo}-{hi} failed")
        with lock:
            with open(spool / f"span_{i}.csv", "r+b") as fh:
                fh.seek(lo - a)
                fh.write(data)
            with open(done_file, "a") as fh:
                fh.write(f"{lo}\n")
            progress_bytes[0] += len(data)
            if progress:
                print(f"\r  {progress_bytes[0] / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch, todo))
    if progress:
        print()
    cols = ["idx", "Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Author", "Article",
            "Lsa_summary", "Luhn_summary", "Textrank_summary", "Lexrank_summary"]
    frames = []
    for i in range(len(merged)):
        for chunk in pd.read_csv(spool / f"span_{i}.csv", header=None, names=cols, dtype=str, on_bad_lines="skip",
                                 usecols=["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"],
                                 chunksize=20_000, engine="c"):
            hit = chunk[chunk["Stock_symbol"].str.upper().isin(wanted)].copy()
            if len(hit):
                hit["Article"] = hit["Article"].str.slice(0, max_chars)
                frames.append(hit)
        if progress:
            print(f"  parsed span {i + 1}/{len(merged)}: {sum(len(f) for f in frames):,} rows so far")
    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["Stock_symbol"].map(canonical_ticker) if not tickers else df["Stock_symbol"].str.upper()
    df["published_at"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["ticker", "published_at", "Article_title"]).drop_duplicates(["Url", "ticker"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def _parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Unix seconds (original Kaggle export) or ISO date strings (some mirrors)."""
    numeric = pd.to_numeric(s, errors="coerce")
    from_epoch = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    from_text = pd.to_datetime(s.where(numeric.isna()), utc=True, errors="coerce")
    return from_epoch.fillna(from_text)


def download_fnspid_headlines(progress: bool = True) -> pd.DataFrame:
    """Filter FNSPID headlines to the universe; cache as parquet."""
    out = RAW_DIR / "fnspid_headlines_universe.parquet"
    if out.exists():
        return pd.read_parquet(out)
    wanted = set(universe().tickers) | set(TICKER_HISTORY)
    keep: list[pd.DataFrame] = []
    rows = 0
    cols = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]
    for chunk in _stream_csv(FNSPID_HEADLINES_URL, usecols=cols, dtype=str):
        rows += len(chunk)
        hit = chunk[chunk["Stock_symbol"].str.upper().isin(wanted)]
        if len(hit):
            keep.append(hit)
        if progress:
            print(f"\r  scanned {rows:,} rows, kept {sum(len(k) for k in keep):,}", end="", flush=True)
    if progress:
        print()
    df = pd.concat(keep, ignore_index=True)
    df["ticker"] = df["Stock_symbol"].map(canonical_ticker)
    df["published_at"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["ticker", "published_at", "Article_title"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def download_stock_tweets(progress: bool = True) -> pd.DataFrame:
    out = RAW_DIR / "stock_tweets_universe.parquet"
    if out.exists():
        return pd.read_parquet(out)
    keep = []
    rows = 0
    for chunk in _stream_csv(STOCK_TWEETS_URL, dtype={"tweet_id": str, "writer": str, "body": str,
                                                      "ticker_symbol": str}):
        rows += len(chunk)
        chunk["ticker"] = chunk["ticker_symbol"].map(canonical_ticker)
        keep.append(chunk.dropna(subset=["ticker", "body"]))
        if progress:
            print(f"\r  scanned {rows:,} rows", end="", flush=True)
    if progress:
        print()
    df = pd.concat(keep, ignore_index=True)
    df["published_at"] = _parse_mixed_dates(df["post_date"])
    df = df.dropna(subset=["published_at"]).drop(columns=["Unnamed: 0"], errors="ignore")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def fnspid_items(df: pd.DataFrame) -> Iterator[RawItem]:
    for r in df.itertuples(index=False):
        yield RawItem(
            item_id=stable_id("fnspid", r.Url, r.ticker),
            source="fnspid",
            source_type="news",
            title=str(r.Article_title),
            url=r.Url if isinstance(r.Url, str) else None,
            publisher=r.Publisher if isinstance(r.Publisher, str) else None,
            published_at=r.published_at.to_pydatetime().astimezone(UTC),
            tickers_hint=[r.ticker],
        )


def tweet_items(df: pd.DataFrame) -> Iterator[RawItem]:
    for r in df.itertuples(index=False):
        yield RawItem(
            item_id=stable_id("tweet", r.tweet_id),
            source="tweets_hist",
            source_type="social",
            title="",
            body=str(r.body),
            author=r.writer if isinstance(r.writer, str) else None,
            published_at=r.published_at.to_pydatetime().astimezone(UTC),
            tickers_hint=[r.ticker],
            engagement={"comments": float(r.comment_num or 0), "retweets": float(r.retweet_num or 0),
                        "likes": float(r.like_num or 0)},
        )
