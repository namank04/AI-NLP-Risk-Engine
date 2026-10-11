"""Walk-forward backtest of Module A on historical engine signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from riskengine.modules.index.optimizer import BLConfig, black_litterman, ledoit_wolf_cov, optimise
from riskengine.modules.index.rebalancer import clip_normalise
from riskengine.modules.index.signal import SentimentBook
from riskengine.schema import EventClassification, Evidence, Impact, Sentiment, Signal


@dataclass
class BacktestConfig:
    start: str = "2022-01-03"
    end: str = "2023-12-15"
    half_life_h: float = 72.0
    kappa: float = 1.2
    band: float = 0.01
    cost_bps: float = 5.0
    max_weight: float = 0.20
    cov_lookback: int = 252
    bl: BLConfig = field(default_factory=BLConfig)


def _to_signal(r) -> Signal:
    return Signal(signal_id=r.signal_id, item_id=r.item_id, cluster_id=r.cluster_id, entity=r.entity,
                  entity_type="company", source=r.source, source_type="news", published_at=r.published_at,
                  sentiment=Sentiment(score=float(r.sentiment), confidence=float(r.sentiment_conf), label="neutral"),
                  event=EventClassification(primary=r.event_primary, labels=[r.event_primary], probs={}),
                  impact=Impact(score=int(r.impact), raw=float(r.impact), confidence=0.5, method="hist"),
                  novelty=float(r.novelty), n_sources=int(r.n_sources), source_weight=float(r.source_weight),
                  evidence=Evidence(headline=""), model_version="hist")


def _metrics(daily: pd.Series, bench: pd.Series | None = None) -> dict:
    ann = 252
    nav = (1 + daily).cumprod()
    out = {"ann_return": float(nav.iloc[-1] ** (ann / len(daily)) - 1), "ann_vol": float(daily.std() * np.sqrt(ann)),
           "sharpe": float(daily.mean() / daily.std() * np.sqrt(ann)) if daily.std() > 0 else 0.0,
           "max_drawdown": float((nav / nav.cummax() - 1).min())}
    if bench is not None:
        act = daily - bench
        te = act.std() * np.sqrt(ann)
        out.update({"active_return": float(act.mean() * ann), "tracking_error": float(te),
                    "information_ratio": float(act.mean() * ann / te) if te > 0 else 0.0,
                    "active_t_stat": float(act.mean() / act.std() * np.sqrt(len(act))) if act.std() > 0 else 0.0})
    return {k: round(v, 4) for k, v in out.items()}


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, shares: pd.Series, market: str = "SPY",
                 cfg: BacktestConfig | None = None, impact_col: str = "impact_rule") -> dict:
    cfg = cfg or BacktestConfig()
    tick = [t for t in shares.index if t in prices.columns]
    px = prices[tick + [market]].sort_index().ffill()     # holidays/gaps: carry last close
    rets = px.pct_change(fill_method=None)
    days = rets.loc[cfg.start:cfg.end].index
    sig = signals[signals["entity"].isin(tick)].copy()
    sig["published_at"] = pd.to_datetime(sig["published_at"], utc=True)
    sig["date"] = sig["published_at"].dt.tz_localize(None).dt.normalize()
    sig["impact"] = sig[impact_col].clip(1, 10)
    sig = sig.sort_values("published_at")
    by_date = {d: g for d, g in sig.groupby("date")}

    book = SentimentBook(half_life=timedelta(hours=cfg.half_life_h))
    w_prev = {k: None for k in ("naive_tilt", "engine_tilt", "black_litterman")}
    rows, ic_rows, weights_log = [], [], []
    sigma = None
    from riskengine.market.event_study import fit_market_model

    ar = fit_market_model(px).ar
    for i, D in enumerate(days[:-1]):
        nxt = days[i + 1]
        # signals dated D-1 or earlier are known at the close of D
        prev_day = days[i - 1] if i > 0 else D - pd.Timedelta(days=1)
        for d in pd.date_range(prev_day if i > 0 else D - pd.Timedelta(days=7), D - pd.Timedelta(days=1)):
            g = by_date.get(d)
            if g is None:
                continue
            for r in g.itertuples(index=False):
                book.update(_to_signal(r))
        cut = pd.Timestamp(D, tz="UTC")
        state = book.snapshot(cut.to_pydatetime(), tick)
        x = pd.Series({t: state[t]["tilt"] for t in tick}).clip(-1, 1)
        ev = pd.Series({t: state[t]["evidence"] for t in tick})
        g_prev = by_date.get(D - pd.Timedelta(days=1))
        naive_x = (g_prev.groupby("entity")["sentiment"].mean().reindex(tick).fillna(0)
                   if g_prev is not None else pd.Series(0.0, index=tick))

        cap = (px.loc[D, tick] * shares[tick]).astype(float)
        b = cap / cap.sum()
        if sigma is None or D.weekday() == 0:
            hist = rets[tick].loc[:D].tail(cfg.cov_lookback)
            sigma = ledoit_wolf_cov(hist)

        targets = {
            "cap_weight": b,
            "equal_weight": pd.Series(1 / len(tick), index=tick),
            "naive_tilt": clip_normalise(b * np.exp(cfg.kappa * naive_x.clip(-1, 1)), 0.0, cfg.max_weight),
            "engine_tilt": clip_normalise(b * np.exp(cfg.kappa * x), 0.0, max(cfg.max_weight, b.max() + 0.05)),
        }
        mu, _ = black_litterman(sigma, b, x, ev, cfg.bl)
        targets["black_litterman"] = optimise(mu, sigma, b, cfg.bl)

        r_next = rets.loc[nxt, tick].fillna(0)
        row = {"date": nxt}
        for name, tgt in targets.items():
            if name in w_prev:
                prev = w_prev[name]
                turn = 0.0 if prev is None else float((tgt - prev).abs().sum() / 2)
                if prev is not None and turn < cfg.band:
                    tgt, turn = prev, 0.0
                w_prev[name] = tgt
            else:
                turn = 0.0
            row[name] = float((tgt * r_next).sum()) - turn * cfg.cost_bps / 1e4
            row[f"{name}_turnover"] = turn
        rows.append(row)
        weights_log.append({"date": str(nxt.date()), **{t: round(float(targets["black_litterman"][t]), 5) for t in tick}})
        a_next = ar.loc[nxt, tick]
        mask = a_next.notna()
        if (x[mask].abs() > 1e-6).sum() >= 5:
            ic_rows.append({"date": nxt, "ic_engine": spearmanr(x[mask], a_next[mask]).statistic,
                            "ic_naive": spearmanr(naive_x[mask], a_next[mask]).statistic
                            if (naive_x[mask].abs() > 1e-6).sum() >= 5 else np.nan})

    res = pd.DataFrame(rows).set_index("date")
    strategies = ["cap_weight", "equal_weight", "naive_tilt", "engine_tilt", "black_litterman"]
    metrics = {s: _metrics(res[s], None if s == "cap_weight" else res["cap_weight"]) for s in strategies}
    for s in ("naive_tilt", "engine_tilt", "black_litterman"):
        metrics[s]["avg_daily_turnover"] = round(float(res[f"{s}_turnover"].mean()), 5)
    ic = pd.DataFrame(ic_rows).set_index("date")

    def ic_stats(col):
        v = ic[col].dropna()
        return {"mean_ic": round(float(v.mean()), 4), "ic_t_stat": round(float(v.mean() / v.std() * np.sqrt(len(v))), 3),
                "hit_rate": round(float((v > 0).mean()), 4), "n_days": int(len(v))}

    nav = (1 + res[strategies]).cumprod()
    return {
        "config": {k: v for k, v in cfg.__dict__.items() if k != "bl"} | {"bl": cfg.bl.__dict__},
        "metrics": metrics,
        "ic": {"engine": ic_stats("ic_engine"), "naive": ic_stats("ic_naive")},
        "nav": [{"date": str(d.date()), **{s: round(float(v), 5) for s, v in r.items()}} for d, r in nav.iterrows()],
        "weights_bl": weights_log,
        "ic_series": [{"date": str(d.date()), "engine": None if pd.isna(r.ic_engine) else round(float(r.ic_engine), 4)}
                      for d, r in ic.iterrows()],
    }
