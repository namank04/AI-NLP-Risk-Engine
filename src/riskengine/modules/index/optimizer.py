"""Black-Litterman construction for the sentiment-tilted index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    r = returns.dropna(how="all").fillna(0.0)
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)


@dataclass
class BLConfig:
    delta: float = 2.5            # risk aversion
    tau: float = 0.05
    ic: float = 0.03              # information coefficient of the sentiment signal (Grinold sizing)
    conf_c0: float = 2.0          # evidence mass at which confidence reaches 50%
    max_weight: float = 0.20
    min_weight: float = 0.0
    te_budget: float = 0.02       # ex-ante annual tracking-error budget (enhanced-index convention)


def black_litterman(sigma: pd.DataFrame, w_mkt: pd.Series, x: pd.Series, evidence: pd.Series,
                    cfg: BLConfig) -> tuple[pd.Series, pd.Series]:
    """Posterior expected excess returns and prior, both indexed like w_mkt."""
    tick = list(w_mkt.index)
    S = sigma.loc[tick, tick].to_numpy()
    w = w_mkt.to_numpy()
    pi = cfg.delta * S @ w
    has_view = (x.reindex(tick).fillna(0).abs() > 1e-6).to_numpy()
    if not has_view.any():
        return pd.Series(pi, index=tick), pd.Series(pi, index=tick)
    idx = np.where(has_view)[0]
    P = np.zeros((len(idx), len(tick)))
    P[np.arange(len(idx)), idx] = 1.0
    vol = np.sqrt(np.diag(S))
    xs = x.reindex(tick).to_numpy()
    z = xs / (xs[idx].std() + 1e-9) if len(idx) > 1 else np.sign(xs)
    Q = pi[idx] + cfg.ic * vol[idx] * z[idx]            # Grinold: alpha = IC x vol x score
    ev = evidence.reindex(tick).fillna(0).to_numpy()[idx]
    conf = np.clip(ev / (ev + cfg.conf_c0), 1e-3, 0.999)
    omega = np.diag(cfg.tau * np.diag(S)[idx] / conf)
    tS_inv = np.linalg.inv(cfg.tau * S)
    om_inv = np.linalg.inv(omega)
    A = tS_inv + P.T @ om_inv @ P
    b = tS_inv @ pi + P.T @ om_inv @ Q
    mu = np.linalg.solve(A, b)
    return pd.Series(mu, index=tick), pd.Series(pi, index=tick)


def optimise(mu: pd.Series, sigma: pd.DataFrame, w_bench: pd.Series, cfg: BLConfig) -> pd.Series:
    """max mu'w - delta/2 w'Sw  s.t. sum w = 1, bounds, ex-ante tracking error <= te_budget."""
    import cvxpy as cp

    tick = list(w_bench.index)
    S = sigma.loc[tick, tick].to_numpy()
    if not (np.isfinite(S).all() and np.isfinite(mu.to_numpy()).all() and np.isfinite(w_bench.to_numpy()).all()):
        return w_bench.copy()           # bad inputs: hold the benchmark rather than trade on noise
    # daily covariances (~1e-4) leave the objective near solver tolerance and the solution
    # arbitrary; annualised percent units give the same optimum with sane conditioning
    scale = 252 * 100
    S = scale * ((S + S.T) / 2) + 1e-8 * np.eye(len(tick))
    m = scale * mu.to_numpy()
    b = w_bench.to_numpy()
    w = cp.Variable(len(tick))
    obj = m @ w - cfg.delta / 2 * cp.quad_form(w, S)
    # a mega-cap can exceed the flat cap in the benchmark itself; never force an underweight
    upper = np.maximum(cfg.max_weight, b + 0.05)
    # S is in annualised-variance x 100 units, so the TE budget scales the same way
    cons = [cp.sum(w) == 1, w >= cfg.min_weight, w <= upper,
            cp.quad_form(w - b, S) <= 100 * cfg.te_budget ** 2]
    prob = cp.Problem(cp.Maximize(obj), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve()
    if w.value is None:
        return w_bench.copy()
    out = pd.Series(np.clip(w.value, 0, None), index=tick)
    return out / out.sum()
