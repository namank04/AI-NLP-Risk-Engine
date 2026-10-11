"""Credit risk building blocks: rating scale, PDs, Basel IRB, Vasicek stress, IFRS 9.

All functions are vectorised over numpy arrays.

Rating PDs are rounded, approximate long-run one-year corporate default rates
of the kind published in public academic and agency studies; they are
illustrative, not taken from any single proprietary table.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_PD = np.array([0.0001, 0.0002, 0.0006, 0.0018, 0.0070, 0.0300, 0.2500])
INVESTMENT_GRADE = {"AAA", "AA", "A", "BBB"}
PD_FLOOR = 0.0003                      # Basel corporate PD floor (3 bp)


def rating_pd(ratings) -> np.ndarray:
    idx = np.array([RATINGS.index(r) for r in np.atleast_1d(ratings)])
    return RATING_PD[idx]


def implied_rating(pd_: np.ndarray) -> np.ndarray:
    """Map PDs back to the rating whose PD is closest on a log scale."""
    pd_ = np.clip(np.atleast_1d(pd_), 1e-6, 0.999)
    d = np.abs(np.log(pd_)[:, None] - np.log(RATING_PD)[None, :])
    return np.array(RATINGS)[d.argmin(axis=1)]


def irb_correlation(pd_: np.ndarray) -> np.ndarray:
    """Basel II/III asset correlation for corporate exposures."""
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def maturity_adjustment(pd_: np.ndarray, m_years: np.ndarray) -> np.ndarray:
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    m = np.clip(m_years, 1.0, 5.0)
    return (1 + (m - 2.5) * b) / (1 - 1.5 * b)


def vasicek_conditional_pd(pd_: np.ndarray, rho: np.ndarray, z: np.ndarray | float) -> np.ndarray:
    """PD conditional on the systematic factor being z standard deviations adverse.

    z = norm.ppf(0.999) is the Basel capital quantile; z = 0 returns roughly the
    unconditional PD (slightly below it, as it is the median scenario).
    """
    pd_ = np.clip(pd_, 1e-6, 0.999)
    return norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * z) / np.sqrt(1 - rho))


def irb_capital_k(pd_: np.ndarray, lgd: np.ndarray, m_years: np.ndarray) -> np.ndarray:
    """Basel IRB capital requirement K per unit of EAD (unexpected loss at 99.9%)."""
    pd_ = np.maximum(pd_, PD_FLOOR)
    rho = irb_correlation(pd_)
    cond = vasicek_conditional_pd(pd_, rho, norm.ppf(0.999))
    return np.maximum(lgd * (cond - pd_) * maturity_adjustment(pd_, m_years), 0.0)


def irb_rwa(pd_: np.ndarray, lgd: np.ndarray, ead: np.ndarray, m_years: np.ndarray) -> np.ndarray:
    return 12.5 * irb_capital_k(pd_, lgd, m_years) * ead


def lifetime_pd(pd_1y: np.ndarray, m_years: np.ndarray) -> np.ndarray:
    """Cumulative PD over remaining life assuming a constant annual hazard."""
    return 1 - (1 - pd_1y) ** np.maximum(m_years, 1.0)


def ifrs9_stage(orig_rating: np.ndarray, cur_rating: np.ndarray) -> np.ndarray:
    """Stage 1 (performing), 2 (significant increase in credit risk), 3 (credit-impaired).

    SICR proxy: downgrade of 2+ rating grades since origination, or now CCC.
    Stage 3 is not reachable from a scenario PD alone (it needs an actual default).
    """
    o = np.array([RATINGS.index(r) for r in orig_rating])
    c = np.array([RATINGS.index(r) for r in cur_rating])
    return np.where((c - o >= 2) | (c == RATINGS.index("CCC")), 2, 1)


def ifrs9_ecl(pd_1y: np.ndarray, lgd: np.ndarray, ead: np.ndarray, m_years: np.ndarray,
              stage: np.ndarray, pd_ttc: np.ndarray | None = None) -> np.ndarray:
    """12-month ECL for stage 1, lifetime ECL for stage 2.

    Lifetime PD applies the (stressed) PD in year 1 and reverts to the
    through-the-cycle PD for the remaining years, the usual stress-test
    convention: a one-year scenario does not stay stressed for a loan's whole life.
    """
    ttc = pd_1y if pd_ttc is None else pd_ttc
    remaining = np.maximum(m_years, 1.0) - 1.0
    life = 1 - (1 - pd_1y) * (1 - ttc) ** remaining
    horizon_pd = np.where(stage >= 2, life, pd_1y)
    return horizon_pd * lgd * ead
