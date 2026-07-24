"""Distribution fitting for the M4 response variables (pure: arrays in, fit summaries out).

Candidates: **normal** (baseline), **lognormal**, **gamma**, **weibull** — the last three fit with
loc fixed at 0 (positive support, more stable at low n; weibull's shape also captures the *left*-skew
of latency). All four then have exactly 2 free parameters, so model selection is a fair, clean
comparison.

Ranked by **AICc** (small-sample-corrected AIC) — the right criterion here since groups are only
n≈9-17. The KS statistic is reported as a descriptive distance only: its p-value is optimistic when
the parameters were fit to the same data (Lilliefors problem), so we don't lean on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import dataset

# name -> (scipy distribution, positive-support?  → fit with floc=0)
CANDIDATES = {
    "normal":    (stats.norm,        False),
    "lognormal": (stats.lognorm,     True),
    "gamma":     (stats.gamma,       True),
    "weibull":   (stats.weibull_min, True),
}


@dataclass
class Fit:
    name: str
    params: tuple
    loglik: float
    k: int            # free parameters (always 2 here)
    aic: float
    aicc: float       # small-sample-corrected AIC (ranking criterion)
    ks: float         # KS statistic vs the fitted CDF (descriptive distance)
    n: int

    def pdf(self, x):
        return CANDIDATES[self.name][0].pdf(x, *self.params)

    def summary(self) -> str:
        return f"{self.name} (AICc {self.aicc:.1f}, KS {self.ks:.2f})"


def _fit_one(name: str, data: np.ndarray) -> Fit:
    dist, positive = CANDIDATES[name]
    if positive:
        data = data[data > 0]                       # positive-support dists need x>0
        params = dist.fit(data, floc=0)             # fix loc=0 → 2 free params, stable at low n
    else:
        params = dist.fit(data)
    n = data.size
    k = 2
    ll = float(np.sum(dist.logpdf(data, *params)))
    aic = 2 * k - 2 * ll
    aicc = aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
    ks = float(stats.kstest(data, dist.cdf, args=params).statistic)
    return Fit(name, params, ll, k, aic, aicc, ks, n)


def fit_all(data, min_n: int = 5) -> List[Fit]:
    """Fit every candidate to ``data``; return them ranked best-AICc-first (empty if too few points)."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < min_n:
        return []
    fits = []
    for name in CANDIDATES:
        try:
            fits.append(_fit_one(name, data))
        except Exception:
            pass                                    # a candidate that won't converge is just dropped
    fits.sort(key=lambda f: f.aicc)
    return fits


def best_fit(data, min_n: int = 5) -> Optional[Fit]:
    fits = fit_all(data, min_n)
    return fits[0] if fits else None


def fit_table(df: pd.DataFrame, var_label: str, pool: bool = False, min_n: int = 5) -> pd.DataFrame:
    """AICc comparison per group for one response variable. One row per (species, condition) — or per
    species if ``pool`` combines conditions (borrows strength for the small groups). Columns: n, the
    winning distribution, ΔAICc of normal vs the winner (>0 ⇒ normal is worse), and each candidate's AICc."""
    col, _ = dataset.RESPONSE_VARS[var_label]
    rows = []
    for sp in dataset.SPECIES:
        conds = [None] if pool else list(dataset.CONDITIONS)
        for cond in conds:
            m = (df["species"] == sp) if cond is None else (
                (df["species"] == sp) & (df["condition"] == cond))
            vals = df.loc[m, col].dropna().to_numpy(float)
            fits = fit_all(vals, min_n=min_n)
            row = {"species": sp, "condition": "pooled" if cond is None else cond, "n": vals.size}
            if fits:
                by = {f.name: f.aicc for f in fits}
                row["best"] = fits[0].name
                row["normal_minus_best"] = round(by.get("normal", float("nan")) - fits[0].aicc, 2)
                for name in CANDIDATES:
                    row[name] = round(by[name], 2) if name in by else None
            else:
                row["best"] = None
                row["normal_minus_best"] = None
                for name in CANDIDATES:
                    row[name] = None
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = dataset.load_dataframe()
    for label in dataset.RESPONSE_VARS:
        print(f"\n=== {label} — per-group best fit (AICc) ===")
        print(fit_table(df, label).to_string(index=False))
