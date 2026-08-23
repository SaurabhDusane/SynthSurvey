"""Lightweight significance testing for cross-tabulations.

Turns a synthetic dataset into an *insight* tool: pick two questions, get a
cross-tab, a chi-square test of independence, its p-value, and a Cramer's V
effect size — all without a heavy stats dependency. The chi-square p-value is
computed from the regularized incomplete gamma function (Numerical-Recipes
style), so pandas/numpy are the only requirements.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

_MAX_ITER = 1000
_EPS = 1e-12
_FPMIN = 1e-300


def _gammln(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation)."""
    cof = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _p_series(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x) via series expansion."""
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAX_ITER):
        ap += 1
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - _gammln(a))


def _q_continued_fraction(a: float, x: float) -> float:
    """Upper regularized incomplete gamma Q(a, x) via continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - _gammln(a)) * h


def chi2_sf(x: float, k: int) -> float:
    """Survival function (upper tail p-value) of chi-square with k dof."""
    if k <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _p_series(a, xx)
    return _q_continued_fraction(a, xx)


def chi_square_test(observed: list[list[float]]) -> dict:
    """Chi-square test of independence on a 2-D contingency table.

    Returns {chi2, dof, p_value, cramers_v, n}. p_value/cramers_v are None when
    the table is degenerate (a single row or column).
    """
    rows = len(observed)
    cols = len(observed[0]) if rows else 0
    row_tot = [sum(r) for r in observed]
    col_tot = [sum(observed[i][j] for i in range(rows)) for j in range(cols)]
    n = sum(row_tot)

    if rows < 2 or cols < 2 or n == 0:
        return {"chi2": 0.0, "dof": 0, "p_value": None, "cramers_v": None, "n": int(n)}

    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            exp = row_tot[i] * col_tot[j] / n
            if exp > 0:
                chi2 += (observed[i][j] - exp) ** 2 / exp

    dof = (rows - 1) * (cols - 1)
    p = chi2_sf(chi2, dof)
    cramers_v = math.sqrt(chi2 / (n * min(rows - 1, cols - 1)))
    return {
        "chi2": round(chi2, 4),
        "dof": dof,
        "p_value": p,
        "cramers_v": round(cramers_v, 4),
        "n": int(n),
    }


def crosstab_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Build a cross-tab of two columns and run a chi-square test on it."""
    sub = df[[col_a, col_b]].dropna().astype(str)
    if sub.empty:
        return {"table": pd.DataFrame(), "chi2": 0.0, "dof": 0,
                "p_value": None, "cramers_v": None, "n": 0}
    table = pd.crosstab(sub[col_a], sub[col_b])
    result = chi_square_test(table.values.tolist())
    result["table"] = table
    return result


def effect_size_label(v: Optional[float]) -> str:
    """Qualitative label for a Cramer's V effect size."""
    if v is None:
        return "n/a"
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "moderate"
    return "large"
