"""Sharp sensitivity bounds for rewards missing not at random.

The binary model conditions on a context x=(h,s,a).  We observe
q=P(M=1|x) and p=P(R=1|M=1,x), while
u=P(R=1|M=0,x) is unobserved.  The sensitivity model bounds
odds(M=1|R=1,x) / odds(M=1|R=0,x) in [1/Gamma, Gamma].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta


Array = np.ndarray


def _validate_probability(name: str, value: Array | float) -> Array:
    arr = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return arr


def binary_missing_success_bounds(p_obs: Array | float, gamma: float) -> tuple[Array, Array]:
    """Return sharp bounds on P(R=1 | M=0, x).

    Parameters
    ----------
    p_obs:
        P(R=1 | M=1, x).
    gamma:
        Odds-ratio sensitivity parameter, at least one.
    """
    p = _validate_probability("p_obs", p_obs)
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")

    lower = p / (p + gamma * (1.0 - p))
    upper = gamma * p / (1.0 - p + gamma * p)
    return lower, upper


def binary_reward_mean_bounds(
    q_obs: Array | float,
    p_obs: Array | float,
    gamma: float,
) -> tuple[Array, Array]:
    """Return sharp bounds on E[R | x] for binary rewards."""
    q = _validate_probability("q_obs", q_obs)
    p = _validate_probability("p_obs", p_obs)
    q, p = np.broadcast_arrays(q, p)
    lower_u, upper_u = binary_missing_success_bounds(p, gamma)
    lower = q * p + (1.0 - q) * lower_u
    upper = q * p + (1.0 - q) * upper_u
    return lower, upper


def odds_ratio_from_observed_missing(p_obs: float, p_miss: float) -> float:
    """Compute odds(M=1|R=1)/odds(M=1|R=0) from p_obs and p_miss.

    Boundary cases use the natural extended-real convention.
    """
    p = float(_validate_probability("p_obs", p_obs))
    u = float(_validate_probability("p_miss", p_miss))
    numerator = p * (1.0 - u)
    denominator = (1.0 - p) * u
    if denominator == 0.0:
        return np.inf if numerator > 0.0 else 1.0
    return numerator / denominator


def clopper_pearson_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact binomial interval with noncoverage probability alpha."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if trials == 0:
        return 0.0, 1.0
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return lower, upper


@dataclass(frozen=True)
class BinaryConfidenceBounds:
    """Simultaneous outer reward intervals and nuisance confidence intervals."""

    reward_lower: Array
    reward_upper: Array
    q_lower: Array
    q_upper: Array
    p_lower: Array
    p_upper: Array


def binary_reward_confidence_bounds(
    total_counts: Array,
    observed_counts: Array,
    observed_successes: Array,
    gamma: float,
    delta: float = 0.05,
) -> BinaryConfidenceBounds:
    """Construct simultaneous outer intervals for all reward cells.

    Clopper--Pearson intervals are Bonferroni-corrected over the q and p
    nuisance parameters in every cell.  The mapping to reward bounds uses
    monotonicity: both outer endpoints use the lower confidence limit for q;
    the lower (upper) reward endpoint uses the lower (upper) limit for p.
    """
    n = np.asarray(total_counts, dtype=int)
    m = np.asarray(observed_counts, dtype=int)
    y = np.asarray(observed_successes, dtype=int)
    if n.shape != m.shape or n.shape != y.shape:
        raise ValueError("count arrays must have identical shapes")
    if np.any(n < 0) or np.any(m < 0) or np.any(y < 0) or np.any(m > n) or np.any(y > m):
        raise ValueError("invalid nested counts")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")

    cells = max(int(n.size), 1)
    alpha_each = delta / (2.0 * cells)
    q_lo = np.empty(n.shape, dtype=float)
    q_hi = np.empty(n.shape, dtype=float)
    p_lo = np.empty(n.shape, dtype=float)
    p_hi = np.empty(n.shape, dtype=float)

    for idx in np.ndindex(n.shape):
        q_lo[idx], q_hi[idx] = clopper_pearson_interval(int(m[idx]), int(n[idx]), alpha_each)
        p_lo[idx], p_hi[idx] = clopper_pearson_interval(int(y[idx]), int(m[idx]), alpha_each)

    lower, _ = binary_reward_mean_bounds(q_lo, p_lo, gamma)
    _, upper = binary_reward_mean_bounds(q_lo, p_hi, gamma)
    return BinaryConfidenceBounds(lower, upper, q_lo, q_hi, p_lo, p_hi)


def finite_reward_mean_bounds(
    q_obs: float,
    observed_probabilities: Iterable[float],
    reward_values: Iterable[float],
    gamma: float,
) -> tuple[float, float]:
    """Sharp finite-support reward bounds under pairwise observation-odds sensitivity.

    For observed probabilities p_k and missing probabilities u_k, the model is

        1/Gamma <= (p_k/u_k)/(p_l/u_l) <= Gamma

    for all categories k,l with the usual limiting interpretation.  Since p is
    fixed, these restrictions are linear in u, so both endpoints are small LPs.
    """
    q = float(_validate_probability("q_obs", q_obs))
    p = np.asarray(list(observed_probabilities), dtype=float)
    rewards = np.asarray(list(reward_values), dtype=float)
    if p.ndim != 1 or rewards.ndim != 1 or p.shape != rewards.shape or p.size == 0:
        raise ValueError("probabilities and reward values must be nonempty vectors of equal length")
    _validate_probability("observed_probabilities", p)
    if not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("observed probabilities must sum to one")
    if np.any(~np.isfinite(rewards)):
        raise ValueError("reward values must be finite")
    if gamma < 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and at least one")

    a_ub: list[Array] = []
    b_ub: list[float] = []
    k = p.size
    for i in range(k):
        for j in range(i + 1, k):
            # p_i u_j <= Gamma p_j u_i
            row = np.zeros(k)
            row[j] = p[i]
            row[i] = -gamma * p[j]
            a_ub.append(row)
            b_ub.append(0.0)
            # p_j u_i <= Gamma p_i u_j
            row = np.zeros(k)
            row[i] = p[j]
            row[j] = -gamma * p[i]
            a_ub.append(row)
            b_ub.append(0.0)

    constraints = np.vstack(a_ub) if a_ub else None
    rhs = np.asarray(b_ub) if b_ub else None
    common = dict(
        A_ub=constraints,
        b_ub=rhs,
        A_eq=np.ones((1, k)),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * k,
        method="highs",
    )
    min_result = linprog(rewards, **common)
    max_result = linprog(-rewards, **common)
    if not min_result.success or not max_result.success:
        raise RuntimeError("finite-support sensitivity LP failed")

    observed_mean = float(p @ rewards)
    lower = q * observed_mean + (1.0 - q) * float(min_result.fun)
    upper = q * observed_mean - (1.0 - q) * float(max_result.fun)
    return lower, upper
