"""Sharp sensitivity bounds for rewards missing not at random."""

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
    """Sharp bounds on P(R=1 | M=0,x) under odds-ratio sensitivity."""
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
    """Sharp bounds on E[R | x] for binary rewards."""
    q = _validate_probability("q_obs", q_obs)
    p = _validate_probability("p_obs", p_obs)
    q, p = np.broadcast_arrays(q, p)
    lower_u, upper_u = binary_missing_success_bounds(p, gamma)
    lower = q * p + (1.0 - q) * lower_u
    upper = q * p + (1.0 - q) * upper_u
    return lower, upper


def binary_reward_interval_width(
    q_obs: Array | float,
    p_obs: Array | float,
    gamma: float,
) -> Array:
    """Closed-form sharp reward interval width.

    w=(1-q)p(1-p)(Gamma^2-1)/[(1-p+Gamma p)(p+Gamma(1-p))].
    """
    q = _validate_probability("q_obs", q_obs)
    p = _validate_probability("p_obs", p_obs)
    q, p = np.broadcast_arrays(q, p)
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")
    numerator = (1.0 - q) * p * (1.0 - p) * (gamma**2 - 1.0)
    denominator = (1.0 - p + gamma * p) * (p + gamma * (1.0 - p))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def maximum_binary_reward_interval_width(q_obs: Array | float, gamma: float) -> Array:
    """Maximum binary interval width over observed success probability p."""
    q = _validate_probability("q_obs", q_obs)
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")
    return (1.0 - q) * (gamma - 1.0) / (gamma + 1.0)


def odds_ratio_from_observed_missing(p_obs: float, p_miss: float) -> float:
    """Compute odds(M=1|R=1)/odds(M=1|R=0) from p_obs and p_miss."""
    p = float(_validate_probability("p_obs", p_obs))
    u = float(_validate_probability("p_miss", p_miss))
    numerator = p * (1.0 - u)
    denominator = (1.0 - p) * u
    if denominator == 0.0:
        return np.inf if numerator > 0.0 else 1.0
    return numerator / denominator


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided lower confidence limit with tail probability alpha."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """Exact one-sided upper confidence limit with tail probability alpha."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if trials == 0 or successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def clopper_pearson_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Two-sided exact interval with total noncoverage probability alpha."""
    return (
        clopper_pearson_lower(successes, trials, alpha / 2.0),
        clopper_pearson_upper(successes, trials, alpha / 2.0),
    )


@dataclass(frozen=True)
class BinaryConfidenceBounds:
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
    """Simultaneous outer reward intervals using three exact one-sided tails/cell."""
    n = np.asarray(total_counts, dtype=int)
    m = np.asarray(observed_counts, dtype=int)
    y = np.asarray(observed_successes, dtype=int)
    if n.shape != m.shape or n.shape != y.shape:
        raise ValueError("count arrays must have identical shapes")
    if np.any(n < 0) or np.any(m < 0) or np.any(y < 0) or np.any(m > n) or np.any(y > m):
        raise ValueError("invalid nested counts")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")

    cells = max(int(n.size), 1)
    alpha_each = delta / (3.0 * cells)
    q_lo = np.empty(n.shape, dtype=float)
    q_hi = np.empty(n.shape, dtype=float)
    p_lo = np.empty(n.shape, dtype=float)
    p_hi = np.empty(n.shape, dtype=float)

    for idx in np.ndindex(n.shape):
        q_lo[idx] = clopper_pearson_lower(int(m[idx]), int(n[idx]), alpha_each)
        q_hi[idx] = clopper_pearson_upper(int(m[idx]), int(n[idx]), alpha_each)
        p_lo[idx] = clopper_pearson_lower(int(y[idx]), int(m[idx]), alpha_each)
        p_hi[idx] = clopper_pearson_upper(int(y[idx]), int(m[idx]), alpha_each)

    lower, _ = binary_reward_mean_bounds(q_lo, p_lo, gamma)
    _, upper = binary_reward_mean_bounds(q_lo, p_hi, gamma)
    return BinaryConfidenceBounds(lower, upper, q_lo, q_hi, p_lo, p_hi)


def binary_reward_confidence_bounds_twosided(
    total_counts: Array,
    observed_counts: Array,
    observed_successes: Array,
    gamma: float,
    delta: float = 0.05,
) -> BinaryConfidenceBounds:
    """Older two-sided Bonferroni construction, retained as an efficiency baseline."""
    n = np.asarray(total_counts, dtype=int)
    m = np.asarray(observed_counts, dtype=int)
    y = np.asarray(observed_successes, dtype=int)
    if n.shape != m.shape or n.shape != y.shape:
        raise ValueError("count arrays must have identical shapes")
    if np.any(n < 0) or np.any(m < 0) or np.any(y < 0) or np.any(m > n) or np.any(y > m):
        raise ValueError("invalid nested counts")
    cells = max(int(n.size), 1)
    alpha_interval = delta / (2.0 * cells)
    q_lo = np.empty(n.shape, dtype=float)
    q_hi = np.empty(n.shape, dtype=float)
    p_lo = np.empty(n.shape, dtype=float)
    p_hi = np.empty(n.shape, dtype=float)
    for idx in np.ndindex(n.shape):
        q_lo[idx], q_hi[idx] = clopper_pearson_interval(int(m[idx]), int(n[idx]), alpha_interval)
        p_lo[idx], p_hi[idx] = clopper_pearson_interval(int(y[idx]), int(m[idx]), alpha_interval)
    lower, _ = binary_reward_mean_bounds(q_lo, p_lo, gamma)
    _, upper = binary_reward_mean_bounds(q_lo, p_hi, gamma)
    return BinaryConfidenceBounds(lower, upper, q_lo, q_hi, p_lo, p_hi)


def finite_reward_mean_bounds(
    q_obs: float,
    observed_probabilities: Iterable[float],
    reward_values: Iterable[float],
    gamma: float,
) -> tuple[float, float]:
    """Sharp finite-support reward bounds under pairwise observation-odds sensitivity."""
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
            row = np.zeros(k)
            row[j] = p[i]
            row[i] = -gamma * p[j]
            a_ub.append(row)
            b_ub.append(0.0)
            row = np.zeros(k)
            row[i] = p[j]
            row[j] = -gamma * p[i]
            a_ub.append(row)
            b_ub.append(0.0)

    common = dict(
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
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
