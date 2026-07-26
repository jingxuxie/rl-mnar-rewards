"""Sharp sensitivity and confidence bounds for binary MNAR rewards.

For a context x=(h,s,a), the observed law identifies
q=P(M=1|x) and p=P(R=1|M=1,x).  The missing-case success
probability nu=P(R=1|M=0,x) is only partially identified when
reward recording is missing not at random.
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


def binary_missing_success_bounds(
    p_obs: Array | float,
    gamma: float,
) -> tuple[Array, Array]:
    """Sharp bounds on ``P(R=1 | M=0,x)``.

    The sensitivity restriction is

        1/gamma <= odds(M=1|R=1,x) / odds(M=1|R=0,x) <= gamma.
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
    """Sharp bounds on ``E[R|x]`` for binary rewards."""
    q = _validate_probability("q_obs", q_obs)
    p = _validate_probability("p_obs", p_obs)
    q, p = np.broadcast_arrays(q, p)
    missing_lower, missing_upper = binary_missing_success_bounds(p, gamma)
    lower = q * p + (1.0 - q) * missing_lower
    upper = q * p + (1.0 - q) * missing_upper
    return lower, upper


def odds_ratio_from_observed_missing(p_obs: float, p_miss: float) -> float:
    """Return the reward-recording odds ratio implied by ``p_obs,p_miss``."""
    p = float(_validate_probability("p_obs", p_obs))
    nu = float(_validate_probability("p_miss", p_miss))
    numerator = p * (1.0 - nu)
    denominator = (1.0 - p) * nu
    if denominator == 0.0:
        return np.inf if numerator > 0.0 else 1.0
    return numerator / denominator


def clopper_pearson_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact lower confidence limit with failure probability ``alpha``."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided exact upper confidence limit with failure probability ``alpha``."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if trials == 0 or successes == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


def clopper_pearson_interval(
    successes: int,
    trials: int,
    alpha: float,
) -> tuple[float, float]:
    """Two-sided exact interval with total noncoverage probability ``alpha``."""
    return (
        clopper_pearson_lower(successes, trials, alpha / 2.0),
        clopper_pearson_upper(successes, trials, alpha / 2.0),
    )


@dataclass(frozen=True)
class BinaryConfidenceBounds:
    """Simultaneous outer reward intervals.

    Safety only uses ``q_lower,p_lower,p_upper``.  ``q_upper`` is returned for
    diagnostics but is not needed in the three-tail Bonferroni guarantee.
    """

    reward_lower: Array
    reward_upper: Array
    q_lower: Array
    q_upper: Array
    p_lower: Array
    p_upper: Array
    tail_alpha: float


def binary_reward_confidence_bounds(
    total_counts: Array,
    observed_counts: Array,
    observed_successes: Array,
    gamma: float,
    delta: float = 0.05,
) -> BinaryConfidenceBounds:
    """Simultaneous outer reward intervals for all cells.

    Only three one-sided nuisance statements are required per cell:
    ``q >= q_lower``, ``p >= p_lower``, and ``p <= p_upper``.  Allocating
    ``delta/(3D)`` to each tail is sharper than constructing two unused
    two-sided intervals and remains finite-sample exact conditional on visits.
    """
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
    tail_alpha = delta / (3.0 * cells)
    q_lower = np.empty(n.shape, dtype=float)
    q_upper = np.empty(n.shape, dtype=float)
    p_lower = np.empty(n.shape, dtype=float)
    p_upper = np.empty(n.shape, dtype=float)

    for idx in np.ndindex(n.shape):
        q_lower[idx] = clopper_pearson_lower(int(m[idx]), int(n[idx]), tail_alpha)
        q_upper[idx] = clopper_pearson_upper(int(m[idx]), int(n[idx]), tail_alpha)
        p_lower[idx] = clopper_pearson_lower(int(y[idx]), int(m[idx]), tail_alpha)
        p_upper[idx] = clopper_pearson_upper(int(y[idx]), int(m[idx]), tail_alpha)

    lower, _ = binary_reward_mean_bounds(q_lower, p_lower, gamma)
    _, upper = binary_reward_mean_bounds(q_lower, p_upper, gamma)
    return BinaryConfidenceBounds(
        lower,
        upper,
        q_lower,
        q_upper,
        p_lower,
        p_upper,
        tail_alpha,
    )


def finite_reward_mean_bounds(
    q_obs: float,
    observed_probabilities: Iterable[float],
    reward_values: Iterable[float],
    gamma: float,
) -> tuple[float, float]:
    """Sharp finite-support reward bounds under pairwise odds sensitivity."""
    q = float(_validate_probability("q_obs", q_obs))
    p = np.asarray(list(observed_probabilities), dtype=float)
    rewards = np.asarray(list(reward_values), dtype=float)
    if p.ndim != 1 or rewards.ndim != 1 or p.shape != rewards.shape or p.size == 0:
        raise ValueError("probabilities and reward values must be equal nonempty vectors")
    _validate_probability("observed_probabilities", p)
    if not np.isclose(p.sum(), 1.0, atol=1e-10):
        raise ValueError("observed probabilities must sum to one")
    if np.any(~np.isfinite(rewards)):
        raise ValueError("reward values must be finite")
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")

    a_ub: list[Array] = []
    b_ub: list[float] = []
    categories = p.size
    for i in range(categories):
        for j in range(i + 1, categories):
            row = np.zeros(categories)
            row[j] = p[i]
            row[i] = -gamma * p[j]
            a_ub.append(row)
            b_ub.append(0.0)

            row = np.zeros(categories)
            row[i] = p[j]
            row[j] = -gamma * p[i]
            a_ub.append(row)
            b_ub.append(0.0)

    common = dict(
        A_ub=np.vstack(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.ones((1, categories)),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * categories,
        method="highs",
    )
    minimum = linprog(rewards, **common)
    maximum = linprog(-rewards, **common)
    if not minimum.success or not maximum.success:
        raise RuntimeError("finite-support sensitivity LP failed")

    observed_mean = float(p @ rewards)
    lower = q * observed_mean + (1.0 - q) * float(minimum.fun)
    upper = q * observed_mean - (1.0 - q) * float(maximum.fun)
    return lower, upper


@dataclass(frozen=True)
class ContrastIntervalMinimax:
    """Exact minimax regret for choosing between a candidate and baseline.

    ``candidate_probability`` is the probability of deploying the candidate.
    The input interval is the sharp identified set for candidate-minus-baseline
    value, not two separately constructed value intervals.
    """

    deterministic_regret: float
    randomized_regret: float
    candidate_probability: float


def contrast_interval_minimax_regret(
    improvement_lower: float,
    improvement_upper: float,
) -> ContrastIntervalMinimax:
    """Return the exact minimax decision rule for a sharp improvement interval.

    If ``lower < 0 < upper``, observationally equivalent models disagree on
    which policy is better. A selector deploying the candidate with probability
    ``alpha`` has worst-case simple regret

        max(alpha * (-lower), (1-alpha) * upper).

    If the interval lies on one side of zero, one policy weakly dominates and
    minimax regret is zero.
    """
    lower = float(improvement_lower)
    upper = float(improvement_upper)
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("require finite improvement_lower <= improvement_upper")
    if lower >= 0.0:
        return ContrastIntervalMinimax(0.0, 0.0, 1.0)
    if upper <= 0.0:
        return ContrastIntervalMinimax(0.0, 0.0, 0.0)

    downside = -lower
    upside = upper
    candidate_probability = upside / (downside + upside)
    randomized_regret = downside * upside / (downside + upside)
    deterministic_regret = min(downside, upside)
    return ContrastIntervalMinimax(
        deterministic_regret,
        randomized_regret,
        candidate_probability,
    )


# Backward-compatible name for the special case of a point-identified baseline.
AmbiguousBanditMinimax = ContrastIntervalMinimax


def ambiguous_bandit_minimax_regret(
    candidate_lower: float,
    baseline_reward: float,
    candidate_upper: float,
) -> ContrastIntervalMinimax:
    """Specialize :func:`contrast_interval_minimax_regret` to a point baseline."""
    lower = float(candidate_lower)
    baseline = float(baseline_reward)
    upper = float(candidate_upper)
    if not np.isfinite([lower, baseline, upper]).all() or lower > upper:
        raise ValueError("require finite candidate_lower <= candidate_upper")
    return contrast_interval_minimax_regret(lower - baseline, upper - baseline)
