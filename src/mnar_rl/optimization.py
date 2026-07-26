"""Robust policy evaluation and baseline-relative optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .mdp import TabularMDP, optimal_policy, policy_occupancy, policy_value, validate_policy


Array = np.ndarray


@dataclass(frozen=True)
class RobustImprovementResult:
    policy: Array
    occupancy: Array
    certificate: float
    solver_status: str


def sharp_improvement_lower_bound(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Exact worst-case value difference over rectangular reward intervals."""
    d = np.asarray(candidate_occupancy, dtype=float)
    db = np.asarray(baseline_occupancy, dtype=float)
    lo = np.asarray(reward_lower, dtype=float)
    hi = np.asarray(reward_upper, dtype=float)
    if not (d.shape == db.shape == lo.shape == hi.shape):
        raise ValueError("all arrays must have identical shapes")
    if np.any(lo > hi + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")
    contrast = d - db
    return float(np.sum(np.where(contrast >= 0.0, contrast * lo, contrast * hi)))


def separate_value_lower_bound(
    mdp: TabularMDP,
    candidate_policy: Array,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Conservative difference obtained from two separate value bounds."""
    return policy_value(mdp, reward_lower, candidate_policy) - policy_value(mdp, reward_upper, baseline_policy)


def robust_absolute_policy(mdp: TabularMDP, reward_lower: Array) -> tuple[Array, float]:
    """Maximize worst-case absolute value for rectangular reward intervals."""
    return optimal_policy(mdp, reward_lower)


def optimize_robust_improvement(
    mdp: TabularMDP,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> RobustImprovementResult:
    """Solve max_pi min_r V_r(pi)-V_r(pi_b) as a linear program.

    The LP uses occupancy variables d and hypograph variables t satisfying
    t_x <= (d_x-d_b,x) * lower_x and
    t_x <= (d_x-d_b,x) * upper_x.
    """
    baseline = validate_policy(mdp, baseline_policy)
    lo = np.asarray(reward_lower, dtype=float)
    hi = np.asarray(reward_upper, dtype=float)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if lo.shape != shape or hi.shape != shape:
        raise ValueError(f"reward arrays must have shape {shape}")
    if np.any(lo > hi + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")

    db = policy_occupancy(mdp, baseline)
    n_cells = int(np.prod(shape))
    n_vars = 2 * n_cells
    d_offset = 0
    t_offset = n_cells

    objective = np.zeros(n_vars)
    objective[t_offset:] = -1.0

    # Flow equalities: one per (h,s).
    rows: list[Array] = []
    rhs: list[float] = []
    for h in range(mdp.horizon):
        for s in range(mdp.n_states):
            row = np.zeros(n_vars)
            for a in range(mdp.n_actions):
                idx = np.ravel_multi_index((h, s, a), shape)
                row[d_offset + idx] = 1.0
            if h == 0:
                target = float(mdp.initial[s])
            else:
                target = 0.0
                for sp in range(mdp.n_states):
                    for ap in range(mdp.n_actions):
                        prev_idx = np.ravel_multi_index((h - 1, sp, ap), shape)
                        row[d_offset + prev_idx] -= mdp.transition[h - 1, sp, ap, s]
            rows.append(row)
            rhs.append(target)

    # Hypograph inequalities.
    a_ub: list[Array] = []
    b_ub: list[float] = []
    lo_flat = lo.ravel()
    hi_flat = hi.ravel()
    db_flat = db.ravel()
    for i in range(n_cells):
        row = np.zeros(n_vars)
        row[t_offset + i] = 1.0
        row[d_offset + i] = -lo_flat[i]
        a_ub.append(row)
        b_ub.append(-lo_flat[i] * db_flat[i])

        row = np.zeros(n_vars)
        row[t_offset + i] = 1.0
        row[d_offset + i] = -hi_flat[i]
        a_ub.append(row)
        b_ub.append(-hi_flat[i] * db_flat[i])

    bounds = [(0.0, None)] * n_cells + [(None, None)] * n_cells
    result = linprog(
        objective,
        A_ub=np.vstack(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.vstack(rows),
        b_eq=np.asarray(rhs),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"robust-improvement LP failed: {result.message}")

    occupancy = result.x[:n_cells].reshape(shape)
    policy = np.empty(shape)
    for h in range(mdp.horizon):
        for s in range(mdp.n_states):
            mass = float(occupancy[h, s].sum())
            if mass > 1e-10:
                policy[h, s] = occupancy[h, s] / mass
            else:
                policy[h, s] = baseline[h, s]

    certificate = sharp_improvement_lower_bound(occupancy, db, lo, hi)
    return RobustImprovementResult(policy, occupancy, certificate, result.message)
