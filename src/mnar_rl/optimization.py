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


def sharp_improvement_interval(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> tuple[float, float]:
    """Sharp identified interval for ``V(candidate)-V(baseline)``."""
    d = np.asarray(candidate_occupancy, dtype=float)
    baseline = np.asarray(baseline_occupancy, dtype=float)
    lower = np.asarray(reward_lower, dtype=float)
    upper = np.asarray(reward_upper, dtype=float)
    if not (d.shape == baseline.shape == lower.shape == upper.shape):
        raise ValueError("all arrays must have identical shapes")
    if np.any(lower > upper + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")

    contrast = d - baseline
    identified_lower = float(
        np.sum(np.where(contrast >= 0.0, contrast * lower, contrast * upper))
    )
    identified_upper = float(
        np.sum(np.where(contrast >= 0.0, contrast * upper, contrast * lower))
    )
    return identified_lower, identified_upper


def sharp_improvement_lower_bound(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Exact worst-case value difference over rectangular reward intervals."""
    return sharp_improvement_interval(
        candidate_occupancy,
        baseline_occupancy,
        reward_lower,
        reward_upper,
    )[0]


def cancellation_gain(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Exact gain of direct comparison over subtracting separate value bounds."""
    d = np.asarray(candidate_occupancy, dtype=float)
    baseline = np.asarray(baseline_occupancy, dtype=float)
    lower = np.asarray(reward_lower, dtype=float)
    upper = np.asarray(reward_upper, dtype=float)
    if not (d.shape == baseline.shape == lower.shape == upper.shape):
        raise ValueError("all arrays must have identical shapes")
    return float(np.sum(np.minimum(d, baseline) * (upper - lower)))


def separate_value_lower_bound(
    mdp: TabularMDP,
    candidate_policy: Array,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Difference obtained by separately robustifying the two policy values."""
    return policy_value(mdp, reward_lower, candidate_policy) - policy_value(
        mdp,
        reward_upper,
        baseline_policy,
    )


def robust_absolute_policy(
    mdp: TabularMDP,
    reward_lower: Array,
) -> tuple[Array, float]:
    """Maximize worst-case absolute value under rectangular reward intervals."""
    return optimal_policy(mdp, reward_lower)


def optimize_robust_improvement(
    mdp: TabularMDP,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> RobustImprovementResult:
    """Solve ``max_pi min_r V_r(pi)-V_r(pi_b)`` by occupancy-measure LP."""
    baseline_policy = validate_policy(mdp, baseline_policy)
    lower = np.asarray(reward_lower, dtype=float)
    upper = np.asarray(reward_upper, dtype=float)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if lower.shape != shape or upper.shape != shape:
        raise ValueError(f"reward arrays must have shape {shape}")
    if np.any(lower > upper + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")

    baseline_occupancy = policy_occupancy(mdp, baseline_policy)
    cells = int(np.prod(shape))
    variables = 2 * cells
    t_offset = cells

    objective = np.zeros(variables)
    objective[t_offset:] = -1.0

    equality_rows: list[Array] = []
    equality_rhs: list[float] = []
    for h in range(mdp.horizon):
        for state in range(mdp.n_states):
            row = np.zeros(variables)
            for action in range(mdp.n_actions):
                index = np.ravel_multi_index((h, state, action), shape)
                row[index] = 1.0
            if h == 0:
                target = float(mdp.initial[state])
            else:
                target = 0.0
                for previous_state in range(mdp.n_states):
                    for previous_action in range(mdp.n_actions):
                        index = np.ravel_multi_index(
                            (h - 1, previous_state, previous_action),
                            shape,
                        )
                        row[index] -= mdp.transition[
                            h - 1,
                            previous_state,
                            previous_action,
                            state,
                        ]
            equality_rows.append(row)
            equality_rhs.append(target)

    inequality_rows: list[Array] = []
    inequality_rhs: list[float] = []
    lower_flat = lower.ravel()
    upper_flat = upper.ravel()
    baseline_flat = baseline_occupancy.ravel()
    for index in range(cells):
        row = np.zeros(variables)
        row[t_offset + index] = 1.0
        row[index] = -lower_flat[index]
        inequality_rows.append(row)
        inequality_rhs.append(-lower_flat[index] * baseline_flat[index])

        row = np.zeros(variables)
        row[t_offset + index] = 1.0
        row[index] = -upper_flat[index]
        inequality_rows.append(row)
        inequality_rhs.append(-upper_flat[index] * baseline_flat[index])

    result = linprog(
        objective,
        A_ub=np.vstack(inequality_rows),
        b_ub=np.asarray(inequality_rhs),
        A_eq=np.vstack(equality_rows),
        b_eq=np.asarray(equality_rhs),
        bounds=[(0.0, None)] * cells + [(None, None)] * cells,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"robust-improvement LP failed: {result.message}")

    occupancy = result.x[:cells].reshape(shape)
    policy = np.empty(shape)
    for h in range(mdp.horizon):
        for state in range(mdp.n_states):
            mass = float(occupancy[h, state].sum())
            if mass > 1e-10:
                policy[h, state] = occupancy[h, state] / mass
            else:
                policy[h, state] = baseline_policy[h, state]

    certificate = sharp_improvement_lower_bound(
        occupancy,
        baseline_occupancy,
        lower,
        upper,
    )
    return RobustImprovementResult(policy, occupancy, certificate, result.message)
