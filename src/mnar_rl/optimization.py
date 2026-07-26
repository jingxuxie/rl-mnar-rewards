"""Robust policy evaluation and baseline-relative optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .mdp import (
    TabularMDP,
    optimal_policy,
    policy_from_occupancy,
    policy_occupancy,
    policy_value,
    validate_policy,
)

Array = np.ndarray


@dataclass(frozen=True)
class RobustImprovementResult:
    """Solution of the occupancy-measure robust-improvement program."""

    policy: Array
    occupancy: Array
    certificate: float
    solver_status: str


@dataclass(frozen=True)
class MinimaxRegretResult:
    """Optimal binary deployment randomization for an identified interval."""

    candidate_probability: float
    randomized_regret: float
    deterministic_regret: float


def _validate_contrast_arrays(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> tuple[Array, Array, Array, Array]:
    candidate = np.asarray(candidate_occupancy, dtype=float)
    baseline = np.asarray(baseline_occupancy, dtype=float)
    lower = np.asarray(reward_lower, dtype=float)
    upper = np.asarray(reward_upper, dtype=float)
    if not (candidate.shape == baseline.shape == lower.shape == upper.shape):
        raise ValueError("all arrays must have identical shapes")
    if np.any(lower > upper + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")
    return candidate, baseline, lower, upper


def sharp_improvement_lower_bound(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Exact worst-case candidate-minus-baseline value over a reward rectangle."""
    candidate, baseline, lower, upper = _validate_contrast_arrays(
        candidate_occupancy,
        baseline_occupancy,
        reward_lower,
        reward_upper,
    )
    contrast = candidate - baseline
    return float(np.sum(np.where(contrast >= 0.0, contrast * lower, contrast * upper)))


def sharp_improvement_upper_bound(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Exact best-case candidate-minus-baseline value over a reward rectangle."""
    candidate, baseline, lower, upper = _validate_contrast_arrays(
        candidate_occupancy,
        baseline_occupancy,
        reward_lower,
        reward_upper,
    )
    contrast = candidate - baseline
    return float(np.sum(np.where(contrast >= 0.0, contrast * upper, contrast * lower)))


def contrastive_ambiguity_width(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Width of the sharp candidate-minus-baseline value interval."""
    candidate, baseline, lower, upper = _validate_contrast_arrays(
        candidate_occupancy,
        baseline_occupancy,
        reward_lower,
        reward_upper,
    )
    return float(np.sum(np.abs(candidate - baseline) * (upper - lower)))


def contrastive_missingness_budget(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    q_observed: Array,
    gamma: float,
) -> float:
    """Sharp-envelope upper bound on contrastive MNAR ambiguity."""
    candidate = np.asarray(candidate_occupancy, dtype=float)
    baseline = np.asarray(baseline_occupancy, dtype=float)
    q_observed_array = np.asarray(q_observed, dtype=float)
    if not (candidate.shape == baseline.shape == q_observed_array.shape):
        raise ValueError("all arrays must have identical shapes")
    if gamma < 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and at least one")
    sensitivity_factor = (gamma - 1.0) / (gamma + 1.0)
    return float(
        sensitivity_factor
        * np.sum(np.abs(candidate - baseline) * (1.0 - q_observed_array))
    )


def cancellation_gain(
    candidate_occupancy: Array,
    baseline_occupancy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Gain of direct comparison over subtracting separate robust values."""
    candidate, baseline, lower, upper = _validate_contrast_arrays(
        candidate_occupancy,
        baseline_occupancy,
        reward_lower,
        reward_upper,
    )
    return float(np.sum(np.minimum(candidate, baseline) * (upper - lower)))


def separate_value_lower_bound(
    mdp: TabularMDP,
    candidate_policy: Array,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> float:
    """Lower bound obtained by separately bounding candidate and baseline values."""
    return policy_value(mdp, reward_lower, candidate_policy) - policy_value(
        mdp, reward_upper, baseline_policy
    )


def robust_absolute_policy(
    mdp: TabularMDP,
    reward_lower: Array,
) -> tuple[Array, float]:
    """Maximize worst-case absolute value for a rectangular reward set."""
    return optimal_policy(mdp, reward_lower)


def contrast_interval_minimax_regret(
    lower: float,
    upper: float,
) -> MinimaxRegretResult:
    """Solve the minimax binary deployment problem for a sharp interval."""
    if lower > upper:
        raise ValueError("lower cannot exceed upper")
    if lower >= 0.0:
        return MinimaxRegretResult(1.0, 0.0, 0.0)
    if upper <= 0.0:
        return MinimaxRegretResult(0.0, 0.0, 0.0)
    candidate_probability = upper / (upper - lower)
    randomized_regret = (-lower) * upper / (upper - lower)
    deterministic_regret = min(-lower, upper)
    return MinimaxRegretResult(
        candidate_probability=candidate_probability,
        randomized_regret=randomized_regret,
        deterministic_regret=deterministic_regret,
    )


def ambiguous_bandit_minimax_regret(
    lower: float,
    upper: float,
) -> MinimaxRegretResult:
    """Backward-compatible alias for the contrastive minimax formula."""
    return contrast_interval_minimax_regret(lower, upper)


def weissman_l1_radius(count: int, n_states: int, alpha: float) -> float:
    """Weissman et al. simultaneous L1 radius for one multinomial row."""
    if count < 0 or n_states < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid arguments")
    if count == 0:
        return 2.0
    factor = max(2**n_states - 2, 1)
    return float(min(2.0, np.sqrt(2.0 * np.log(factor / alpha) / count)))


def transition_improvement_penalty(eta_by_time: Array) -> float:
    """Return the candidate-plus-baseline transition simulation penalty."""
    eta = np.asarray(eta_by_time, dtype=float)
    if eta.ndim != 1 or np.any(eta < 0.0):
        raise ValueError("eta_by_time must be a nonnegative vector")
    horizon = eta.size + 1
    return float(
        sum((horizon - (time + 1)) * eta[time] for time in range(eta.size))
    )


def optimize_robust_improvement(
    mdp: TabularMDP,
    baseline_policy: Array,
    reward_lower: Array,
    reward_upper: Array,
) -> RobustImprovementResult:
    """Solve ``max_pi min_r V_r(pi)-V_r(pi_b)`` as an occupancy LP."""
    baseline = validate_policy(mdp, baseline_policy)
    lower = np.asarray(reward_lower, dtype=float)
    upper = np.asarray(reward_upper, dtype=float)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if lower.shape != shape or upper.shape != shape:
        raise ValueError(f"reward arrays must have shape {shape}")
    if np.any(lower > upper + 1e-12):
        raise ValueError("reward lower endpoint exceeds upper endpoint")

    baseline_occupancy = policy_occupancy(mdp, baseline)
    n_cells = int(np.prod(shape))
    t_offset = n_cells
    n_variables = 2 * n_cells

    # scipy.optimize.linprog minimizes, so negate the sum of hypograph variables.
    objective = np.zeros(n_variables)
    objective[t_offset:] = -1.0

    flow_rows: list[Array] = []
    flow_rhs: list[float] = []
    for time in range(mdp.horizon):
        for state in range(mdp.n_states):
            row = np.zeros(n_variables)
            for action in range(mdp.n_actions):
                cell = np.ravel_multi_index((time, state, action), shape)
                row[cell] = 1.0
            if time == 0:
                target = float(mdp.initial[state])
            else:
                target = 0.0
                for previous_state in range(mdp.n_states):
                    for previous_action in range(mdp.n_actions):
                        previous_cell = np.ravel_multi_index(
                            (time - 1, previous_state, previous_action),
                            shape,
                        )
                        row[previous_cell] -= mdp.transition[
                            time - 1,
                            previous_state,
                            previous_action,
                            state,
                        ]
            flow_rows.append(row)
            flow_rhs.append(target)

    inequality_rows: list[Array] = []
    inequality_rhs: list[float] = []
    lower_flat = lower.ravel()
    upper_flat = upper.ravel()
    baseline_flat = baseline_occupancy.ravel()
    for cell in range(n_cells):
        lower_row = np.zeros(n_variables)
        lower_row[t_offset + cell] = 1.0
        lower_row[cell] = -lower_flat[cell]
        inequality_rows.append(lower_row)
        inequality_rhs.append(-lower_flat[cell] * baseline_flat[cell])

        upper_row = np.zeros(n_variables)
        upper_row[t_offset + cell] = 1.0
        upper_row[cell] = -upper_flat[cell]
        inequality_rows.append(upper_row)
        inequality_rhs.append(-upper_flat[cell] * baseline_flat[cell])

    result = linprog(
        objective,
        A_ub=np.vstack(inequality_rows),
        b_ub=np.asarray(inequality_rhs),
        A_eq=np.vstack(flow_rows),
        b_eq=np.asarray(flow_rhs),
        bounds=[(0.0, None)] * n_cells + [(None, None)] * n_cells,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"robust-improvement LP failed: {result.message}")

    occupancy = result.x[:n_cells].reshape(shape)
    policy = policy_from_occupancy(
        mdp,
        occupancy,
        fallback_policy=baseline,
    )
    certificate = sharp_improvement_lower_bound(
        occupancy,
        baseline_occupancy,
        lower,
        upper,
    )
    return RobustImprovementResult(
        policy=policy,
        occupancy=occupancy,
        certificate=certificate,
        solver_status=result.message,
    )
