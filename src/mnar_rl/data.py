"""Simulation and estimation for tabular MNAR reward models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import TabularMDP, validate_policy

Array = np.ndarray


@dataclass(frozen=True)
class MNARRewardModel:
    """Cellwise binary reward and observation probabilities."""

    q_observed: Array
    p_success_observed: Array
    p_success_missing: Array

    def __post_init__(self) -> None:
        q_observed = np.asarray(self.q_observed, dtype=float)
        p_observed = np.asarray(self.p_success_observed, dtype=float)
        p_missing = np.asarray(self.p_success_missing, dtype=float)
        if q_observed.shape != p_observed.shape or q_observed.shape != p_missing.shape:
            raise ValueError("MNAR arrays must have identical shapes")
        for name, array in (
            ("q_observed", q_observed),
            ("p_success_observed", p_observed),
            ("p_success_missing", p_missing),
        ):
            if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                raise ValueError(f"{name} probabilities must lie in [0,1]")

    @property
    def mean_reward(self) -> Array:
        q_observed = np.asarray(self.q_observed, dtype=float)
        return (
            q_observed * np.asarray(self.p_success_observed, dtype=float)
            + (1.0 - q_observed) * np.asarray(self.p_success_missing, dtype=float)
        )


@dataclass(frozen=True)
class TabularCounts:
    """Sufficient statistics for the tabular estimators."""

    total: Array
    observed: Array
    observed_success: Array
    transition: Array


def missing_probability_from_odds_ratio(
    p_obs: Array,
    odds_ratio: Array,
) -> Array:
    """Solve ``odds(p_obs)/odds(p_miss)=odds_ratio`` for ``p_miss``."""
    p_observed = np.asarray(p_obs, dtype=float)
    ratio = np.asarray(odds_ratio, dtype=float)
    if np.any((p_observed <= 0.0) | (p_observed >= 1.0)) or np.any(ratio <= 0.0):
        raise ValueError("p_obs must be interior and odds_ratio positive")
    missing_odds = (p_observed / (1.0 - p_observed)) / ratio
    return missing_odds / (1.0 + missing_odds)


def simulate_counts(
    rng: np.random.Generator,
    mdp: TabularMDP,
    behavior_policy: Array,
    reward_model: MNARRewardModel,
    n_episodes: int,
) -> TabularCounts:
    """Simulate the nested counts observed in an offline MNAR dataset."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    behavior = validate_policy(mdp, behavior_policy)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if np.asarray(reward_model.q_observed).shape != shape:
        raise ValueError(f"reward model must have shape {shape}")

    total = np.zeros(shape, dtype=int)
    observed = np.zeros(shape, dtype=int)
    observed_success = np.zeros(shape, dtype=int)
    transition = np.zeros(
        (mdp.horizon, mdp.n_states, mdp.n_actions, mdp.n_states),
        dtype=int,
    )

    for _ in range(n_episodes):
        state = int(rng.choice(mdp.n_states, p=mdp.initial))
        for time in range(mdp.horizon):
            action = int(rng.choice(mdp.n_actions, p=behavior[time, state]))
            cell = (time, state, action)
            total[cell] += 1

            is_observed = bool(rng.random() < reward_model.q_observed[cell])
            if is_observed:
                observed[cell] += 1
                reward = bool(
                    rng.random() < reward_model.p_success_observed[cell]
                )
                observed_success[cell] += int(reward)
            else:
                # Sample the latent reward for fidelity, then intentionally discard it.
                _ = bool(rng.random() < reward_model.p_success_missing[cell])

            next_state = int(
                rng.choice(
                    mdp.n_states,
                    p=mdp.transition[time, state, action],
                )
            )
            transition[time, state, action, next_state] += 1
            state = next_state

    return TabularCounts(
        total=total,
        observed=observed,
        observed_success=observed_success,
        transition=transition,
    )


def plug_in_q_p(
    counts: TabularCounts,
    default_q: float = 0.0,
    default_p: float = 0.5,
) -> tuple[Array, Array]:
    """Return cellwise plug-in estimates of observation and observed-success rates."""
    q_estimate = np.full(counts.total.shape, default_q, dtype=float)
    p_estimate = np.full(counts.total.shape, default_p, dtype=float)
    np.divide(
        counts.observed,
        counts.total,
        out=q_estimate,
        where=counts.total > 0,
    )
    np.divide(
        counts.observed_success,
        counts.observed,
        out=p_estimate,
        where=counts.observed > 0,
    )
    return q_estimate, p_estimate
