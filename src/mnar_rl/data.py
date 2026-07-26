"""Simulation and estimation for tabular MNAR reward models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mdp import TabularMDP, validate_policy


Array = np.ndarray


@dataclass(frozen=True)
class MNARRewardModel:
    q_observed: Array
    p_success_observed: Array
    p_success_missing: Array

    def __post_init__(self) -> None:
        q = np.asarray(self.q_observed, dtype=float)
        p = np.asarray(self.p_success_observed, dtype=float)
        u = np.asarray(self.p_success_missing, dtype=float)
        if q.shape != p.shape or q.shape != u.shape:
            raise ValueError("MNAR arrays must have identical shapes")
        if np.any((q < 0) | (q > 1)) or np.any((p < 0) | (p > 1)) or np.any((u < 0) | (u > 1)):
            raise ValueError("MNAR probabilities must lie in [0,1]")

    @property
    def mean_reward(self) -> Array:
        q = np.asarray(self.q_observed)
        return q * np.asarray(self.p_success_observed) + (1.0 - q) * np.asarray(self.p_success_missing)


@dataclass(frozen=True)
class TabularCounts:
    total: Array
    observed: Array
    observed_success: Array
    transition: Array


def missing_probability_from_odds_ratio(p_obs: Array, odds_ratio: Array) -> Array:
    """Solve odds(p_obs)/odds(p_miss)=odds_ratio."""
    p = np.asarray(p_obs, dtype=float)
    theta = np.asarray(odds_ratio, dtype=float)
    if np.any((p <= 0) | (p >= 1)) or np.any(theta <= 0):
        raise ValueError("p_obs must be interior and odds_ratio positive")
    odds_u = (p / (1.0 - p)) / theta
    return odds_u / (1.0 + odds_u)


def simulate_counts(
    rng: np.random.Generator,
    mdp: TabularMDP,
    behavior_policy: Array,
    reward_model: MNARRewardModel,
    n_episodes: int,
) -> TabularCounts:
    """Simulate sufficient statistics from independent trajectories."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    beta = validate_policy(mdp, behavior_policy)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if reward_model.q_observed.shape != shape:
        raise ValueError(f"reward model must have shape {shape}")

    total = np.zeros(shape, dtype=int)
    observed = np.zeros(shape, dtype=int)
    success = np.zeros(shape, dtype=int)
    transition = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions, mdp.n_states), dtype=int)

    for _ in range(n_episodes):
        state = int(rng.choice(mdp.n_states, p=mdp.initial))
        for h in range(mdp.horizon):
            action = int(rng.choice(mdp.n_actions, p=beta[h, state]))
            idx = (h, state, action)
            total[idx] += 1
            is_observed = bool(rng.random() < reward_model.q_observed[idx])
            if is_observed:
                observed[idx] += 1
                reward = bool(rng.random() < reward_model.p_success_observed[idx])
                success[idx] += int(reward)
            else:
                # The latent reward is sampled for fidelity but intentionally discarded.
                _ = bool(rng.random() < reward_model.p_success_missing[idx])

            next_state = int(rng.choice(mdp.n_states, p=mdp.transition[h, state, action]))
            transition[h, state, action, next_state] += 1
            state = next_state

    return TabularCounts(total, observed, success, transition)


def plug_in_q_p(counts: TabularCounts, default_q: float = 0.0, default_p: float = 0.5) -> tuple[Array, Array]:
    q = np.full(counts.total.shape, default_q, dtype=float)
    p = np.full(counts.total.shape, default_p, dtype=float)
    np.divide(counts.observed, counts.total, out=q, where=counts.total > 0)
    np.divide(counts.observed_success, counts.observed, out=p, where=counts.observed > 0)
    return q, p
