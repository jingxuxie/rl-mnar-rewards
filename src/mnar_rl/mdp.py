"""Small finite-horizon tabular MDP utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TabularMDP:
    transition: Array  # (H, S, A, S); last slice may be arbitrary
    initial: Array  # (S,)

    def __post_init__(self) -> None:
        p = np.asarray(self.transition, dtype=float)
        mu = np.asarray(self.initial, dtype=float)
        if p.ndim != 4:
            raise ValueError("transition must have shape (H,S,A,S)")
        h, s, a, s2 = p.shape
        if s != s2 or mu.shape != (s,):
            raise ValueError("incompatible transition and initial shapes")
        if np.any(p < -1e-12) or not np.allclose(p.sum(axis=-1), 1.0, atol=1e-9):
            raise ValueError("transition rows must be probability vectors")
        if np.any(mu < -1e-12) or not np.isclose(mu.sum(), 1.0, atol=1e-9):
            raise ValueError("initial must be a probability vector")

    @property
    def horizon(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.transition.shape[2])


def validate_policy(mdp: TabularMDP, policy: Array) -> Array:
    pi = np.asarray(policy, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if pi.shape != expected:
        raise ValueError(f"policy must have shape {expected}")
    if np.any(pi < -1e-12) or not np.allclose(pi.sum(axis=-1), 1.0, atol=1e-9):
        raise ValueError("policy rows must be probability vectors")
    return pi


def policy_occupancy(mdp: TabularMDP, policy: Array) -> Array:
    """Return d_h(s,a)=P(S_h=s,A_h=a)."""
    pi = validate_policy(mdp, policy)
    d = np.zeros_like(pi)
    state = mdp.initial.copy()
    for h in range(mdp.horizon):
        d[h] = state[:, None] * pi[h]
        if h + 1 < mdp.horizon:
            state = np.einsum("sa,san->n", d[h], mdp.transition[h])
    return d


def policy_value(mdp: TabularMDP, reward: Array, policy: Array) -> float:
    r = np.asarray(reward, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if r.shape != expected:
        raise ValueError(f"reward must have shape {expected}")
    return float(np.sum(policy_occupancy(mdp, policy) * r))


def optimal_policy(mdp: TabularMDP, reward: Array) -> tuple[Array, float]:
    """Backward induction for a deterministic Markov policy."""
    r = np.asarray(reward, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if r.shape != expected:
        raise ValueError(f"reward must have shape {expected}")

    value_next = np.zeros(mdp.n_states)
    policy = np.zeros(expected)
    for h in range(mdp.horizon - 1, -1, -1):
        continuation = np.einsum("san,n->sa", mdp.transition[h], value_next)
        q = r[h] + continuation
        actions = np.argmax(q, axis=1)
        policy[h, np.arange(mdp.n_states), actions] = 1.0
        value_next = q[np.arange(mdp.n_states), actions]
    return policy, float(mdp.initial @ value_next)


def epsilon_soft(policy: Array, epsilon: float) -> Array:
    pi = np.asarray(policy, dtype=float)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    actions = pi.shape[-1]
    return (1.0 - epsilon) * pi + epsilon / actions


def random_mdp(
    rng: np.random.Generator,
    horizon: int,
    n_states: int,
    n_actions: int,
    concentration: float = 0.7,
) -> TabularMDP:
    if horizon < 1 or n_states < 1 or n_actions < 1 or concentration <= 0:
        raise ValueError("invalid random MDP dimensions")
    transition = rng.dirichlet(
        np.full(n_states, concentration),
        size=(horizon, n_states, n_actions),
    )
    initial = np.zeros(n_states)
    initial[0] = 1.0
    return TabularMDP(transition=transition, initial=initial)
