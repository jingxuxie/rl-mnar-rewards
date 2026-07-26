"""Small finite-horizon tabular MDP utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TabularMDP:
    """Finite-horizon MDP with time-indexed transition kernels.

    Attributes
    ----------
    transition:
        Array with shape ``(H, S, A, S)``. The final time slice is retained for
        a uniform representation, even though no continuation value follows it.
    initial:
        Initial-state distribution with shape ``(S,)``.
    """

    transition: Array
    initial: Array

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=float)
        initial = np.asarray(self.initial, dtype=float)
        if transition.ndim != 4:
            raise ValueError("transition must have shape (H,S,A,S)")
        _, n_states, _, next_states = transition.shape
        if n_states != next_states or initial.shape != (n_states,):
            raise ValueError("incompatible transition and initial shapes")
        if np.any(transition < -1e-12) or not np.allclose(
            transition.sum(axis=-1), 1.0, atol=1e-9
        ):
            raise ValueError("transition rows must be probability vectors")
        if np.any(initial < -1e-12) or not np.isclose(initial.sum(), 1.0, atol=1e-9):
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
    """Validate and return a Markov policy with shape ``(H,S,A)``."""
    policy_array = np.asarray(policy, dtype=float)
    expected_shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if policy_array.shape != expected_shape:
        raise ValueError(f"policy must have shape {expected_shape}")
    if np.any(policy_array < -1e-12) or not np.allclose(
        policy_array.sum(axis=-1), 1.0, atol=1e-9
    ):
        raise ValueError("policy rows must be probability vectors")
    return policy_array


def policy_occupancy(mdp: TabularMDP, policy: Array) -> Array:
    """Return ``d_h(s,a)=P_pi(S_h=s,A_h=a)``."""
    policy_array = validate_policy(mdp, policy)
    occupancy = np.zeros_like(policy_array)
    state_distribution = np.asarray(mdp.initial, dtype=float).copy()
    for time in range(mdp.horizon):
        occupancy[time] = state_distribution[:, None] * policy_array[time]
        if time + 1 < mdp.horizon:
            state_distribution = np.einsum(
                "sa,san->n", occupancy[time], mdp.transition[time]
            )
    return occupancy


def policy_from_occupancy(
    mdp: TabularMDP,
    occupancy: Array,
    fallback_policy: Array | None = None,
) -> Array:
    """Recover a Markov policy from a feasible occupancy array.

    Actions at unreachable state-time pairs are taken from ``fallback_policy``;
    a uniform policy is used when no fallback is supplied.
    """
    occupancy_array = np.asarray(occupancy, dtype=float)
    expected_shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if occupancy_array.shape != expected_shape or np.any(occupancy_array < -1e-10):
        raise ValueError(
            f"occupancy must have shape {expected_shape} and be nonnegative"
        )
    if fallback_policy is None:
        fallback = np.full(expected_shape, 1.0 / mdp.n_actions)
    else:
        fallback = validate_policy(mdp, fallback_policy)

    policy = np.empty(expected_shape)
    for time in range(mdp.horizon):
        for state in range(mdp.n_states):
            mass = float(occupancy_array[time, state].sum())
            policy[time, state] = (
                occupancy_array[time, state] / mass
                if mass > 1e-12
                else fallback[time, state]
            )
    return policy


def policy_value(mdp: TabularMDP, reward: Array, policy: Array) -> float:
    """Evaluate a policy exactly for an expected reward table."""
    reward_array = np.asarray(reward, dtype=float)
    expected_shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if reward_array.shape != expected_shape:
        raise ValueError(f"reward must have shape {expected_shape}")
    return float(np.sum(policy_occupancy(mdp, policy) * reward_array))


def optimal_policy(mdp: TabularMDP, reward: Array) -> tuple[Array, float]:
    """Compute an optimal deterministic policy by backward induction."""
    reward_array = np.asarray(reward, dtype=float)
    expected_shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if reward_array.shape != expected_shape:
        raise ValueError(f"reward must have shape {expected_shape}")

    next_value = np.zeros(mdp.n_states)
    policy = np.zeros(expected_shape)
    for time in range(mdp.horizon - 1, -1, -1):
        continuation = np.einsum(
            "san,n->sa", mdp.transition[time], next_value
        )
        action_values = reward_array[time] + continuation
        actions = np.argmax(action_values, axis=1)
        policy[time, np.arange(mdp.n_states), actions] = 1.0
        next_value = action_values[np.arange(mdp.n_states), actions]
    return policy, float(np.asarray(mdp.initial) @ next_value)


def epsilon_soft(policy: Array, epsilon: float) -> Array:
    """Mix a policy with the uniform action distribution."""
    policy_array = np.asarray(policy, dtype=float)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    return (1.0 - epsilon) * policy_array + epsilon / policy_array.shape[-1]


def random_mdp(
    rng: np.random.Generator,
    horizon: int,
    n_states: int,
    n_actions: int,
    concentration: float = 0.7,
) -> TabularMDP:
    """Generate a random tabular MDP with Dirichlet transition rows."""
    if horizon < 1 or n_states < 1 or n_actions < 1 or concentration <= 0:
        raise ValueError("invalid random MDP dimensions")
    transition = rng.dirichlet(
        np.full(n_states, concentration),
        size=(horizon, n_states, n_actions),
    )
    initial = np.zeros(n_states)
    initial[0] = 1.0
    return TabularMDP(transition=transition, initial=initial)
