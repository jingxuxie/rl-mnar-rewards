"""Small finite-horizon tabular MDP utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class TabularMDP:
    """Finite-horizon MDP with time-indexed transition kernels."""

    transition: Array  # (H, S, A, S); final slice may be arbitrary
    initial: Array  # (S,)

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=float)
        initial = np.asarray(self.initial, dtype=float)
        if transition.ndim != 4:
            raise ValueError("transition must have shape (H,S,A,S)")
        _, states, _, next_states = transition.shape
        if states != next_states or initial.shape != (states,):
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
    """Validate and return a Markov policy array."""
    candidate = np.asarray(policy, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if candidate.shape != expected:
        raise ValueError(f"policy must have shape {expected}")
    if np.any(candidate < -1e-12) or not np.allclose(
        candidate.sum(axis=-1), 1.0, atol=1e-9
    ):
        raise ValueError("policy rows must be probability vectors")
    return candidate


def policy_occupancy(mdp: TabularMDP, policy: Array) -> Array:
    """Return ``d_h(s,a)=P(S_h=s,A_h=a)``."""
    candidate = validate_policy(mdp, policy)
    occupancy = np.zeros_like(candidate)
    state_mass = mdp.initial.copy()
    for h in range(mdp.horizon):
        occupancy[h] = state_mass[:, None] * candidate[h]
        if h + 1 < mdp.horizon:
            state_mass = np.einsum(
                "sa,san->n",
                occupancy[h],
                mdp.transition[h],
            )
    return occupancy


def policy_value(mdp: TabularMDP, reward: Array, policy: Array) -> float:
    """Evaluate a policy exactly from its occupancy measure."""
    reward_array = np.asarray(reward, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if reward_array.shape != expected:
        raise ValueError(f"reward must have shape {expected}")
    return float(np.sum(policy_occupancy(mdp, policy) * reward_array))


def optimal_policy(mdp: TabularMDP, reward: Array) -> tuple[Array, float]:
    """Return a deterministic optimal policy by backward induction."""
    reward_array = np.asarray(reward, dtype=float)
    expected = (mdp.horizon, mdp.n_states, mdp.n_actions)
    if reward_array.shape != expected:
        raise ValueError(f"reward must have shape {expected}")

    value_next = np.zeros(mdp.n_states)
    policy = np.zeros(expected)
    for h in range(mdp.horizon - 1, -1, -1):
        continuation = np.einsum("san,n->sa", mdp.transition[h], value_next)
        action_values = reward_array[h] + continuation
        actions = np.argmax(action_values, axis=1)
        policy[h, np.arange(mdp.n_states), actions] = 1.0
        value_next = action_values[np.arange(mdp.n_states), actions]
    return policy, float(mdp.initial @ value_next)


def epsilon_soft(policy: Array, epsilon: float) -> Array:
    """Mix a policy with the uniform policy."""
    candidate = np.asarray(policy, dtype=float)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    return (1.0 - epsilon) * candidate + epsilon / candidate.shape[-1]


def weissman_l1_radius(samples: int, n_states: int, alpha: float) -> float:
    """Finite-sample L1 radius for one multinomial transition row.

    The radius inverts the Weissman et al. inequality

    ``P(||P_hat-P||_1 >= eps) <= (2**S-2) exp(-N eps**2/2)``

    and is clipped at the largest possible L1 distance, two.
    """
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    if n_states < 1:
        raise ValueError("n_states must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if n_states == 1:
        return 0.0
    if samples == 0:
        return 2.0
    multiplicity = float(2**n_states - 2)
    radius = np.sqrt(2.0 * np.log(multiplicity / alpha) / samples)
    return float(min(2.0, radius))


def transition_improvement_penalty(radii_by_time: Array) -> float:
    """Uniform baseline-relative penalty for estimated transitions.

    ``radii_by_time[h]`` bounds the largest transition-row L1 error at
    zero-indexed decision time ``h``. For an H-step problem there are H-1
    consequential transition layers, and the improvement penalty is

    ``sum_{h=0}^{H-2} (H-h-1) * eta_h``.
    """
    radii = np.asarray(radii_by_time, dtype=float)
    if radii.ndim != 1 or np.any(~np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("radii_by_time must be a finite nonnegative vector")
    coefficients = np.arange(radii.size, 0, -1, dtype=float)
    return float(coefficients @ radii)


def random_mdp(
    rng: np.random.Generator,
    horizon: int,
    n_states: int,
    n_actions: int,
    concentration: float = 0.7,
) -> TabularMDP:
    """Generate a small random tabular MDP for controlled validations."""
    if horizon < 1 or n_states < 1 or n_actions < 1 or concentration <= 0.0:
        raise ValueError("invalid random MDP dimensions")
    transition = rng.dirichlet(
        np.full(n_states, concentration),
        size=(horizon, n_states, n_actions),
    )
    initial = np.zeros(n_states)
    initial[0] = 1.0
    return TabularMDP(transition=transition, initial=initial)
