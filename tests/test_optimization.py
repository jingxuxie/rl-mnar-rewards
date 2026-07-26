import itertools

import numpy as np

from mnar_rl.mdp import (
    TabularMDP,
    policy_occupancy,
    policy_value,
    transition_improvement_penalty,
    weissman_l1_radius,
)
from mnar_rl.optimization import (
    cancellation_gain,
    optimize_robust_improvement,
    separate_value_lower_bound,
    sharp_improvement_interval,
    sharp_improvement_lower_bound,
)


def small_mdp():
    horizon, states, actions = 2, 2, 2
    transition = np.zeros((horizon, states, actions, states))
    transition[0, 0, 0, 0] = 1.0
    transition[0, 0, 1, 1] = 1.0
    transition[0, 1, :, 1] = 1.0
    transition[1, :, :, :] = np.eye(states)[:, None, :]
    return TabularMDP(transition, np.array([1.0, 0.0]))


def deterministic_policy(mdp, actions):
    policy = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions))
    for h in range(mdp.horizon):
        for state in range(mdp.n_states):
            policy[h, state, actions[h, state]] = 1.0
    return policy


def test_lp_dominates_enumeration_over_deterministic_policies():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    lower = np.array([[[0.1, 0.2], [0.0, 0.0]], [[0.2, 0.8], [0.3, 0.9]]])
    upper = lower + 0.2
    result = optimize_robust_improvement(mdp, baseline, lower, upper)
    baseline_occupancy = policy_occupancy(mdp, baseline)

    best = -np.inf
    for flat in itertools.product(range(mdp.n_actions), repeat=mdp.horizon * mdp.n_states):
        actions = np.asarray(flat).reshape(mdp.horizon, mdp.n_states)
        policy = deterministic_policy(mdp, actions)
        value = sharp_improvement_lower_bound(
            policy_occupancy(mdp, policy),
            baseline_occupancy,
            lower,
            upper,
        )
        best = max(best, value)
    assert result.certificate >= best - 1e-8
    assert result.certificate >= -1e-9


def test_certificate_is_valid_for_sampled_rewards():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    lower = np.full((2, 2, 2), 0.2)
    upper = np.full((2, 2, 2), 0.8)
    lower[1, 1, 1] = 0.9
    upper[1, 1, 1] = 1.0
    result = optimize_robust_improvement(mdp, baseline, lower, upper)

    rng = np.random.default_rng(0)
    for _ in range(100):
        reward = rng.uniform(lower, upper)
        improvement = policy_value(mdp, reward, result.policy) - policy_value(mdp, reward, baseline)
        assert improvement + 1e-8 >= result.certificate


def test_contrastive_interval_is_jointly_sharp_by_endpoint_construction():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    candidate_actions = np.array([[1, 0], [0, 1]])
    candidate = deterministic_policy(mdp, candidate_actions)
    baseline_occupancy = policy_occupancy(mdp, baseline)
    candidate_occupancy = policy_occupancy(mdp, candidate)
    lower = np.full((2, 2, 2), 0.1)
    upper = np.full((2, 2, 2), 0.9)
    identified_lower, identified_upper = sharp_improvement_interval(
        candidate_occupancy,
        baseline_occupancy,
        lower,
        upper,
    )
    contrast = candidate_occupancy - baseline_occupancy
    reward_for_lower = np.where(contrast >= 0.0, lower, upper)
    reward_for_upper = np.where(contrast >= 0.0, upper, lower)
    assert np.isclose(
        policy_value(mdp, reward_for_lower, candidate) - policy_value(mdp, reward_for_lower, baseline),
        identified_lower,
    )
    assert np.isclose(
        policy_value(mdp, reward_for_upper, candidate) - policy_value(mdp, reward_for_upper, baseline),
        identified_upper,
    )


def test_cancellation_identity():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    candidate = deterministic_policy(mdp, np.ones((2, 2), dtype=int))
    lower = np.full((2, 2, 2), 0.2)
    upper = np.full((2, 2, 2), 0.8)
    direct = sharp_improvement_lower_bound(
        policy_occupancy(mdp, candidate),
        policy_occupancy(mdp, baseline),
        lower,
        upper,
    )
    separate = separate_value_lower_bound(mdp, candidate, baseline, lower, upper)
    gain = cancellation_gain(
        policy_occupancy(mdp, candidate),
        policy_occupancy(mdp, baseline),
        lower,
        upper,
    )
    assert np.isclose(direct, separate + gain)


def test_contrastive_interval_width_equals_occupancy_difference_weighted_width():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    candidate = deterministic_policy(mdp, np.ones((2, 2), dtype=int))
    baseline_occupancy = policy_occupancy(mdp, baseline)
    candidate_occupancy = policy_occupancy(mdp, candidate)
    lower = np.array([[[0.1, 0.3], [0.2, 0.4]], [[0.15, 0.25], [0.5, 0.6]]])
    upper = lower + np.array([[[0.2, 0.1], [0.3, 0.2]], [[0.1, 0.4], [0.2, 0.3]]])
    identified_lower, identified_upper = sharp_improvement_interval(
        candidate_occupancy,
        baseline_occupancy,
        lower,
        upper,
    )
    predicted_width = np.sum(
        np.abs(candidate_occupancy - baseline_occupancy) * (upper - lower)
    )
    assert np.isclose(identified_upper - identified_lower, predicted_width)


def test_weissman_radius_and_transition_penalty():
    assert weissman_l1_radius(0, n_states=3, alpha=0.05) == 2.0
    assert weissman_l1_radius(10, n_states=1, alpha=0.05) == 0.0
    assert weissman_l1_radius(1000, n_states=2, alpha=0.05) < weissman_l1_radius(100, n_states=2, alpha=0.05)
    np.testing.assert_allclose(
        transition_improvement_penalty(np.array([0.2, 0.1, 0.05])),
        0.85,
    )


def test_transition_penalty_covers_two_policy_value_difference():
    # One consequential transition layer. For a binary next state, row L1
    # error is twice the success-probability error. Each policy value moves by
    # at most eta/2 and their difference by at most eta.
    p_true = np.array([0.55, 0.50])
    p_hat = np.array([0.58, 0.46])
    row_l1 = 2.0 * np.abs(p_true - p_hat)
    penalty = transition_improvement_penalty(np.array([row_l1.max()]))
    true_difference = p_true[1] - p_true[0]
    estimated_difference = p_hat[1] - p_hat[0]
    assert true_difference >= estimated_difference - penalty - 1e-12
