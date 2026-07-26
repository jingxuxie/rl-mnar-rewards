import itertools

import numpy as np

from mnar_rl.mdp import TabularMDP, policy_occupancy, policy_value
from mnar_rl.optimization import optimize_robust_improvement, sharp_improvement_lower_bound


def small_mdp():
    h, s, a = 2, 2, 2
    p = np.zeros((h, s, a, s))
    p[0, 0, 0, 0] = 1.0
    p[0, 0, 1, 1] = 1.0
    p[0, 1, :, 1] = 1.0
    p[1, :, :, :] = np.eye(s)[:, None, :]
    mu = np.array([1.0, 0.0])
    return TabularMDP(p, mu)


def deterministic_policy(mdp, actions):
    pi = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions))
    for h in range(mdp.horizon):
        for s in range(mdp.n_states):
            pi[h, s, actions[h, s]] = 1.0
    return pi


def test_lp_matches_enumeration_over_deterministic_policies():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    lo = np.array([[[0.1, 0.2], [0.0, 0.0]], [[0.2, 0.8], [0.3, 0.9]]])
    hi = lo + 0.2
    result = optimize_robust_improvement(mdp, baseline, lo, hi)
    db = policy_occupancy(mdp, baseline)

    best = -np.inf
    for flat in itertools.product(range(mdp.n_actions), repeat=mdp.horizon * mdp.n_states):
        actions = np.asarray(flat).reshape(mdp.horizon, mdp.n_states)
        pi = deterministic_policy(mdp, actions)
        value = sharp_improvement_lower_bound(policy_occupancy(mdp, pi), db, lo, hi)
        best = max(best, value)
    assert result.certificate >= best - 1e-8
    # Randomization can improve a concave robust objective, so equality to the
    # deterministic enumeration is not required; the baseline guarantees >=0.
    assert result.certificate >= -1e-9


def test_certificate_is_valid_for_sampled_rewards():
    mdp = small_mdp()
    baseline = deterministic_policy(mdp, np.zeros((2, 2), dtype=int))
    lo = np.full((2, 2, 2), 0.2)
    hi = np.full((2, 2, 2), 0.8)
    lo[1, 1, 1] = 0.9
    hi[1, 1, 1] = 1.0
    result = optimize_robust_improvement(mdp, baseline, lo, hi)

    rng = np.random.default_rng(0)
    for _ in range(100):
        reward = rng.uniform(lo, hi)
        improvement = policy_value(mdp, reward, result.policy) - policy_value(mdp, reward, baseline)
        assert improvement + 1e-8 >= result.certificate
