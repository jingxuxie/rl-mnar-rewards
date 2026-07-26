import numpy as np

from mnar_rl.bounds import (
    ambiguous_bandit_minimax_regret,
    binary_missing_success_bounds,
    binary_reward_confidence_bounds,
    binary_reward_mean_bounds,
    contrast_interval_minimax_regret,
    finite_reward_mean_bounds,
    odds_ratio_from_observed_missing,
)


def test_gamma_one_collapses_to_observed_mean():
    q = np.array([0.1, 0.5, 1.0])
    p = np.array([0.2, 0.6, 0.9])
    lower, upper = binary_reward_mean_bounds(q, p, gamma=1.0)
    np.testing.assert_allclose(lower, p)
    np.testing.assert_allclose(upper, p)


def test_binary_endpoints_saturate_odds_ratio():
    p = 0.37
    gamma = 2.4
    lower_missing, upper_missing = binary_missing_success_bounds(p, gamma)
    assert np.isclose(odds_ratio_from_observed_missing(p, lower_missing), gamma)
    assert np.isclose(odds_ratio_from_observed_missing(p, upper_missing), 1.0 / gamma)


def test_bounds_expand_monotonically():
    q, p = 0.35, 0.62
    lower_one, upper_one = binary_reward_mean_bounds(q, p, 1.2)
    lower_two, upper_two = binary_reward_mean_bounds(q, p, 3.0)
    assert lower_two <= lower_one <= upper_one <= upper_two


def test_finite_support_matches_binary_formula():
    q, p, gamma = 0.42, 0.63, 2.2
    binary = binary_reward_mean_bounds(q, p, gamma)
    finite = finite_reward_mean_bounds(q, [1 - p, p], [0.0, 1.0], gamma)
    np.testing.assert_allclose(finite, binary, atol=1e-9)


def test_one_sided_confidence_mapping_is_outer_on_known_example():
    total = np.array([[[400, 400]]])
    observed = np.array([[[160, 160]]])
    successes = np.array([[[64, 96]]])
    bounds = binary_reward_confidence_bounds(total, observed, successes, gamma=2.0, delta=0.05)
    population_lower, population_upper = binary_reward_mean_bounds(0.4, np.array([0.4, 0.6]), 2.0)
    assert np.all(bounds.reward_lower.reshape(-1) <= population_lower + 1e-12)
    assert np.all(bounds.reward_upper.reshape(-1) >= population_upper - 1e-12)
    assert np.isclose(bounds.tail_alpha, 0.05 / 6.0)


def test_ambiguous_bandit_minimax_formula_and_dominance_cases():
    result = ambiguous_bandit_minimax_regret(0.2, 0.5, 0.9)
    assert np.isclose(result.deterministic_regret, 0.3)
    assert np.isclose(result.candidate_probability, 0.4 / 0.7)
    assert np.isclose(result.randomized_regret, 0.3 * 0.4 / 0.7)

    candidate_dominates = ambiguous_bandit_minimax_regret(0.6, 0.5, 0.9)
    assert candidate_dominates.candidate_probability == 1.0
    assert candidate_dominates.randomized_regret == 0.0

    baseline_dominates = ambiguous_bandit_minimax_regret(0.1, 0.95, 0.9)
    assert baseline_dominates.candidate_probability == 0.0
    assert baseline_dominates.randomized_regret == 0.0


def test_contrast_interval_minimax_formula():
    result = contrast_interval_minimax_regret(-0.3, 0.4)
    assert np.isclose(result.deterministic_regret, 0.3)
    assert np.isclose(result.candidate_probability, 0.4 / 0.7)
    assert np.isclose(result.randomized_regret, 0.3 * 0.4 / 0.7)

    candidate_dominates = contrast_interval_minimax_regret(0.1, 0.6)
    assert candidate_dominates.candidate_probability == 1.0
    assert candidate_dominates.randomized_regret == 0.0

    baseline_dominates = contrast_interval_minimax_regret(-0.6, -0.1)
    assert baseline_dominates.candidate_probability == 0.0
    assert baseline_dominates.randomized_regret == 0.0
