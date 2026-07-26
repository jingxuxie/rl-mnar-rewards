import numpy as np
from mnar_rl import (
    binary_missing_success_bounds,
    binary_reward_confidence_bounds,
    binary_reward_interval_width,
    binary_reward_mean_bounds,
    finite_reward_mean_bounds,
    maximum_binary_reward_interval_width,
    odds_ratio_from_observed_missing,
)


def test_gamma_one_collapses_to_observed_mean():
    q=np.array([0.1,0.5,1.0]);p=np.array([0.2,0.6,0.9])
    lo,hi=binary_reward_mean_bounds(q,p,1.0)
    np.testing.assert_allclose(lo,p);np.testing.assert_allclose(hi,p)


def test_binary_endpoints_saturate_odds_ratio():
    p=0.37;gamma=2.4;lo_u,hi_u=binary_missing_success_bounds(p,gamma)
    assert np.isclose(odds_ratio_from_observed_missing(p,lo_u),gamma)
    assert np.isclose(odds_ratio_from_observed_missing(p,hi_u),1/gamma)


def test_bounds_expand_monotonically():
    q,p=0.35,0.62
    lo1,hi1=binary_reward_mean_bounds(q,p,1.2);lo2,hi2=binary_reward_mean_bounds(q,p,3.0)
    assert lo2<=lo1<=hi1<=hi2


def test_finite_support_matches_binary_formula():
    q,p,gamma=0.42,0.63,2.2
    binary=binary_reward_mean_bounds(q,p,gamma)
    finite=finite_reward_mean_bounds(q,[1-p,p],[0.0,1.0],gamma)
    np.testing.assert_allclose(finite,binary,atol=1e-9)


def test_closed_form_width_matches_endpoint_difference():
    q=np.array([0.2,0.6]);p=np.array([0.17,0.71]);gamma=3.4
    lo,hi=binary_reward_mean_bounds(q,p,gamma)
    np.testing.assert_allclose(binary_reward_interval_width(q,p,gamma),hi-lo)


def test_width_envelope_is_attained_at_half():
    gamma=4.0;q=0.3
    envelope=maximum_binary_reward_interval_width(q,gamma)
    assert np.isclose(binary_reward_interval_width(q,0.5,gamma),envelope)
    grid=np.linspace(0,1,1001)
    assert np.max(binary_reward_interval_width(q,grid,gamma))<=envelope+1e-12


def test_one_sided_outer_interval_is_vacuous_without_observed_rewards():
    result=binary_reward_confidence_bounds(
        np.array([[[10]]]),np.array([[[0]]]),np.array([[[0]]]),gamma=2.0,delta=0.05
    )
    assert result.reward_lower.item()==0.0
    assert result.reward_upper.item()==1.0
