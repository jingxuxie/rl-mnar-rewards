import itertools
import numpy as np
from mnar_rl import (
    TabularMDP,
    binary_reward_mean_bounds,
    cancellation_gain,
    contrast_interval_minimax_regret,
    contrastive_ambiguity_width,
    contrastive_missingness_budget,
    optimize_robust_improvement,
    policy_occupancy,
    policy_value,
    sharp_improvement_lower_bound,
    sharp_improvement_upper_bound,
    transition_improvement_penalty,
    weissman_l1_radius,
)


def small_mdp():
    h,s,a=2,2,2;p=np.zeros((h,s,a,s))
    p[0,0,0,0]=1;p[0,0,1,1]=1;p[0,1,:,1]=1;p[1,:,:,:]=np.eye(s)[:,None,:]
    return TabularMDP(p,np.array([1.0,0.0]))


def deterministic_policy(mdp,actions):
    pi=np.zeros((mdp.horizon,mdp.n_states,mdp.n_actions))
    for h in range(mdp.horizon):
        for s in range(mdp.n_states): pi[h,s,actions[h,s]]=1
    return pi


def test_lp_dominates_deterministic_enumeration():
    mdp=small_mdp();baseline=deterministic_policy(mdp,np.zeros((2,2),dtype=int))
    lo=np.array([[[.1,.2],[0,0]],[[.2,.8],[.3,.9]]]);hi=lo+.2
    result=optimize_robust_improvement(mdp,baseline,lo,hi);db=policy_occupancy(mdp,baseline)
    best=-np.inf
    for flat in itertools.product(range(mdp.n_actions),repeat=mdp.horizon*mdp.n_states):
        pi=deterministic_policy(mdp,np.asarray(flat).reshape(mdp.horizon,mdp.n_states))
        best=max(best,sharp_improvement_lower_bound(policy_occupancy(mdp,pi),db,lo,hi))
    assert result.certificate>=best-1e-8
    assert result.certificate>=-1e-9


def test_certificate_valid_for_sampled_rewards():
    mdp=small_mdp();baseline=deterministic_policy(mdp,np.zeros((2,2),dtype=int))
    lo=np.full((2,2,2),.2);hi=np.full((2,2,2),.8);lo[1,1,1]=.9;hi[1,1,1]=1
    result=optimize_robust_improvement(mdp,baseline,lo,hi);rng=np.random.default_rng(0)
    for _ in range(100):
        r=rng.uniform(lo,hi)
        improvement=policy_value(mdp,r,result.policy)-policy_value(mdp,r,baseline)
        assert improvement+1e-8>=result.certificate


def test_contrast_width_identity():
    d=np.array([.2,.3,.5]);db=np.array([.4,.1,.5]);lo=np.array([.1,.2,.3]);hi=np.array([.5,.7,.9])
    lower=sharp_improvement_lower_bound(d,db,lo,hi);upper=sharp_improvement_upper_bound(d,db,lo,hi)
    assert np.isclose(upper-lower,contrastive_ambiguity_width(d,db,lo,hi))


def test_cancellation_identity():
    d=np.array([.2,.3,.5]);db=np.array([.4,.1,.5]);lo=np.array([.1,.2,.3]);hi=np.array([.5,.7,.9])
    direct=sharp_improvement_lower_bound(d,db,lo,hi)
    separate=float(d@lo-db@hi)
    assert np.isclose(direct,separate+cancellation_gain(d,db,lo,hi))


def test_contrastive_missingness_envelope():
    q=np.array([.2,.5,.8]);p=np.array([.1,.5,.9]);gamma=3
    lo,hi=binary_reward_mean_bounds(q,p,gamma)
    d=np.array([.1,.3,.6]);db=np.array([.4,.2,.4])
    assert contrastive_ambiguity_width(d,db,lo,hi)<=contrastive_missingness_budget(d,db,q,gamma)+1e-12


def test_minimax_regret_formula():
    r=contrast_interval_minimax_regret(-.2,.6)
    assert np.isclose(r.candidate_probability,.75)
    assert np.isclose(r.randomized_regret,.15)
    assert np.isclose(r.deterministic_regret,.2)


def test_randomization_can_be_strictly_necessary():
    mdp=TabularMDP(np.ones((1,1,3,1)),np.array([1.0]))
    q=np.array([[[9/25,4/25,477/800]]]);p=np.array([[[1/6,5/8,5/9]]])
    lo,hi=binary_reward_mean_bounds(q,p,3);baseline=np.array([[[1/5,7/20,9/20]]])
    result=optimize_robust_improvement(mdp,baseline,lo,hi);db=policy_occupancy(mdp,baseline)
    deterministic=[]
    for action in range(3):
        pi=np.zeros_like(baseline);pi[0,0,action]=1
        deterministic.append(sharp_improvement_lower_bound(policy_occupancy(mdp,pi),db,lo,hi))
    assert max(deterministic)<0
    assert np.isclose(result.certificate,.03)
    np.testing.assert_allclose(result.policy,np.array([[[0,.35,.65]]]),atol=1e-8)


def test_transition_penalty_and_weissman_radius():
    assert transition_improvement_penalty(np.array([.2,.1]))==.5
    assert weissman_l1_radius(0,3,.05)==2.0
    assert 0<weissman_l1_radius(1000,3,.05)<1
