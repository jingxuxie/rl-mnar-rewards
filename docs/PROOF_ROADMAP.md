# Proof and research roadmap

## Completed core results

1. **Nonidentification:** two observationally equivalent one-step models have opposite optimal actions.
2. **Sharp binary reward bounds:** closed-form endpoints under the reward-observation odds-ratio sensitivity model.
3. **Finite-valued rewards:** sharp endpoints reduce to a linear program over missing-case reward probabilities.
4. **Sharp policy values:** rectangular cellwise reward intervals propagate exactly through fixed-policy occupancies.
5. **Sharp baseline-relative improvement:** the adversary selects the lower endpoint on positive occupancy contrast and the upper endpoint on negative contrast.
6. **Cancellation identity:** direct improvement equals separate value bounds plus shared occupancy times interval width.
7. **Exact policy optimization:** a linear program over occupancy and hypograph variables maximizes worst-case baseline improvement.
8. **Finite-sample simultaneous confidence:** Bonferroni Clopper–Pearson nuisance intervals induce outer reward intervals valid for all data-dependent policies.
9. **Safety and robust regret:** simultaneous coverage gives safe improvement and a `2 H rho` objective-regret bound.
10. **Estimated transitions:** a uniform simulation penalty yields a conservative extension.

## Next proof priorities

1. Replace Bonferroni intervals with a sharper joint multinomial or empirical-Bernstein construction and quantify the gain.
2. Derive a minimax lower bound showing the interval width and linear dependence on `Gamma` are unavoidable.
3. Characterize non-rectangular sensitivity models that share a global recording mechanism across state-action cells.
4. Intersect sensitivity sets with approximate future-state bridge restrictions and prove monotone tightening.
5. Extend baseline-relative optimization to continuous rewards through discretization or moment constraints.

## Next experiment priorities

1. Add finite-sample random-MDP results with estimated transitions and explicit transition penalties.
2. Add an approximate shadow-variable experiment showing how future-state information shrinks sensitivity intervals.
3. Stress-test misspecified `Gamma`, dataset coverage, and behavior-policy quality.
4. Add a small healthcare-style sepsis simulator only after the theoretical ablations are complete.
