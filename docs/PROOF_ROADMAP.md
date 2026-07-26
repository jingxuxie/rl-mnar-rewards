# Proof roadmap and audit status

## Completed theorem stack

1. **Unrestricted nonidentification.** Two full-data reward mechanisms induce the same observed law but reverse the optimal action.
2. **Sharp binary cell bounds.** The odds-ratio model is algebraically equivalent to an interval for the missing-case success probability; every point is attainable.
3. **Geometry of MNAR ambiguity.** The interval width has a closed form, is maximized at observed success probability `1/2`, and is bounded by `(1-q)(Gamma-1)/(Gamma+1)`.
4. **Contrastive missingness budget.** Pairwise policy ambiguity is at most the sensitivity factor times missingness-weighted candidate-baseline occupancy distance, and at most `2H(Gamma-1)/(Gamma+1)` globally.
5. **Finite-alphabet extension.** Pairwise observation-odds restrictions form a polytope, so sharp mean endpoints are small linear programs.
6. **Sharp fixed-policy values.** Rectangular cellwise ambiguity propagates exactly through nonnegative policy occupancies.
7. **Sharp contrastive interval.** Candidate-minus-baseline value uses lower reward endpoints on positive occupancy contrast and upper endpoints on negative contrast; reversing endpoints gives the sharp upper bound.
8. **Cancellation identity.** Direct comparison equals separate robust values plus interval width weighted by shared occupancy.
9. **Exact occupancy LP.** Hypograph variables represent the minimum of the two affine cell contributions without a relaxation gap.
10. **Strict benefit from randomized occupancies.** A valid three-action MNAR construction has negative robust improvement for every deterministic policy but a positive certificate for the LP's randomized policy.
11. **Exact ambiguity-regret tradeoff.** For a sharp improvement interval crossing zero, the minimax randomized and deterministic regrets are explicit; uniform safety restricts deployment probability.
12. **Uniform finite-sample outer set.** Three one-sided exact binomial statements per cell yield a reward rectangle valid simultaneously for all policies.
13. **Post-selection safety.** Optimizing over the same simultaneous set preserves the baseline-improvement guarantee.
14. **Robust-objective regret and explicit rate.** The objective gap is at most `2 H rho`; clipped Hoeffding bounds make the count and `Gamma` dependence explicit.
15. **Three-part deployment-margin decomposition.** A sufficient deployment condition cleanly separates the population identification margin, reward-estimation error, and transition-estimation error.
16. **Estimated-transition extension.** A simulation lemma and simultaneous multinomial L1 radii yield an additive improvement penalty.

## Proof-audit notes

- The ambiguity-regret calculation is deliberately positioned as a sequential contrastive specialization of classical minimax-regret decision theory for missing outcomes, not as the invention of minimax regret.
- The finite-sample mapping needs only a lower confidence limit for `q` and lower/upper limits for `p`; allocating probability to the unused upper tail of `q` is unnecessary.
- Unsupported reward cells receive `[0,1]`, and unseen transition rows receive L1 radius two. This preserves validity while correctly destroying power.
- The base model assumes `(R,M)` is conditionally independent of the realized next state given `(S,A)`. Transition-conditioned rewards follow by treating `(h,s,a,s')` as the cell.
- Randomized Markov policies are necessary in the optimization domain because the robust objective is concave and piecewise linear in occupancy.
- The power corollary is intentionally sufficient rather than necessary: it upper-bounds two model-comparison errors and prioritizes interpretability.

## High-value extensions after submission

- Intersect the sensitivity set with approximate future-state bridge restrictions and prove monotone tightening.
- Study nonrectangular shared recording mechanisms across states and time.
- Develop function-approximation confidence sets while keeping extrapolation error separate from MNAR identification error.
- Derive sharper transition confidence sets than a worst-row Weissman radius.
- Add a domain-calibration case study for choosing scientifically plausible `Gamma` ranges.
