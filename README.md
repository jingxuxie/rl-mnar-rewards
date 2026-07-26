# Partial Identification and Safe Offline Policy Improvement with MNAR Rewards

This repository studies offline reinforcement learning when states, actions, and transitions are logged but reward observation may depend on the latent reward itself. The project is theory-first and designed to run entirely on a laptop.

## Main results in the current draft

- An impossibility result showing that optimal policies are not point identified without restrictions on reward-dependent missingness.
- Closed-form **sharp** reward intervals for binary rewards under an odds-ratio sensitivity parameter `Gamma`.
- A finite-support extension using small linear programs.
- Sharp policy-value and baseline-relative improvement bounds.
- An exact cancellation identity: uncertainty on occupancy shared by the candidate and baseline does not affect their comparison.
- An occupancy-measure LP that maximizes worst-case improvement over a baseline.
- Simultaneous finite-sample confidence sets, post-selection-safe improvement, and a robust-objective regret bound.
- Lightweight validations on bandits, deterministic chains, and random tabular MDPs.

## Reproduce

```bash
python -m pip install -e '.[dev]'
pytest
python experiments/run_all.py
cd paper && bash build.sh
```

The experiment script regenerates all CSV files and figures with fixed random seeds. The default run evaluates 200 random MDPs and 500 finite-sample repetitions per setting.

## Repository layout

- `src/mnar_rl/`: sensitivity bounds, tabular MDP utilities, data simulation, and robust optimization.
- `experiments/run_all.py`: all experiments and paper figures.
- `tests/`: unit tests for sharp endpoints and LP validity.
- `paper/`: manuscript and bibliography.
- `docs/PROOF_ROADMAP.md`: completed arguments and next theoretical milestones.
- `results/`, `figures/`: generated outputs used by the paper.

## Current empirical snapshot

Across 200 random MNAR MDPs with `Gamma=3`, direct baseline-relative robust optimization had zero harmful deployments and mean improvement `0.049` over a near-optimal baseline. Complete-case optimization was harmful on `26%` of instances; absolute maximin planning was harmful on `32%`; subtracting separate value bounds was safe but never deployed.
