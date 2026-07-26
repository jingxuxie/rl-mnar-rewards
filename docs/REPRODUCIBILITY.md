# Reproducibility record

## Reference environment

The frozen full run was executed on:

- Linux 6.12, x86-64
- Intel Xeon Platinum 8370C, five visible vCPUs
- 5.9 GiB memory
- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- pandas 2.2.3
- Matplotlib 3.10.8

No GPU was used. Wall-clock time for the complete experiment command was 28 seconds; peak resident memory was approximately 276 MiB.

## Installation

```bash
python -m pip install -e '.[dev]'
```

The project requires Python 3.10 or newer. The runtime dependencies are declared in `pyproject.toml`.

## Unit tests

```bash
pytest
```

The frozen release contains 14 tests covering:

- collapse at `Gamma=1`;
- odds-ratio saturation at both binary endpoints;
- monotone interval expansion;
- equality of binary and finite-support LP bounds;
- one-sided outer confidence mapping;
- point-baseline and general contrastive minimax formulas;
- sharp improvement endpoint constructions;
- cancellation identity and interval width;
- occupancy-LP validity;
- Weissman radii and transition penalties.

## Full experiment command

```bash
python experiments/run_all.py \
  --random-mdps 200 \
  --replicates 1000 \
  --post-selection-replicates 3000 \
  --gamma-mdps 100 \
  --workers 4
```

`--workers` changes only parallel scheduling in the sensitivity-misspecification grid; all random seeds are assigned from instance identifiers, so the output is independent of worker count.

## Seeds and data generation

Each experiment uses explicit deterministic seeds in `experiments/run_all.py`:

- analytic sensitivity, cancellation, and minimax curves are deterministic;
- finite-sample bandit seeds are functions of instance, sample size, and repetition;
- post-selection uses one fixed generator per library size;
- transition seeds are functions of null/alternative, sample size, and repetition;
- random MDP and baseline-quality studies use the MDP index as seed;
- sensitivity-misspecification seeds include the true `Gamma` and MDP index.

No simulator return is used to tune an evaluated policy. Latent rewards are used only to construct controlled data-generating mechanisms and to calculate final ground-truth evaluation metrics.

## Metrics

- **True improvement:** exact candidate value minus exact baseline value.
- **Unsafe deployment:** a nonbaseline policy is deployed and its true improvement is negative.
- **Deployment rate / power:** fraction of repetitions with a positive certificate.
- **False-improvement rate:** strict improvement declared when the true candidate improvement is nonpositive.
- **Robust certificate:** sharp worst-case candidate-minus-baseline value over the compatible reward rectangle, with transition penalty when applicable.
- **Standard error:** sample standard error for continuous improvements or binomial standard error for rates.

## Outputs

The command regenerates:

- `results/sensitivity_curve.csv`
- `results/cancellation_curve.csv`
- `results/minimax_ambiguity.csv`
- `results/finite_sample.csv` and `finite_sample_plot.csv`
- `results/post_selection.csv` and `post_selection_plot.csv`
- `results/transition_uncertainty.csv` and `transition_uncertainty_plot.csv`
- `results/random_mdp.csv` and `random_mdp_summary.csv`
- `results/baseline_quality.csv` and `baseline_quality_summary.csv`
- `results/gamma_misspecification.csv` and `gamma_misspecification_summary.csv`
- `results/summary.json`
- corresponding PDF and PNG files under `figures/`.

## Manuscript build

Local development build:

```bash
cd paper
bash build.sh
```

Anonymous-review build with author-kit files present:

```bash
cd paper
AAAI_MODE=review bash build.sh
cd ..
python scripts/check_submission.py
```

The checklist is a separate PDF and is not appended to the nine-page main manuscript.
