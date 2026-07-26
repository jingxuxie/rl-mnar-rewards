#!/usr/bin/env python3
"""Run all lightweight validation experiments and generate paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnar_rl import (  # noqa: E402
    MNARRewardModel,
    TabularMDP,
    binary_missing_success_bounds,
    binary_reward_confidence_bounds,
    binary_reward_mean_bounds,
    epsilon_soft,
    missing_probability_from_odds_ratio,
    optimal_policy,
    optimize_robust_improvement,
    policy_occupancy,
    policy_value,
    random_mdp,
    robust_absolute_policy,
    separate_value_lower_bound,
    sharp_improvement_lower_bound,
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def deterministic_policy(mdp: TabularMDP, action_by_time: list[int]) -> np.ndarray:
    policy = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions))
    for h, action in enumerate(action_by_time):
        policy[h, :, action] = 1.0
    return policy


def run_sensitivity_curve() -> pd.DataFrame:
    gamma_grid = np.linspace(1.0, 5.0, 101)
    q = np.array([0.5, 0.5])
    p = np.array([0.4, 0.6])
    rows = []
    for gamma in gamma_grid:
        lower, upper = binary_reward_mean_bounds(q, p, gamma)
        rows.append(
            {
                "gamma": gamma,
                "candidate_lower": lower[1],
                "baseline_upper": upper[0],
                "improvement_certificate": lower[1] - upper[0],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "sensitivity_curve.csv", index=False)

    positive = frame[frame["improvement_certificate"] >= -1e-12]
    critical_gamma = float(positive["gamma"].max())

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(frame["gamma"], frame["improvement_certificate"], label="Sharp worst-case improvement")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.axvline(critical_gamma, linewidth=1.0, linestyle=":", label=rf"Critical $\Gamma\approx{critical_gamma:.2f}$")
    ax.set_xlabel(r"Sensitivity parameter $\Gamma$")
    ax.set_ylabel("Certified improvement")
    ax.set_title("Certification weakens smoothly with MNAR severity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "sensitivity_curve")
    return frame


def build_shared_prefix_mdp() -> TabularMDP:
    horizon, states, actions = 4, 4, 2
    transition = np.zeros((horizon, states, actions, states))
    for h in range(horizon):
        for s in range(states):
            next_state = min(h + 1, states - 1)
            transition[h, s, :, next_state] = 1.0
    initial = np.zeros(states)
    initial[0] = 1.0
    return TabularMDP(transition, initial)


def run_cancellation_curve(gamma: float = 2.0) -> pd.DataFrame:
    mdp = build_shared_prefix_mdp()
    baseline = deterministic_policy(mdp, [0, 0, 0, 0])
    candidate = deterministic_policy(mdp, [0, 0, 0, 1])
    db = policy_occupancy(mdp, baseline)
    dc = policy_occupancy(mdp, candidate)

    rows = []
    for q_shared in np.linspace(0.02, 1.0, 51):
        q = np.ones((mdp.horizon, mdp.n_states, mdp.n_actions))
        p = np.full_like(q, 0.05)

        # The baseline and candidate share three highly uncertain reward cells.
        for h in range(3):
            q[h, h, 0] = q_shared
            p[h, h, 0] = 0.5

        # At the final decision, the candidate has a robust observed advantage.
        q[3, 3, 0] = 0.8
        p[3, 3, 0] = 0.4
        q[3, 3, 1] = 0.8
        p[3, 3, 1] = 0.6

        lower, upper = binary_reward_mean_bounds(q, p, gamma)
        direct = sharp_improvement_lower_bound(dc, db, lower, upper)
        separate = policy_value(mdp, lower, candidate) - policy_value(mdp, upper, baseline)
        rows.append(
            {
                "shared_observation_rate": q_shared,
                "direct_contrastive_bound": direct,
                "separate_value_bound": separate,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "cancellation_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(frame["shared_observation_rate"], frame["direct_contrastive_bound"], label="Direct improvement bound")
    ax.plot(frame["shared_observation_rate"], frame["separate_value_bound"], label="Separate value bounds")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Observation rate on shared rewards")
    ax.set_ylabel("Certified improvement")
    ax.set_title("Direct comparison cancels shared reward uncertainty")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "shared_uncertainty_cancellation")
    return frame


def bandit_counts(
    rng: np.random.Generator,
    n: int,
    behavior: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = rng.multinomial(n, behavior)
    observed = np.asarray([rng.binomial(total[a], q[a]) for a in range(len(q))])
    success = np.asarray([rng.binomial(observed[a], p[a]) for a in range(len(q))])
    return total.reshape(1, 1, -1), observed.reshape(1, 1, -1), success.reshape(1, 1, -1)


def run_finite_sample(
    gamma: float = 2.0,
    delta: float = 0.05,
    replicates: int = 500,
) -> pd.DataFrame:
    lower_u, upper_u = binary_missing_success_bounds(np.array([0.4, 0.6]), gamma)
    instances = {
        "adversarial_null": {
            "q": np.array([0.3, 0.3]),
            "p": np.array([0.4, 0.6]),
            "u": np.array([upper_u[0], lower_u[1]]),
        },
        "robust_alternative": {
            "q": np.array([0.8, 0.8]),
            "p": np.array([0.35, 0.65]),
            "u": np.array([0.35, 0.65]),
        },
    }
    sample_sizes = [100, 300, 1000, 3000, 10000]
    behavior = np.array([0.5, 0.5])
    rows = []

    for instance_index, (name, model) in enumerate(instances.items()):
        true_reward = model["q"] * model["p"] + (1.0 - model["q"]) * model["u"]
        true_candidate_improvement = float(true_reward[1] - true_reward[0])
        for n in sample_sizes:
            robust_deploy = 0
            robust_unsafe = 0
            naive_deploy = 0
            naive_unsafe = 0
            positive_certificates: list[float] = []
            for replicate in range(replicates):
                rng = np.random.default_rng(10_000 * instance_index + 37 * n + replicate)
                total, observed, success = bandit_counts(rng, n, behavior, model["q"], model["p"])
                intervals = binary_reward_confidence_bounds(total, observed, success, gamma, delta)
                certificate = float(intervals.reward_lower[0, 0, 1] - intervals.reward_upper[0, 0, 0])
                deploy = certificate > 0.0
                robust_deploy += int(deploy)
                robust_unsafe += int(deploy and true_candidate_improvement < 0.0)
                positive_certificates.append(max(certificate, 0.0))

                p_hat = np.divide(
                    success.reshape(-1),
                    observed.reshape(-1),
                    out=np.full(2, 0.5),
                    where=observed.reshape(-1) > 0,
                )
                naive = bool(p_hat[1] > p_hat[0])
                naive_deploy += int(naive)
                naive_unsafe += int(naive and true_candidate_improvement < 0.0)

            rows.extend(
                [
                    {
                        "instance": name,
                        "n": n,
                        "method": "simultaneous_robust_certificate",
                        "deploy_rate": robust_deploy / replicates,
                        "unsafe_rate": robust_unsafe / replicates,
                        "median_positive_certificate": float(np.median(positive_certificates)),
                        "true_candidate_improvement": true_candidate_improvement,
                    },
                    {
                        "instance": name,
                        "n": n,
                        "method": "complete_case_plugin",
                        "deploy_rate": naive_deploy / replicates,
                        "unsafe_rate": naive_unsafe / replicates,
                        "median_positive_certificate": np.nan,
                        "true_candidate_improvement": true_candidate_improvement,
                    },
                ]
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "finite_sample.csv", index=False)

    robust_alt = frame[(frame.instance == "robust_alternative") & (frame.method == "simultaneous_robust_certificate")]
    robust_null = frame[(frame.instance == "adversarial_null") & (frame.method == "simultaneous_robust_certificate")]
    naive_null = frame[(frame.instance == "adversarial_null") & (frame.method == "complete_case_plugin")]
    plot_frame = pd.DataFrame(
        {
            "n": robust_alt.n.to_numpy(),
            "robust_power": robust_alt.deploy_rate.to_numpy(),
            "robust_false_deployment": robust_null.unsafe_rate.to_numpy(),
            "plugin_false_deployment": naive_null.unsafe_rate.to_numpy(),
        }
    )
    plot_frame.to_csv(ROOT / "results" / "finite_sample_plot.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(robust_alt.n, robust_alt.deploy_rate, marker="o", label="Robust power (alternative)")
    ax.plot(robust_null.n, robust_null.unsafe_rate, marker="s", label="Robust false deployment (null)")
    ax.plot(naive_null.n, naive_null.unsafe_rate, marker="^", label="Plug-in false deployment (null)")
    ax.axhline(delta, linewidth=1.0, linestyle=":", label=rf"Target $\delta={delta:.2f}$")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Number of logged bandit rounds")
    ax.set_ylabel("Probability")
    ax.set_title("Finite-sample safety and power")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save_figure(fig, "finite_sample_safety_power")
    return frame


def run_random_mdps(
    n_instances: int = 200,
    gamma: float = 3.0,
    baseline_exploration: float = 0.1,
) -> pd.DataFrame:
    rows = []
    for seed in range(n_instances):
        rng = np.random.default_rng(seed)
        mdp = random_mdp(rng, horizon=4, n_states=5, n_actions=3, concentration=0.7)
        shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
        p_obs = rng.uniform(0.12, 0.88, size=shape)
        q_obs = rng.uniform(0.10, 0.55, size=shape)
        odds_ratio = np.exp(rng.uniform(-np.log(gamma), np.log(gamma), size=shape))
        p_miss = missing_probability_from_odds_ratio(p_obs, odds_ratio)
        true_reward = q_obs * p_obs + (1.0 - q_obs) * p_miss
        lower, upper = binary_reward_mean_bounds(q_obs, p_obs, gamma)

        oracle, _ = optimal_policy(mdp, true_reward)
        baseline = epsilon_soft(oracle, baseline_exploration)
        baseline_value = policy_value(mdp, true_reward, baseline)

        complete_case, _ = optimal_policy(mdp, p_obs)
        zero_fill, _ = optimal_policy(mdp, q_obs * p_obs)
        absolute, _ = robust_absolute_policy(mdp, lower)
        direct = optimize_robust_improvement(mdp, baseline, lower, upper)
        direct_policy = direct.policy if direct.certificate > 1e-10 else baseline
        separate_certificate = separate_value_lower_bound(mdp, absolute, baseline, lower, upper)
        separate_policy = absolute if separate_certificate > 1e-10 else baseline

        policies = {
            "complete_case": (complete_case, np.nan, True),
            "zero_fill": (zero_fill, np.nan, True),
            "absolute_robust": (absolute, np.nan, True),
            "separate_safe": (separate_policy, separate_certificate, separate_certificate > 1e-10),
            "direct_safe": (direct_policy, direct.certificate, direct.certificate > 1e-10),
        }
        for method, (policy, certificate, deployed) in policies.items():
            improvement = policy_value(mdp, true_reward, policy) - baseline_value
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "true_improvement": improvement,
                    "unsafe": improvement < -1e-9,
                    "deployed": bool(deployed),
                    "certificate": certificate,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "random_mdp.csv", index=False)

    order = ["complete_case", "zero_fill", "absolute_robust", "separate_safe", "direct_safe"]
    labels = ["Complete\ncase", "Zero\nfill", "Absolute\nrobust", "Separate\nsafe", "Direct\nsafe"]
    data = [frame[frame.method == method].true_improvement.to_numpy() for method in order]

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel("True improvement over baseline")
    ax.set_title(f"Population study over {n_instances} random MNAR MDPs")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "random_mdp_improvement")

    summary = (
        frame.groupby("method")
        .agg(
            mean_improvement=("true_improvement", "mean"),
            median_improvement=("true_improvement", "median"),
            unsafe_rate=("unsafe", "mean"),
            deployment_rate=("deployed", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    summary.to_csv(ROOT / "results" / "random_mdp_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(order))
    ax.bar(x, summary.unsafe_rate)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, max(0.4, float(summary.unsafe_rate.max()) * 1.15))
    ax.set_ylabel("Fraction worse than baseline")
    ax.set_title("Only baseline-relative robust optimization is uniformly safe")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "random_mdp_unsafe_rate")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-mdps", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    sensitivity = run_sensitivity_curve()
    cancellation = run_cancellation_curve()
    finite_sample = run_finite_sample(replicates=args.replicates)
    random_frame = run_random_mdps(n_instances=args.random_mdps)

    critical_gamma = float(
        sensitivity.loc[sensitivity.improvement_certificate >= -1e-12, "gamma"].max()
    )
    random_summary = pd.read_csv(ROOT / "results" / "random_mdp_summary.csv")
    metadata = {
        "critical_gamma": critical_gamma,
        "direct_cancellation_certificate": float(cancellation.direct_contrastive_bound.iloc[0]),
        "separate_cancellation_certificate_at_low_observation": float(cancellation.separate_value_bound.iloc[0]),
        "finite_sample_replicates": args.replicates,
        "random_mdp_instances": args.random_mdps,
        "random_mdp_summary": random_summary.to_dict(orient="records"),
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
