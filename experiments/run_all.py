#!/usr/bin/env python3
"""Run all lightweight validation experiments used by the paper.

Every reported policy value is computed exactly by finite-horizon dynamic
programming.  The experiments are deliberately small: they validate the
identification and safety claims rather than benchmark neural architectures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnar_rl import (  # noqa: E402
    TabularMDP,
    ambiguous_bandit_minimax_regret,
    contrast_interval_minimax_regret,
    binary_missing_success_bounds,
    binary_reward_confidence_bounds,
    binary_reward_mean_bounds,
    cancellation_gain,
    clopper_pearson_lower,
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
    transition_improvement_penalty,
    weissman_l1_radius,
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def binomial_standard_error(rate: float, repetitions: int) -> float:
    return float(np.sqrt(max(rate * (1.0 - rate), 0.0) / repetitions))


def deterministic_policy(mdp: TabularMDP, action_by_time: list[int]) -> np.ndarray:
    policy = np.zeros((mdp.horizon, mdp.n_states, mdp.n_actions))
    for h, action in enumerate(action_by_time):
        policy[h, :, action] = 1.0
    return policy


def run_sensitivity_curve() -> pd.DataFrame:
    gamma_grid = np.linspace(1.0, 5.0, 401)
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

    # The exact crossing in this symmetric example is Gamma=9/4.
    critical_gamma = 9.0 / 4.0
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(frame["gamma"], frame["improvement_certificate"], label="Sharp worst-case improvement")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.axvline(critical_gamma, linewidth=1.0, linestyle=":", label=rf"Critical $\Gamma={critical_gamma:.2f}$")
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
        for state in range(states):
            next_state = min(h + 1, states - 1)
            transition[h, state, :, next_state] = 1.0
    initial = np.zeros(states)
    initial[0] = 1.0
    return TabularMDP(transition, initial)


def run_cancellation_curve(gamma: float = 2.0) -> pd.DataFrame:
    mdp = build_shared_prefix_mdp()
    baseline = deterministic_policy(mdp, [0, 0, 0, 0])
    candidate = deterministic_policy(mdp, [0, 0, 0, 1])
    baseline_occupancy = policy_occupancy(mdp, baseline)
    candidate_occupancy = policy_occupancy(mdp, candidate)

    rows = []
    for q_shared in np.linspace(0.02, 1.0, 51):
        q = np.ones((mdp.horizon, mdp.n_states, mdp.n_actions))
        p = np.full_like(q, 0.05)
        for h in range(3):
            q[h, h, 0] = q_shared
            p[h, h, 0] = 0.5
        q[3, 3, 0] = 0.8
        p[3, 3, 0] = 0.4
        q[3, 3, 1] = 0.8
        p[3, 3, 1] = 0.6

        lower, upper = binary_reward_mean_bounds(q, p, gamma)
        direct = sharp_improvement_lower_bound(
            candidate_occupancy,
            baseline_occupancy,
            lower,
            upper,
        )
        separate = policy_value(mdp, lower, candidate) - policy_value(
            mdp,
            upper,
            baseline,
        )
        rows.append(
            {
                "shared_observation_rate": q_shared,
                "direct_contrastive_bound": direct,
                "separate_value_bound": separate,
                "cancellation_gain": cancellation_gain(
                    candidate_occupancy,
                    baseline_occupancy,
                    lower,
                    upper,
                ),
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


def run_minimax_ambiguity() -> pd.DataFrame:
    """Visualize the infinite-data lower bound induced by ambiguity alone."""
    rows = []
    for width in np.linspace(0.02, 0.80, 40):
        lower = -width / 2.0
        upper = width / 2.0
        result = contrast_interval_minimax_regret(lower, upper)
        rows.append(
            {
                "identified_width": width,
                "deterministic_minimax_regret": result.deterministic_regret,
                "randomized_minimax_regret": result.randomized_regret,
                "optimal_candidate_probability": result.candidate_probability,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "minimax_ambiguity.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(frame["identified_width"], frame["deterministic_minimax_regret"], label="Deterministic selector")
    ax.plot(frame["identified_width"], frame["randomized_minimax_regret"], label="Randomized selector")
    ax.set_xlabel("Width of sharp improvement interval")
    ax.set_ylabel("Infinite-data minimax regret")
    ax.set_title("Partial identification creates irreducible decision regret")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "minimax_ambiguity")
    return frame


def bandit_counts(
    rng: np.random.Generator,
    n: int,
    behavior: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = rng.multinomial(n, behavior)
    observed = np.asarray([rng.binomial(total[action], q[action]) for action in range(len(q))])
    success = np.asarray([rng.binomial(observed[action], p[action]) for action in range(len(q))])
    return total.reshape(1, 1, -1), observed.reshape(1, 1, -1), success.reshape(1, 1, -1)


def run_finite_sample(
    gamma: float = 2.0,
    delta: float = 0.05,
    replicates: int = 1000,
) -> pd.DataFrame:
    missing_lower, missing_upper = binary_missing_success_bounds(np.array([0.4, 0.6]), gamma)
    instances = {
        "adversarial_null": {
            "q": np.array([0.3, 0.3]),
            "p": np.array([0.4, 0.6]),
            "nu": np.array([missing_upper[0], missing_lower[1]]),
        },
        "robust_alternative": {
            "q": np.array([0.8, 0.8]),
            "p": np.array([0.35, 0.65]),
            "nu": np.array([0.35, 0.65]),
        },
    }
    sample_sizes = [100, 300, 1000, 3000, 10000]
    behavior = np.array([0.5, 0.5])
    rows = []

    for instance_index, (name, model) in enumerate(instances.items()):
        true_reward = model["q"] * model["p"] + (1.0 - model["q"]) * model["nu"]
        true_improvement = float(true_reward[1] - true_reward[0])
        for n in sample_sizes:
            robust_deploy = 0
            robust_unsafe = 0
            plugin_deploy = 0
            plugin_unsafe = 0
            positive_certificates: list[float] = []
            for replicate in range(replicates):
                rng = np.random.default_rng(10_000 * instance_index + 37 * n + replicate)
                total, observed, success = bandit_counts(
                    rng,
                    n,
                    behavior,
                    model["q"],
                    model["p"],
                )
                intervals = binary_reward_confidence_bounds(total, observed, success, gamma, delta)
                certificate = float(
                    intervals.reward_lower[0, 0, 1]
                    - intervals.reward_upper[0, 0, 0]
                )
                deploy = certificate > 0.0
                robust_deploy += int(deploy)
                robust_unsafe += int(deploy and true_improvement < 0.0)
                positive_certificates.append(max(certificate, 0.0))

                p_hat = np.divide(
                    success.reshape(-1),
                    observed.reshape(-1),
                    out=np.full(2, 0.5),
                    where=observed.reshape(-1) > 0,
                )
                plugin = bool(p_hat[1] > p_hat[0])
                plugin_deploy += int(plugin)
                plugin_unsafe += int(plugin and true_improvement < 0.0)

            for method, deployments, unsafe, median in [
                (
                    "simultaneous_robust_certificate",
                    robust_deploy,
                    robust_unsafe,
                    float(np.median(positive_certificates)),
                ),
                ("complete_case_plugin", plugin_deploy, plugin_unsafe, np.nan),
            ]:
                deployment_rate = deployments / replicates
                unsafe_rate = unsafe / replicates
                rows.append(
                    {
                        "instance": name,
                        "n": n,
                        "method": method,
                        "deploy_rate": deployment_rate,
                        "deploy_se": binomial_standard_error(deployment_rate, replicates),
                        "unsafe_rate": unsafe_rate,
                        "unsafe_se": binomial_standard_error(unsafe_rate, replicates),
                        "median_positive_certificate": median,
                        "true_candidate_improvement": true_improvement,
                        "replicates": replicates,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "finite_sample.csv", index=False)

    robust_alt = frame[(frame.instance == "robust_alternative") & (frame.method == "simultaneous_robust_certificate")]
    robust_null = frame[(frame.instance == "adversarial_null") & (frame.method == "simultaneous_robust_certificate")]
    plugin_null = frame[(frame.instance == "adversarial_null") & (frame.method == "complete_case_plugin")]
    plot_frame = pd.DataFrame(
        {
            "n": robust_alt.n.to_numpy(),
            "robust_power": robust_alt.deploy_rate.to_numpy(),
            "robust_power_se": robust_alt.deploy_se.to_numpy(),
            "robust_false_deployment": robust_null.unsafe_rate.to_numpy(),
            "robust_false_deployment_se": robust_null.unsafe_se.to_numpy(),
            "plugin_false_deployment": plugin_null.unsafe_rate.to_numpy(),
            "plugin_false_deployment_se": plugin_null.unsafe_se.to_numpy(),
        }
    )
    plot_frame.to_csv(ROOT / "results" / "finite_sample_plot.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(robust_alt.n, robust_alt.deploy_rate, marker="o", label="Robust power (alternative)")
    ax.plot(robust_null.n, robust_null.unsafe_rate, marker="s", label="Robust false deployment (null)")
    ax.plot(plugin_null.n, plugin_null.unsafe_rate, marker="^", label="Plug-in false deployment (null)")
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


def _vectorized_candidate_lower_bound(
    total: int,
    observed: np.ndarray,
    success: np.ndarray,
    gamma: float,
    tail_alpha: float,
) -> np.ndarray:
    q_lower = np.where(
        observed == 0,
        0.0,
        beta.ppf(tail_alpha, observed, total - observed + 1),
    )
    p_lower = np.where(
        success == 0,
        0.0,
        beta.ppf(tail_alpha, success, observed - success + 1),
    )
    missing_lower, _ = binary_missing_success_bounds(p_lower, gamma)
    return q_lower * p_lower + (1.0 - q_lower) * missing_lower


def run_post_selection(
    gamma: float = 2.0,
    delta: float = 0.05,
    n_per_candidate: int = 1000,
    replicates: int = 3000,
) -> pd.DataFrame:
    """Show why fixed-policy intervals do not survive policy-library search.

    The point-identified baseline has reward 1/2.  Every candidate has the same
    observed law (q=5/12,p=3/5) and its true reward equals its sharp lower
    endpoint, also 1/2.  Thus every strict improvement declaration is false.
    """
    q_candidate = 5.0 / 12.0
    p_candidate = 3.0 / 5.0
    baseline_reward = 0.5
    library_sizes = [1, 5, 10, 25, 50, 100]
    rows = []

    for library_size in library_sizes:
        rng = np.random.default_rng(200_000 + library_size)
        observed = rng.binomial(
            n_per_candidate,
            q_candidate,
            size=(replicates, library_size),
        )
        success = rng.binomial(observed, p_candidate)

        pointwise_lower = _vectorized_candidate_lower_bound(
            n_per_candidate,
            observed,
            success,
            gamma,
            delta / 2.0,
        )
        simultaneous_lower = _vectorized_candidate_lower_bound(
            n_per_candidate,
            observed,
            success,
            gamma,
            delta / (2.0 * library_size),
        )
        complete_case = np.divide(
            success,
            observed,
            out=np.full_like(success, 0.5, dtype=float),
            where=observed > 0,
        )

        methods = {
            "simultaneous": simultaneous_lower.max(axis=1) > baseline_reward,
            "pointwise_then_select": pointwise_lower.max(axis=1) > baseline_reward,
            "complete_case_then_select": complete_case.max(axis=1) > baseline_reward,
        }
        for method, declarations in methods.items():
            rate = float(declarations.mean())
            rows.append(
                {
                    "library_size": library_size,
                    "method": method,
                    "false_improvement_rate": rate,
                    "standard_error": binomial_standard_error(rate, replicates),
                    "n_per_candidate": n_per_candidate,
                    "replicates": replicates,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "post_selection.csv", index=False)
    plot_frame = (
        frame.pivot(index="library_size", columns="method", values="false_improvement_rate")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    plot_frame.to_csv(ROOT / "results" / "post_selection_plot.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for method, label in [
        ("simultaneous", "Simultaneous certificate"),
        ("pointwise_then_select", "Pointwise, then select"),
        ("complete_case_then_select", "Complete case, then select"),
    ]:
        subset = frame[frame.method == method]
        ax.plot(subset.library_size, subset.false_improvement_rate, marker="o", label=label)
    ax.axhline(delta, linewidth=1.0, linestyle=":", label=rf"Target $\delta={delta:.2f}$")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Number of candidate policies")
    ax.set_ylabel("False strict-improvement rate")
    ax.set_title("Uniform validity survives post-selection")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save_figure(fig, "post_selection")
    return frame



def run_transition_uncertainty(
    delta: float = 0.05,
    replicates: int = 1000,
) -> pd.DataFrame:
    """Validate the estimated-transition penalty on a two-step MDP.

    At the first step, the baseline and candidate choose different actions.
    The next state is binary and the second-step reward is one in the good
    state and zero in the bad state. Therefore each policy value equals its
    first-action probability of reaching the good state. We estimate both
    transition rows from a uniform behavior log.
    """
    sample_sizes = [100, 300, 1000, 3000, 10000]
    instances = {
        "transition_null": np.array([0.55, 0.50]),
        "transition_alternative": np.array([0.55, 0.75]),
    }
    behavior = np.array([0.5, 0.5])
    rows: list[dict[str, float | int | str]] = []

    for instance_index, (name, good_probability) in enumerate(instances.items()):
        true_improvement = float(good_probability[1] - good_probability[0])
        for n in sample_sizes:
            adjusted_deployments = 0
            adjusted_unsafe = 0
            plugin_deployments = 0
            plugin_unsafe = 0
            penalties: list[float] = []
            for replicate in range(replicates):
                rng = np.random.default_rng(
                    3_000_000 + 100_000 * instance_index + 31 * n + replicate
                )
                action_counts = rng.multinomial(n, behavior)
                good_counts = np.array(
                    [
                        rng.binomial(action_counts[action], good_probability[action])
                        for action in range(2)
                    ]
                )
                estimate = np.divide(
                    good_counts,
                    action_counts,
                    out=np.full(2, 0.5),
                    where=action_counts > 0,
                )
                row_radii = np.array(
                    [
                        weissman_l1_radius(
                            int(action_counts[action]),
                            n_states=2,
                            alpha=delta / 2.0,
                        )
                        for action in range(2)
                    ]
                )
                penalty = transition_improvement_penalty(
                    np.array([row_radii.max()])
                )
                estimated_improvement = float(estimate[1] - estimate[0])
                adjusted_certificate = estimated_improvement - penalty
                adjusted_deploy = adjusted_certificate > 0.0
                plugin_deploy = estimated_improvement > 0.0

                adjusted_deployments += int(adjusted_deploy)
                adjusted_unsafe += int(adjusted_deploy and true_improvement < 0.0)
                plugin_deployments += int(plugin_deploy)
                plugin_unsafe += int(plugin_deploy and true_improvement < 0.0)
                penalties.append(penalty)

            for method, deployments, unsafe in [
                (
                    "transition_adjusted_certificate",
                    adjusted_deployments,
                    adjusted_unsafe,
                ),
                ("transition_plugin", plugin_deployments, plugin_unsafe),
            ]:
                deployment_rate = deployments / replicates
                unsafe_rate = unsafe / replicates
                rows.append(
                    {
                        "instance": name,
                        "n": n,
                        "method": method,
                        "deploy_rate": deployment_rate,
                        "deploy_se": binomial_standard_error(
                            deployment_rate,
                            replicates,
                        ),
                        "unsafe_rate": unsafe_rate,
                        "unsafe_se": binomial_standard_error(
                            unsafe_rate,
                            replicates,
                        ),
                        "mean_transition_penalty": float(np.mean(penalties)),
                        "true_candidate_improvement": true_improvement,
                        "replicates": replicates,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "transition_uncertainty.csv", index=False)

    adjusted_alt = frame[
        (frame.instance == "transition_alternative")
        & (frame.method == "transition_adjusted_certificate")
    ]
    adjusted_null = frame[
        (frame.instance == "transition_null")
        & (frame.method == "transition_adjusted_certificate")
    ]
    plugin_null = frame[
        (frame.instance == "transition_null")
        & (frame.method == "transition_plugin")
    ]
    plot_frame = pd.DataFrame(
        {
            "n": adjusted_alt.n.to_numpy(),
            "adjusted_power": adjusted_alt.deploy_rate.to_numpy(),
            "adjusted_false_deployment": adjusted_null.unsafe_rate.to_numpy(),
            "plugin_false_deployment": plugin_null.unsafe_rate.to_numpy(),
        }
    )
    plot_frame.to_csv(
        ROOT / "results" / "transition_uncertainty_plot.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(
        adjusted_alt.n,
        adjusted_alt.deploy_rate,
        marker="o",
        label="Adjusted power",
    )
    ax.plot(
        adjusted_null.n,
        adjusted_null.unsafe_rate,
        marker="s",
        label="Adjusted false deployment",
    )
    ax.plot(
        plugin_null.n,
        plugin_null.unsafe_rate,
        marker="^",
        label="Plug-in false deployment",
    )
    ax.axhline(delta, linewidth=1.0, linestyle=":", label=rf"Target $\delta={delta:.2f}$")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Logged transitions")
    ax.set_ylabel("Probability")
    ax.set_title("Transition penalty restores finite-sample safety")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    save_figure(fig, "transition_uncertainty")
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
        p_observed = rng.uniform(0.12, 0.88, size=shape)
        q_observed = rng.uniform(0.10, 0.55, size=shape)
        odds_ratio = np.exp(rng.uniform(-np.log(gamma), np.log(gamma), size=shape))
        p_missing = missing_probability_from_odds_ratio(p_observed, odds_ratio)
        true_reward = q_observed * p_observed + (1.0 - q_observed) * p_missing
        lower, upper = binary_reward_mean_bounds(q_observed, p_observed, gamma)

        oracle, _ = optimal_policy(mdp, true_reward)
        baseline = epsilon_soft(oracle, baseline_exploration)
        baseline_value = policy_value(mdp, true_reward, baseline)

        complete_case, _ = optimal_policy(mdp, p_observed)
        zero_fill, _ = optimal_policy(mdp, q_observed * p_observed)
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
            improvement_se=("true_improvement", lambda values: values.std(ddof=1) / np.sqrt(len(values))),
            median_improvement=("true_improvement", "median"),
            lower_quartile=("true_improvement", lambda values: values.quantile(0.25)),
            upper_quartile=("true_improvement", lambda values: values.quantile(0.75)),
            unsafe_rate=("unsafe", "mean"),
            deployment_rate=("deployed", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    summary["unsafe_se"] = [
        binomial_standard_error(rate, n_instances) for rate in summary.unsafe_rate
    ]
    summary.to_csv(ROOT / "results" / "random_mdp_summary.csv", index=False)
    return frame



def run_baseline_quality(
    n_instances: int = 100,
    gamma: float = 3.0,
) -> pd.DataFrame:
    """Vary the strength of the trusted baseline on paired random MDPs."""
    exploration_grid = [0.05, 0.10, 0.20, 0.40]
    rows = []
    for seed in range(n_instances):
        rng = np.random.default_rng(700_000 + seed)
        mdp = random_mdp(rng, horizon=4, n_states=5, n_actions=3, concentration=0.7)
        shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
        p_observed = rng.uniform(0.12, 0.88, size=shape)
        q_observed = rng.uniform(0.10, 0.55, size=shape)
        odds_ratio = np.exp(rng.uniform(-np.log(gamma), np.log(gamma), size=shape))
        p_missing = missing_probability_from_odds_ratio(p_observed, odds_ratio)
        true_reward = q_observed * p_observed + (1.0 - q_observed) * p_missing
        lower, upper = binary_reward_mean_bounds(q_observed, p_observed, gamma)
        oracle, _ = optimal_policy(mdp, true_reward)
        complete_case, _ = optimal_policy(mdp, p_observed)

        for exploration in exploration_grid:
            baseline = epsilon_soft(oracle, exploration)
            baseline_value = policy_value(mdp, true_reward, baseline)
            direct = optimize_robust_improvement(mdp, baseline, lower, upper)
            direct_policy = direct.policy if direct.certificate > 1e-10 else baseline
            methods = {
                "complete_case": (complete_case, True),
                "direct_safe": (direct_policy, direct.certificate > 1e-10),
            }
            for method, (policy, deployed) in methods.items():
                improvement = policy_value(mdp, true_reward, policy) - baseline_value
                rows.append(
                    {
                        "seed": seed,
                        "baseline_exploration": exploration,
                        "method": method,
                        "true_improvement": improvement,
                        "unsafe": improvement < -1e-9,
                        "deployed": bool(deployed),
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "baseline_quality.csv", index=False)
    summary = (
        frame.groupby(["baseline_exploration", "method"])
        .agg(
            mean_improvement=("true_improvement", "mean"),
            improvement_se=("true_improvement", lambda values: values.std(ddof=1) / np.sqrt(len(values))),
            unsafe_rate=("unsafe", "mean"),
            deployment_rate=("deployed", "mean"),
        )
        .reset_index()
    )
    summary["unsafe_se"] = [
        binomial_standard_error(rate, n_instances) for rate in summary.unsafe_rate
    ]
    summary.to_csv(ROOT / "results" / "baseline_quality_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for method, label in [("direct_safe", "Direct safe"), ("complete_case", "Complete case")]:
        subset = summary[summary.method == method]
        ax.plot(
            subset.baseline_exploration,
            subset.mean_improvement,
            marker="o",
            label=label,
        )
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Baseline exploration probability")
    ax.set_ylabel("Mean true improvement")
    ax.set_title("Performance across baseline quality")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    save_figure(fig, "baseline_quality")
    return frame

def _gamma_misspecification_instance(
    task: tuple[float, int, float, tuple[float, ...]],
) -> list[dict[str, float | int | bool]]:
    gamma_true, seed, baseline_exploration, gamma_grid = task
    rng = np.random.default_rng(1_000_000 + int(100 * gamma_true) + seed)
    mdp = random_mdp(rng, horizon=4, n_states=5, n_actions=3, concentration=0.7)
    shape = (mdp.horizon, mdp.n_states, mdp.n_actions)
    p_observed = rng.uniform(0.12, 0.88, size=shape)
    q_observed = rng.uniform(0.10, 0.55, size=shape)
    if gamma_true == 1.0:
        odds_ratio = np.ones(shape)
    else:
        odds_ratio = np.where(
            rng.random(shape) < 0.5,
            gamma_true,
            1.0 / gamma_true,
        )
    p_missing = missing_probability_from_odds_ratio(p_observed, odds_ratio)
    true_reward = q_observed * p_observed + (1.0 - q_observed) * p_missing
    oracle, _ = optimal_policy(mdp, true_reward)
    baseline = epsilon_soft(oracle, baseline_exploration)
    baseline_value = policy_value(mdp, true_reward, baseline)

    output: list[dict[str, float | int | bool]] = []
    for gamma_assumed in gamma_grid:
        lower, upper = binary_reward_mean_bounds(q_observed, p_observed, gamma_assumed)
        result = optimize_robust_improvement(mdp, baseline, lower, upper)
        deployed = result.certificate > 1e-10
        policy = result.policy if deployed else baseline
        improvement = policy_value(mdp, true_reward, policy) - baseline_value
        output.append(
            {
                "seed": seed,
                "gamma_true": gamma_true,
                "gamma_assumed": gamma_assumed,
                "covered": gamma_assumed >= gamma_true,
                "deployed": deployed,
                "unsafe": improvement < -1e-9,
                "true_improvement": improvement,
                "certificate": result.certificate,
            }
        )
    return output


def run_gamma_misspecification(
    n_instances: int = 100,
    baseline_exploration: float = 0.1,
    workers: int = 1,
) -> pd.DataFrame:
    """Stress test under- and over-specification of the sensitivity class."""
    gamma_grid = (1.0, 1.5, 2.0, 3.0, 5.0)
    tasks = [
        (gamma_true, seed, baseline_exploration, gamma_grid)
        for gamma_true in gamma_grid
        for seed in range(n_instances)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            nested_rows = list(executor.map(_gamma_misspecification_instance, tasks))
    else:
        nested_rows = [_gamma_misspecification_instance(task) for task in tasks]
    rows = [row for group in nested_rows for row in group]

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "gamma_misspecification.csv", index=False)
    summary = (
        frame.groupby(["gamma_true", "gamma_assumed", "covered"])
        .agg(
            unsafe_rate=("unsafe", "mean"),
            deployment_rate=("deployed", "mean"),
            mean_improvement=("true_improvement", "mean"),
            mean_certificate=("certificate", "mean"),
        )
        .reset_index()
    )
    summary["unsafe_se"] = [
        binomial_standard_error(rate, n_instances) for rate in summary.unsafe_rate
    ]
    summary.to_csv(ROOT / "results" / "gamma_misspecification_summary.csv", index=False)

    pivot = summary.pivot(index="gamma_true", columns="gamma_assumed", values="unsafe_rate")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    image = ax.imshow(
        pivot.to_numpy(),
        vmin=0.0,
        vmax=max(0.5, float(pivot.to_numpy().max())),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    ax.set_xlabel(r"Assumed $\Gamma$")
    ax.set_ylabel(r"True $\Gamma$")
    ax.set_title("Unsafe deployment under sensitivity misspecification")
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            ax.text(column, row, f"{pivot.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Unsafe rate")
    save_figure(fig, "gamma_misspecification")
    return frame

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-mdps", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--post-selection-replicates", type=int, default=3000)
    parser.add_argument("--gamma-mdps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    sensitivity = run_sensitivity_curve()
    cancellation = run_cancellation_curve()
    minimax = run_minimax_ambiguity()
    finite_sample = run_finite_sample(replicates=args.replicates)
    post_selection = run_post_selection(replicates=args.post_selection_replicates)
    transition_frame = run_transition_uncertainty(replicates=args.replicates)
    random_frame = run_random_mdps(n_instances=args.random_mdps)
    baseline_frame = run_baseline_quality(n_instances=min(args.random_mdps, 100))
    gamma_frame = run_gamma_misspecification(n_instances=args.gamma_mdps, workers=args.workers)

    random_summary = pd.read_csv(ROOT / "results" / "random_mdp_summary.csv")
    gamma_summary = pd.read_csv(ROOT / "results" / "gamma_misspecification_summary.csv")
    post_at_100 = post_selection[post_selection.library_size == 100].set_index("method")
    metadata = {
        "critical_gamma": 9.0 / 4.0,
        "direct_cancellation_certificate": float(cancellation.direct_contrastive_bound.iloc[0]),
        "separate_cancellation_certificate_at_low_observation": float(cancellation.separate_value_bound.iloc[0]),
        "symmetric_ambiguity_randomized_regret_at_width_0.8": float(minimax.randomized_minimax_regret.iloc[-1]),
        "finite_sample_replicates": args.replicates,
        "post_selection_replicates": args.post_selection_replicates,
        "post_selection_false_rate_k100": {
            method: float(post_at_100.loc[method, "false_improvement_rate"])
            for method in post_at_100.index
        },
        "transition_uncertainty_summary": transition_frame.to_dict(orient="records"),
        "random_mdp_instances": args.random_mdps,
        "random_mdp_summary": random_summary.to_dict(orient="records"),
        "baseline_quality_summary": pd.read_csv(ROOT / "results" / "baseline_quality_summary.csv").to_dict(orient="records"),
        "gamma_mdp_instances_per_cell": args.gamma_mdps,
        "covered_gamma_max_unsafe_rate": float(gamma_summary[gamma_summary.covered].unsafe_rate.max()),
        "underspecified_gamma_max_unsafe_rate": float(gamma_summary[~gamma_summary.covered].unsafe_rate.max()),
    }
    (ROOT / "results" / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
