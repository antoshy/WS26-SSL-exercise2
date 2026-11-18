import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.helpers import SCENARIOS, TerminationReason
from ssl_strategy_api.run_demos.run_rsoccer_exercise2 import (
    SIMULATION_STEP_SECONDS,
    Exercise2Env,
)
import ssl_strategy_api.strategies.conflict_avoidance as ca
from ssl_strategy_api.strategies.conflict_avoidance import ConflictAvoidanceStrategy
from ssl_strategy_api.strategies.scripted_obstacle import ScriptedObstacleStrategy

RESULTS_DIR = Path("evaluation_history")
N_SEEDS = 50
MAX_STEPS = 1200

MARGIN_RUNS = [
    ("1 margin 0.01m", 0.01),
    ("2 margin 0.08m", 0.08),
    ("3 margin 0.20m", 0.20),
]

FIXED_SCENARIOS = ["basic", "static", "dynamic"]
OBSTACLE_COUNTS = [0, 1, 3, 5, 8, 10]
OBSTACLE_SPEEDS = [0.0, 0.5, 1.0]
DIFFICULTIES = [1, 3, 5]
ENV_CONFIGS = [dict(scenario=name) for name in FIXED_SCENARIOS] + [
    dict(n_obstacles=n, obstacle_speed=speed, difficulty=level)
    for n in OBSTACLE_COUNTS
    for speed in OBSTACLE_SPEEDS
    for level in DIFFICULTIES
    if not (n == 0 and speed > 0)
]

Z = 1.96  # 95% confidence


def wilson(successes: int, n: int):
    p = successes / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return p, centre - half, centre + half


def make_inner_env(config: dict) -> Exercise2Env:
    scenario = config.get("scenario")
    if scenario is not None:
        fixed = SCENARIOS[scenario]
        return Exercise2Env(
            obstacles=fixed.obstacles,
            target=fixed.target,
            blue_start=fixed.blue_start,
            render_mode=None,
        )
    return Exercise2Env(
        n_random_obstacles=config["n_obstacles"],
        max_obstacle_speed_mps=config["obstacle_speed"],
        difficulty=config["difficulty"],
        render_mode=None,
    )


def run_evaluation(label: str, margin_m: float):
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"{label}.csv"

    # Only skip if the file already exists AND contains the avg_time_s column
    if results_file.exists():
        with open(results_file, "r") as f:
            if "avg_time_s" in f.readline():
                print(f"Skipping {label} (already exists with time data)")
                return

    ca.OBSTACLE_MARGIN_M = margin_m
    blue_strategy = ConflictAvoidanceStrategy()
    yellow_strategy = ScriptedObstacleStrategy()

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "config", "episodes", "successes", "avg_time_s"])

        for config in ENV_CONFIGS:
            env = make_inner_env(config)
            successes = 0
            success_times = []

            for seed in range(N_SEEDS):
                state, _ = env.reset(seed=seed)
                blue_strategy.reset(seed=seed)
                blue_strategy.set_target(*env.target)
                yellow_strategy.set_obstacles(env.obstacles)
                yellow_strategy.reset(seed=seed)

                steps = 0
                done = False
                while not done and steps < MAX_STEPS:
                    steps += 1
                    blue_intent = blue_strategy.decide(state, env.blue_config)
                    yellow_intent = yellow_strategy.decide(state, env.yellow_config)

                    commands = to_rsoccer_commands(state, env.blue_config, blue_intent)
                    commands += to_rsoccer_commands(state, env.yellow_config, yellow_intent)
                    state, _, done, _, _ = env.step(commands)

                if env.termination_reason == TerminationReason.TARGET_REACHED:
                    successes += 1
                    success_times.append(steps * SIMULATION_STEP_SECONDS)

            env.close()
            avg_time = float(np.mean(success_times)) if success_times else 30.0
            writer.writerow([label, str(config), N_SEEDS, successes, f"{avg_time:.4f}"])
            print(
                f"{label} | {config} -> {successes}/{N_SEEDS} | avg_time={avg_time:.2f}s",
                flush=True,
            )


def plot_results():
    rows = defaultdict(list)
    allowed_labels = {label for label, _ in MARGIN_RUNS}

    for results_file in sorted(RESULTS_DIR.glob("*.csv")):
        with open(results_file) as f:
            for row in csv.DictReader(f):
                if (
                    row["model"] not in allowed_labels
                    or "scenario" in row["config"]
                    or "avg_time_s" not in row
                ):
                    continue
                rows[row["model"]].append(
                    (int(row["episodes"]), int(row["successes"]), float(row["avg_time_s"]))
                )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    rng = np.random.default_rng(0)
    labels = [label for label, _ in MARGIN_RUNS if label in rows]
    x_labels = [l.split(" ", 1)[1] for l in labels]

    for i, label in enumerate(labels):
        results = rows[label]

        # Left Panel: Success Rate [%]
        rates = [100 * s / n for n, s, _ in results]
        ax1.scatter(i + rng.uniform(-0.12, 0.12, len(rates)), rates, s=14, color="0.75", zorder=1)

        n_total = sum(n for n, _, _ in results)
        s_total = sum(s for _, s, _ in results)
        p, low, high = wilson(s_total, n_total)
        ax1.errorbar(
            i,
            100 * p,
            yerr=[[100 * (p - low)], [100 * (high - p)]],
            fmt="o",
            color="C0",
            markersize=8,
            capsize=6,
            linewidth=2,
            zorder=2,
        )
        ax1.annotate(f"{100 * p:.1f} %", (i + 0.15, 100 * p), va="center", fontweight="bold")

        # Right Panel: Average Time to Target [s]
        times = [t for _, _, t in results]
        ax2.scatter(i + rng.uniform(-0.12, 0.12, len(times)), times, s=14, color="0.75", zorder=1)

        mean_t = float(np.mean(times))
        sem_t = Z * float(np.std(times, ddof=1)) / sqrt(len(times))
        ax2.errorbar(
            i,
            mean_t,
            yerr=sem_t,
            fmt="s",
            color="C1",
            markersize=8,
            capsize=6,
            linewidth=2,
            zorder=2,
        )
        ax2.annotate(f"{mean_t:.2f} s", (i + 0.15, mean_t), va="center", fontweight="bold")

    ax1.set_xticks(range(len(labels)), x_labels)
    ax1.set_ylabel("Erfolgsquote [%]")
    ax1.set_ylim(75, 102)
    ax1.set_xlim(-0.5, len(labels) - 0.2)
    ax1.set_title("Erfolgsquote (95-%-CI)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xticks(range(len(labels)), x_labels)
    ax2.set_ylabel("Ø Zeit bis zum Ziel [s]")
    ax2.set_xlim(-0.5, len(labels) - 0.2)
    ax2.set_title("Durchschnittliche Episodendauer (Erfolge)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("DWA Parameterstudie: Sicherheit vs. Geschwindigkeit", fontweight="bold")
    fig.tight_layout()
    fig.savefig("history.png", dpi=200)
    print("Saved 2-panel plot to: history.png")
    plt.show()


if __name__ == "__main__":
    for label, margin in MARGIN_RUNS:
        run_evaluation(label, margin)
    plot_results()