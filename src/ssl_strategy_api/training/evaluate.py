from pathlib import Path
from ssl_strategy_api.training.train_avoidance import ENV_CONFIGS
from ssl_strategy_api.training.avoidance_env import AvoidanceEnv
from stable_baselines3 import PPO, SAC
from collections import Counter
import csv

STRATEGY_DIR = Path(__file__).parents[1] / "strategies"
# CHECKPOINT_DIR = Path("checkpoints/ppo_best_avoidance_finetuned_long_harder_cases") # to check best checkpoints
RESULTS_DIR = Path("evaluation_history")
N_SEEDS = 50

# build the test scenarios
FIXED_SCENARIOS = ["basic", "static", "dynamic"]
OBSTACLE_COUNTS = [0, 1, 3, 5, 8, 10]
OBSTACLE_SPEEDS = [0.0, 0.5, 1.0]
DIFFICULTIES = [1, 3, 5]
ENV_CONFIGS = [dict(scenario=name) for name in FIXED_SCENARIOS] + [
    dict(n_obstacles=n, obstacle_speed=speed, difficulty=level)
    for n in OBSTACLE_COUNTS
    for speed in OBSTACLE_SPEEDS
    for level in DIFFICULTIES
    if not (n == 0 and speed > 0)    # without obstacles the speed changes nothing
]


def main(path, label, action_repeat):
    RESULTS_DIR.mkdir(exist_ok=True)
    results_file = RESULTS_DIR / f"{label}.csv"
    with open(results_file, "w", newline="") as f:     # "w": every run starts a fresh file
        writer = csv.writer(f)
        writer.writerow(["model", "config", "episodes", "successes"])
        model = SAC.load(path, device="cpu")
        for config in ENV_CONFIGS:
            env = AvoidanceEnv(**config, action_repeat=action_repeat)
            successes = 0
            for seed in range(N_SEEDS):
                obs, _ = env.reset(seed=seed)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                successes += info["is_success"]
            env.close()
            writer.writerow([label, str(config), N_SEEDS, successes])
            print(label, config, successes, "/", N_SEEDS)


if __name__ == "__main__":
    """ # (model name, action_repeat used in its training)
    models = [
        ("ppo_best_avoidance", 1)
    ]
    for name, action_repeat in models:
        main(name, action_repeat) """
    main(STRATEGY_DIR / "sac_dir_radius_2m.zip", "SAC-radius-2m", 1)