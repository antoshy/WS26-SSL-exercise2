from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ssl_strategy_api.training.avoidance_env import AvoidanceEnv

RUN_NAME = "sac"
FAILURE_PENALTY = 5.0
TOTAL_TIMESTEPS = 3_000_000
MODEL_PATH = Path(__file__).parents[1] / "strategies" / RUN_NAME
ENV_CONFIGS = [
    dict(n_obstacles=0, obstacle_speed=0, difficulty=1),
    dict(n_obstacles=3, obstacle_speed=0, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=0, difficulty=5),
    dict(n_obstacles=5, obstacle_speed=0.5, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=1.0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=1.0, difficulty=5),
    dict(n_obstacles=10, obstacle_speed=1.0, difficulty=3),
    dict(n_obstacles=1, obstacle_speed=0.5, difficulty=1),
    dict(n_obstacles=2, obstacle_speed=0, difficulty=4),
    dict(n_obstacles=4, obstacle_speed=0, difficulty=5),
    dict(n_obstacles=5, obstacle_speed=0.5, difficulty=2),
    dict(n_obstacles=5, obstacle_speed=1.0, difficulty=4),
    dict(n_obstacles=7, obstacle_speed=0.5, difficulty=4),
    dict(n_obstacles=10, obstacle_speed=1.0, difficulty=5),
    dict(n_obstacles=10, obstacle_speed=2.0, difficulty=3),
]

def make_env(config):
    def _init():
        return AvoidanceEnv(**config, failure_penalty=FAILURE_PENALTY, action_repeat=3)
    return _init

def main():
    """build 16 copies and lets them run after each other in the same process"""

    vec_env = VecMonitor(DummyVecEnv([make_env(c) for c in ENV_CONFIGS]))

    model = SAC(
        "MlpPolicy",
        vec_env,
        buffer_size=400_000,
        learning_starts=5000,
        batch_size=512,
        train_freq=1,
        gradient_steps=8,
        device="cpu",
        tensorboard_log="runs",
        verbose=1
    )

    # every 100k steps it saves a checkpoint, if it get's worse after
    # save_freq counts steps per copy, thats why it's divided by 16
    checkpoints = CheckpointCallback(
        save_freq=100_000 // len(ENV_CONFIGS),
        save_path=f"checkpoints/{RUN_NAME}",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoints, tb_log_name=RUN_NAME)
    model.save(MODEL_PATH)
    vec_env.close()

if __name__ == "__main__":
    main()