from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ssl_strategy_api.training.avoidance_env import AvoidanceEnv

RUN_NAME = "sac_dir_pen5"
FAILURE_PENALTY = 5.0
GOAL_AREA_PENALTY = 25.0
TOTAL_TIMESTEPS = 500_000
MODEL_PATH = Path(__file__).parents[1] / "strategies" / RUN_NAME
ENV_CONFIGS = [
    dict(n_obstacles=0, obstacle_speed=0, difficulty=1),
    dict(n_obstacles=3, obstacle_speed=0, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=0, difficulty=5),
    dict(n_obstacles=5, obstacle_speed=0.5, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=1.0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=1.0, difficulty=5),
    dict(n_obstacles=10, obstacle_speed=1.0, difficulty=3)
]

def make_env(config):
    def _init():
        return AvoidanceEnv(
            **config,
            failure_penalty=FAILURE_PENALTY,
            goal_area_penalty=GOAL_AREA_PENALTY,
        )
    return _init

def main():
    """Build the weighted curriculum environments in the same process."""

    vec_env = VecMonitor(DummyVecEnv([make_env(c) for c in ENV_CONFIGS]))

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=f"tensorboard/{RUN_NAME}",
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=20_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                qf=[256, 256]))
    )

    # every 50k steps it saves a checkpoint, if it get's worse after
    # save_freq counts steps per copy, hence the division by the number of envs.
    checkpoints = CheckpointCallback(
        save_freq=200_000 // len(ENV_CONFIGS),
        save_path=f"checkpoints/{RUN_NAME}",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoints, tb_log_name=RUN_NAME)
    model.save(MODEL_PATH)
    vec_env.close()

if __name__ == "__main__":
    main()
