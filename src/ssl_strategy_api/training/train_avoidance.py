from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ssl_strategy_api.training.avoidance_env import AvoidanceEnv

RUN_NAME = "sac_dir_pen5"  # name of the folder in strategies/ and tensorboard/
TOTAL_TIMESTEPS = 500_000
FAILURE_PENALTY = 10.0
ENEMY_ZONE_PENALTY = 0.0
CRASH_PENALTY = 10
ACTION_REPEAT = 4
MODEL_PATH = Path(__file__).parents[1] / "strategies" / RUN_NAME
STRATEGY_DIR = Path(__file__).parents[1] / "strategies"

ENV_CONFIGS = [
    dict(scenario="basic"),
    dict(scenario="static"),
    dict(scenario="dynamic"),
    dict(n_obstacles=3, obstacle_speed=0.0, difficulty=5),
    dict(n_obstacles=5, obstacle_speed=0.0, difficulty=5),
    dict(n_obstacles=8, obstacle_speed=0.0, difficulty=5),
    dict(n_obstacles=3, obstacle_speed=1.0, difficulty=4),
    dict(n_obstacles=5, obstacle_speed=1.0, difficulty=4)
]

def make_env(config):
    def _init():
        return AvoidanceEnv(**config, failure_penalty=FAILURE_PENALTY,
                                        enemy_zone_penalty=ENEMY_ZONE_PENALTY,
                                        crash_penalty=CRASH_PENALTY,
                                        action_repeat=ACTION_REPEAT
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
        buffer_size=300_000, #increase buffer size to 300k to avoid early overwriting of experiences
        learning_starts=20_000, #learning starts after 20.000 steps before exploration
        batch_size=256,
        gamma=0.98, #discount factor
        tau=0.005, # soft update coefficient
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
