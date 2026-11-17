from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ssl_strategy_api.training.avoidance_env import AvoidanceEnv

RUN_NAME = "ppo_best_avoidance_finetuned_smaller_lr"
TOTAL_TIMESTEPS = 2_000_000
FAILURE_PENALTY = 10.0
ENEMY_ZONE_PENALTY = 0.0
CRASH_PENALTY = 10
ACTION_REPEAT = 4
MODEL_PATH = Path(__file__).parents[1] / "strategies" / RUN_NAME
FINETUNE_FROM = "ppo_best_avoidance_finetuned"   # model to continue from, None = start from scratch
STRATEGY_DIR = Path(__file__).parents[1] / "strategies"

""" ENV_CONFIGS = [
    dict(n_obstacles=0, obstacle_speed=0, difficulty=1),
    dict(n_obstacles=3, obstacle_speed=0, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=0, difficulty=5),
    dict(n_obstacles=5, obstacle_speed=0.5, difficulty=3),
    dict(n_obstacles=5, obstacle_speed=1.0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=0, difficulty=4),
    dict(n_obstacles=8, obstacle_speed=1.0, difficulty=5),
    dict(n_obstacles=10, obstacle_speed=1.0, difficulty=3)
] """
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
    """build 8 copies and lets them run after each other in the same process"""

    vec_env = VecMonitor(DummyVecEnv([make_env(c) for c in ENV_CONFIGS]))
    
    if FINETUNE_FROM is None:
        model = PPO(
            "MlpPolicy",            # actor and critic as own nets with 2 layers with 64 nodes each
            vec_env,                # 41 inputs, 2 outputs, 8 copies
            ### sampling ###
            n_steps=256,            # 256 steps per copy, in total 2048 steps in the buffer
            ### evaluating ###
            gamma=0.98,             # how much later rewards count
            gae_lambda=0.95,        # how much a surprise flows back into prior steps
            ### learning ###
            n_epochs=10,            # rounds through the buffer
            batch_size=64,          # rows per minibatch, 2048/64=32 gradient-steps per epoch
            clip_range=0.2,         # epsilon, the border for the new/old ratio
            learning_rate=1e-4,     # step-size of the optimizer
            ent_coef=0.0,           # no entropy bonus
            target_kl=0.03,         # stops the epoch once the policy changed this much
            ### others ###
            tensorboard_log="runs", # name of the folder with the plots
            seed=0,                 # makes the run replicable
            device="cpu",
            verbose=1,
        )

    else:
        # load the trained model and attach the new environments
        # all PPO settings (n_steps, gamma, ...) come from the saved file
        model = PPO.load(STRATEGY_DIR / FINETUNE_FROM, env=vec_env,
                         device="cpu", tensorboard_log="runs", custom_objects={"learning_rate": 1e-4})
    
    # every 50k steps it saves a checkpoint, if it get's worse after
    # save_freq counts steps per copy, thats why it's divided by 8
    checkpoints = CheckpointCallback(
        save_freq=50_000 // len(ENV_CONFIGS),
        save_path=f"checkpoints/{RUN_NAME}",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoints, tb_log_name=RUN_NAME,
                reset_num_timesteps=FINETUNE_FROM is None)   # when fine-tuning, keep counting steps)
    model.save(MODEL_PATH)
    vec_env.close()
        
        

if __name__ == "__main__":
    main()