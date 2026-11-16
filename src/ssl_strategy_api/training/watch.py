from pathlib import Path

from stable_baselines3 import PPO, SAC

from ssl_strategy_api.training.avoidance_env import AvoidanceEnv

#MODEL_PATH = Path(__file__).parents[1] / "strategies" / "avoidance_ppo"
EPISODES = 20

MODEL_PATH = Path("checkpoints/sac_dir_pen5/rl_model_300000_steps")
env = AvoidanceEnv(n_obstacles=0, obstacle_speed=0.0, difficulty=3, render_mode="human")

def main():
    # das trainierte Modell laden, also die Gewichte von Actor und Critic
    #model = PPO.load(MODEL_PATH, device="cpu")
    model = SAC.load(MODEL_PATH, device="cpu")
    # ein einzelnes Env mit Fenster
    #env = AvoidanceEnv(n_obstacles=3, obstacle_speed=0.0, difficulty=3, render_mode="human")

    for episode in range(EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            # der Actor wählt die Aktion, deterministic=True nimmt den Gipfel ohne Würfeln
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(episode, env.inner.termination_reason, env.steps)
    env.close()


if __name__ == "__main__":
    main()