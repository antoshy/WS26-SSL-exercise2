import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from ssl_strategy_api.run_demos.run_rsoccer_exercise2 import (
    BLUE_ROBOT_ID, EPISODE_TIME_LIMIT_SECONDS, SIMULATION_STEP_SECONDS, Exercise2Env,
)
from ssl_strategy_api.strategies.scripted_obstacle import ScriptedObstacleStrategy
from ssl_strategy_api.adapters.rsoccer import to_rsoccer_commands
from ssl_strategy_api.helpers import TerminationReason
from ssl_strategy_api.strategies.avoidance_features import OBS_DIM, build_obs, action_to_intent


class AvoidanceEnv(gym.Env):
    """ translates between SB3 (gymnasium) and Exercise2Env """
    

    def __init__(self, n_obstacles=5, obstacle_speed=0.0, difficulty=1, action_repeat=4, failure_penalty=10.0, render_mode=None):
        self.inner = Exercise2Env(
        n_random_obstacles=n_obstacles,
        max_obstacle_speed_mps=obstacle_speed,
        difficulty=difficulty,
        render_mode=render_mode,
    )
        self.yellow = ScriptedObstacleStrategy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-math.inf, math.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.max_steps = 1200
        self.action_repeat = action_repeat
        self.failure_penalty = failure_penalty

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state, _ = self.inner.reset(seed=seed)
        self.yellow.set_obstacles(self.inner.obstacles)
        self.state = state
        self.steps = 0
        self.prev_distance = self._distance_to_target(state)
        return build_obs(state, self.inner.target), {}

    def _sim_step(self, action):
        self.steps += 1
        blue_intent = {BLUE_ROBOT_ID: action_to_intent(action)}
        yellow_intent = self.yellow.decide(self.state, self.inner.yellow_config)
        commands = to_rsoccer_commands(self.state, self.inner.blue_config, blue_intent)
        commands += (to_rsoccer_commands(self.state, self.inner.yellow_config, yellow_intent))
        state, _, done, _, _ = self.inner.step(commands)
        reason = self.inner.termination_reason
        reward = self._reward(state, reason)
        terminated = done
        truncated = not done and self.steps >= self.max_steps
        info = {"is_success": reason == TerminationReason.TARGET_REACHED}
        self.state = state
        
        return reward, terminated, truncated, reason

    def step(self, action):
        """ one decision of the agent: hold the action for several simulation steps """
        total_reward = 0.0
        for _ in range(self.action_repeat):
            reward, terminated, truncated, reason = self._sim_step(action)
            total_reward += reward
            if terminated or truncated:      # episode over
                break
        info = {"is_success": reason == TerminationReason.TARGET_REACHED}
        return build_obs(self.state, self.inner.target), total_reward, terminated, truncated, info

    def _reward(self, state, reason):
        new_distance = self._distance_to_target(state)
        reward = self.prev_distance - new_distance # progress
        self.prev_distance = new_distance
        if reason == TerminationReason.TARGET_REACHED:
            seconds_left = EPISODE_TIME_LIMIT_SECONDS - self.steps * SIMULATION_STEP_SECONDS
            reward += 10.0 + seconds_left / 3 # target reached
        elif reason is not None:
            reward -= self.failure_penalty # collision, entered goal area, left the field
        return reward

    def _distance_to_target(self, state):
        robot = state.robots_blue[BLUE_ROBOT_ID]
        target = self.inner.target
        return math.hypot(target[0] - robot.x_m, target[1] - robot.y_m)

    def close(self):
        self.inner.close()


if __name__ == "__main__":
      
    from stable_baselines3.common.env_checker import check_env

    env = AvoidanceEnv(n_obstacles=0)
    obs, _ = env.reset(seed=1)

    total = 0.0
    while True:
        robot = env.state.robots_blue[0]
        tx, ty = env.inner.target
        direction = np.array([tx - robot.x_m, ty - robot.y_m])
        action = (direction / np.linalg.norm(direction)).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break

    print(env.inner.termination_reason, env.steps, round(total, 2), info)
    env.close()