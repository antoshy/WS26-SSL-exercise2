from pathlib import Path
from typing import Mapping

from stable_baselines3 import SAC

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState
from ssl_strategy_api.strategies.avoidance_features import OBS_DIM, action_to_intent, build_obs


MODEL_PATH = Path(__file__).with_name("sac_dir_pen5.zip")


class StrikerStrategy:
    """Student placeholder for Exercise 1.

    Replace this strategy with your own implementation.

    The default behavior below is intentionally simple and deliberately not optimal:
    it demonstrates how to access the controlled robot and return a valid mapping.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model = SAC.load(model_path, device="cpu")
        if self.model.observation_space.shape != (OBS_DIM,):
            raise ValueError("SAC model has an incompatible observation space")
        self.target = (0.0, 0.0)

    def set_target(self, x: float, y: float) -> None:
        self.target = (x, y)

    def reset(self, seed=None) -> None:
        pass

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        """Simple starter example for students.

        You must change this strategy to implement your own striker behavior.

        This example shows the basic structure:
        - check whether any robots are controlled
        - select the robot for the blue team
        - get its state from the world
        - return a mapping for that robot id

        In this exercise, the blue team is the striker and uses robot id 0 by default.
        """
        if not config.controlled_robot_ids:
            return {}

        if config.team != TeamColor.BLUE:
            return {}

        robot_id = config.controlled_robot_ids[0]
        robots = state.robots_blue
        robot = robots.get(robot_id)
        if robot is None:
            return {}

        observation = build_obs(state, self.target, robot_id=robot_id)
        action, _ = self.model.predict(observation, deterministic=True)
        return {robot_id: action_to_intent(action)}
