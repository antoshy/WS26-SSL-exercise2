import math
from typing import Mapping, Optional, Tuple

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState

from pathlib import Path
from stable_baselines3 import SAC
from ssl_strategy_api.strategies.avoidance_features import build_obs, action_to_intent

MODEL_FILE = "sac_dir_pen5.zip"

class ConflictAvoidanceStrategy:
    """Implement your conflict avoidance strategy here for exercise 2.

    Drive the blue robot to ``self.target`` without touching any yellow robot,
    leaving the field, or entering the opponent goal area.

    The default below drives straight at the target and ignores the obstacles,
    so it will collide. Replace it with your own conflict avoidance.
    """

    def __init__(self) -> None:
        self.target: Optional[Tuple[float, float]] = None

        # load actor and critic weights once
        self.model = SAC.load(Path(__file__).with_name(MODEL_FILE), device="cpu")
        self.calls = 0
        self.action = None

    def set_target(self, x_m: float, y_m: float) -> None:
        """Called by the runner with the position the blue robot must reach."""
        self.target = (x_m, y_m)

    def reset(self, seed=None) -> None:
        self.calls = 0

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}
        robot_id = config.controlled_robot_ids[0]
        robot = state.robots_blue.get(robot_id)
        if robot is None or self.target is None:
            return {}

 
        obs = build_obs(state, self.target, robot_id)
        self.action, _ = self.model.predict(obs, deterministic=True)
        self.calls += 1
        return {robot_id: action_to_intent(self.action)}
        