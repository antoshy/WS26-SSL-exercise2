import math
from typing import Mapping, Optional, Tuple

from ssl_strategy_api.models import RobotIntent, TeamColor, TeamConfig, WorldState

ROBOT_RADIUS_M = 0.09
PENALTY_LENGTH_M = 1.0
PENALTY_WIDTH_M = 2.0
BOX_SAFETY_MARGIN_M = ROBOT_RADIUS_M  # 0.09 m (one robot radius)
MAX_SPEED_MPS = 2.0

class ConflictAvoidanceStrategy:

    def __init__(self) -> None:
        self.target: Optional[Tuple[float, float]] = None


    def set_target(self, x_m: float, y_m: float) -> None:
        """Called by the runner with the position the blue robot must reach."""
        self.target = (x_m, y_m)


    def reset(self, seed=None) -> None:
        pass


    def _is_in_penalty_box(self, x_m: float, y_m: float, half_length: float, margin_m: float) -> bool:
        # Check if the point (x_m, y_m) is inside the penalty box with the robot radius margin.
        x_min = half_length - PENALTY_LENGTH_M - margin_m
        y_max = (PENALTY_WIDTH_M / 2.0) + margin_m
        return x_m >= x_min and abs(y_m) <= y_max


    def _segment_crosses_box(
            self, start: Tuple[float, float], end: Tuple[float, float], half_length: float
        ) -> bool:
            # Use 0.09 m margin, unless the target itself spawned closer than 0.09 m to the box
            check_margin = BOX_SAFETY_MARGIN_M
            if self._is_in_penalty_box(end[0], end[1], half_length, BOX_SAFETY_MARGIN_M):
                check_margin = 0.01

            for i in range(1, 20):
                t = i / 20.0
                px = start[0] + t * (end[0] - start[0])
                py = start[1] + t * (end[1] - start[1])
                if self._is_in_penalty_box(px, py, half_length, check_margin):
                    return True
            return False


    def _get_active_waypoint(
            self, robot_pos: Tuple[float, float], half_length: float
        ) -> Tuple[float, float]:
            assert self.target is not None
            if not self._segment_crosses_box(robot_pos, self.target, half_length):
                return self.target

            # Place corner waypoints at 0.09 + 0.08 = 0.17 m outside the box lines
            corner_x = half_length - PENALTY_LENGTH_M - (BOX_SAFETY_MARGIN_M + 0.08)
            corner_y_abs = (PENALTY_WIDTH_M / 2.0) + (BOX_SAFETY_MARGIN_M + 0.08)

            first_y = corner_y_abs if robot_pos[1] >= 0.0 else -corner_y_abs
            second_y = corner_y_abs if self.target[1] >= 0.0 else -corner_y_abs

            if robot_pos[0] > corner_x + 0.05:
                return (corner_x, first_y)

            second_corner = (corner_x, second_y)
            # If we already reached second_corner and the diagonal still clips the margin, go to target
            if math.hypot(robot_pos[0] - second_corner[0], robot_pos[1] - second_corner[1]) < 0.10:
                return self.target

            return second_corner

    
    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}
        robot_id = config.controlled_robot_ids[0]
        robot = state.robots_blue.get(robot_id)
        if robot is None or self.target is None:
            return {}

        half_length = state.field.length_m / 2.0
        waypoint = self._get_active_waypoint((robot.x_m, robot.y_m), half_length)

        # state.robots_yellow holds the obstacles (positions and measured velocities).
        dx = waypoint[0] - robot.x_m
        dy = waypoint[1] - robot.y_m
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return {robot_id: RobotIntent()}
        # Full 2.0 m/s when far, gentle deceleration in the final 0.35 m to avoid overshooting
        speed = min(MAX_SPEED_MPS, max(0.35, distance * 4.5)) if distance < 0.35 else MAX_SPEED_MPS
        return {
            robot_id: RobotIntent(
                vx_mps=speed * dx / distance,
                vy_mps=speed * dy / distance,
            )
        }
