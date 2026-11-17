import math
from typing import List, Mapping, Optional, Tuple

from ssl_strategy_api.models import RobotIntent, RobotState, TeamColor, TeamConfig, WorldState

ROBOT_RADIUS_M = 0.09
COLLISION_DIST_M = 2.0 * ROBOT_RADIUS_M + 0.05  # 0.18 m center-to-center touch
OBSTACLE_MARGIN_M = 0.02                 
WALL_MARGIN_M = 0.01

PENALTY_LENGTH_M = 1.0
PENALTY_WIDTH_M = 2.0
BOX_SAFETY_MARGIN_M = ROBOT_RADIUS_M
MAX_SPEED_MPS = 2.0

HORIZON_STEPS = 8
STEP_DT_S = 0.10

class ConflictAvoidanceStrategy:

    def __init__(self) -> None:
        self.target: Optional[Tuple[float, float]] = None
        self._last_cmd: Tuple[float, float] = (0.0, 0.0)


    def set_target(self, x_m: float, y_m: float) -> None:
        """Called by the runner with the position the blue robot must reach."""
        self.target = (x_m, y_m)


    def reset(self, seed=None) -> None:
        self._last_cmd = (0.0, 0.0)


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


    def _sample_candidates(self, robot: RobotState, waypoint: Tuple[float, float]) -> List[Tuple[float, float]]:
        dx = waypoint[0] - robot.x_m
        dy = waypoint[1] - robot.y_m
        distance = math.hypot(dx, dy)

        desired_speed = (min(MAX_SPEED_MPS, max(0.35, distance * 4.5)) 
            if distance < 0.35 
            else MAX_SPEED_MPS
        )
        # 24 angles around the circle (every 15 degrees) + exact angle to waypoint
        angles = [2.0 * math.pi * i / 24 for i in range(24)]
        if distance > 1e-6:
            angles.append(math.atan2(dy, dx))

        speeds = [desired_speed, 1.3, 0.6, 0.25]
        candidates: List[Tuple[float, float]] = [(0.0, 0.0)]
        for spd in speeds:
            for ang in angles:
                candidates.append((spd * math.cos(ang), spd * math.sin(ang)))
        return candidates


    def _evaluate_candidate(self,vx: float,vy: float,robot: RobotState,obstacles: List[RobotState],half_length: float,half_width: float,waypoint: Tuple[float, float],
        ) -> Tuple[bool, float, float]:
            """Simulate (vx, vy) for 0.8s. Returns (is_safe, cost, min_clearance)."""
            safe_dist = COLLISION_DIST_M + OBSTACLE_MARGIN_M
            min_clearance = float("inf")
            is_safe = True

            end_x, end_y = robot.x_m, robot.y_m
            box_margin = BOX_SAFETY_MARGIN_M
            if self.target is not None and self._is_in_penalty_box(
                self.target[0], self.target[1], half_length, BOX_SAFETY_MARGIN_M
            ):
                box_margin = 0.01

            for step in range(1, HORIZON_STEPS + 1):
                t = step * STEP_DT_S
                px = robot.x_m + vx * t
                py = robot.y_m + vy * t
                end_x, end_y = px, py

                # 1. Wall check
                if abs(px) > half_length - WALL_MARGIN_M or abs(py) > half_width - WALL_MARGIN_M:
                    is_safe = False

                # 2. Penalty box check
                if self._is_in_penalty_box(px, py, half_length, box_margin):
                    is_safe = False

                # 3. Obstacle check (static positions for Stage C)
                for obs in obstacles:
                    dist = math.hypot(px - obs.x_m, py - obs.y_m)
                    if dist < min_clearance:
                        min_clearance = dist
                    if dist < safe_dist:
                        is_safe = False

                # Stop rollout early if we already reached the final target
                if self.target is not None and math.hypot(self.target[0] - px, self.target[1] - py) <= 0.08:
                    break

            # Score the trajectory (lower cost = better)
            dist_to_wp = math.hypot(waypoint[0] - end_x, waypoint[1] - end_y)
            clearance_penalty = 1.2 * (0.45 - min_clearance) if min_clearance < 0.45 else 0.0
            smoothness_penalty = 0.08 * math.hypot(vx - self._last_cmd[0], vy - self._last_cmd[1])

            cost = dist_to_wp + clearance_penalty + smoothness_penalty
            return is_safe, cost, min_clearance
    

    def decide(self, state: WorldState, config: TeamConfig) -> Mapping[int, RobotIntent]:
        if not config.controlled_robot_ids or config.team != TeamColor.BLUE:
            return {}
        robot_id = config.controlled_robot_ids[0]
        robot = state.robots_blue.get(robot_id)
        if robot is None or self.target is None:
            return {}

        half_length = state.field.length_m / 2.0
        waypoint = self._get_active_waypoint((robot.x_m, robot.y_m), half_length)

        half_width = state.field.width_m / 2.0
        if math.hypot(self.target[0] - robot.x_m, self.target[1] - robot.y_m) < 1e-6:
            self._last_cmd = (0.0, 0.0)
            return {robot_id: RobotIntent()}

        obstacles = list(state.robots_yellow.values())
        candidates = self._sample_candidates(robot, waypoint)

        best_safe_cmd: Optional[Tuple[float, float]] = None
        best_safe_cost = float("inf")
        best_fallback_cmd: Tuple[float, float] = (0.0, 0.0)
        best_fallback_clearance = -1.0

        for vx, vy in candidates:
            is_safe, cost, min_clearance = self._evaluate_candidate(
                vx, vy, robot, obstacles, half_length, half_width, waypoint
            )
            if is_safe and cost < best_safe_cost:
                best_safe_cost = cost
                best_safe_cmd = (vx, vy)
            if min_clearance > best_fallback_clearance:
                best_fallback_clearance = min_clearance
                best_fallback_cmd = (vx, vy)

        chosen_vx, chosen_vy = best_safe_cmd if best_safe_cmd is not None else best_fallback_cmd
        self._last_cmd = (chosen_vx, chosen_vy)
        return {robot_id: RobotIntent(vx_mps=chosen_vx, vy_mps=chosen_vy)}
