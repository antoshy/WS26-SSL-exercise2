import numpy as np
import math
from ssl_strategy_api.models import TeamColor, TeamConfig, WorldState, FieldGeometry, RobotIntent


K_OBSTACLES = 4 # four nearest enemies get into the feature vector
OBS_DIM = 11 + 4 * K_OBSTACLES # 11 features plus 4 for each nearest enemy
MAX_SPEED = 2.0
PENALTY_LENGTH = 1.0
PENALTY_WIDTH = 2.0
FIELD_SCALE = 4.5
NEAR_SCALE = 2.0

def build_obs(state: WorldState, target: tuple[float, float], robot_id=0):
    """ convert a worldstate into the feature vector """
    half_field_length = state.field.length_m / 2            # 4.5
    half_field_width = state.field.width_m / 2              # 3
    penalty_x_boundary = half_field_length - PENALTY_LENGTH # 3.5
    penalty_y_boundary = PENALTY_WIDTH / 2                  # 1
    
    robot = state.robots_blue.get(robot_id)
    dx = (target[0] - robot.x_m)
    dy = (target[1] - robot.y_m)
    distance = max(math.hypot(dx, dy), 1e-6)

    features = np.zeros(OBS_DIM, dtype=np.float32)

    features[0] = dx / distance # direction to target on x-axis
    features[1] = dy / distance # direction to target on y-axis
    features[2] = distance / FIELD_SCALE # distance to target
    features[3] = robot.vx_mps / MAX_SPEED # robot's speed on x-axis                                                                                     
    features[4] = robot.vy_mps / MAX_SPEED # robot's speed on y-axis                                                                                     
    features[5] = (half_field_length - robot.x_m) / FIELD_SCALE   # distance to the right field end                                                      
    features[6] = (robot.x_m + half_field_length) / FIELD_SCALE   # distance to the left field end                                                       
    features[7] = (half_field_width - robot.y_m) / FIELD_SCALE    # distance to the upper field end                                                      
    features[8] = (robot.y_m + half_field_width) / FIELD_SCALE    # distance to the lower field end                                                      
    features[9] = (max(penalty_x_boundary, robot.x_m) - robot.x_m) / NEAR_SCALE                           # distance to nearest point of goal area x-axis
    features[10] = (max(-penalty_y_boundary, min(penalty_y_boundary, robot.y_m)) - robot.y_m) / NEAR_SCALE # distance to nearest point of goal area y-axis
    features[11:] = _obstacle_features(robot, state.robots_yellow.values())

    return features


def _obstacle_features(robot, obstacles):
    by_distance = sorted(
        obstacles,
        key=lambda o: math.hypot(o.x_m - robot.x_m, o.y_m - robot.y_m),
    )
    nearest = by_distance[:K_OBSTACLES]

    features = []
    for obstacle in nearest:
        dx = (obstacle.x_m - robot.x_m) / NEAR_SCALE
        dy = (obstacle.y_m - robot.y_m) / NEAR_SCALE
        relative_vx = (obstacle.vx_mps - robot.vx_mps) / MAX_SPEED
        relative_vy = (obstacle.vy_mps - robot.vy_mps) / MAX_SPEED
        features += [dx, dy, relative_vx, relative_vy]

    for i in range(K_OBSTACLES - len(nearest)):
        features += [-1.1, -1.1, 0, 0] # empty seats (outside the field)

    return features

def action_to_intent(action: np.ndarray) -> RobotIntent:
    """convert two values for x and y -> RobotIntent(vx, vy)"""
    intent = RobotIntent(
        vx_mps = float(action[0]) * MAX_SPEED,
        vy_mps = float(action[1]) * MAX_SPEED
    )
    return intent