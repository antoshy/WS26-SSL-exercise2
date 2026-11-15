from ssl_strategy_api.run_demos.run_rsoccer_exercise2 import Exercise2Env
from avoidance_features import build_obs

env = Exercise2Env(n_random_obstacles=0, seed=1)
state, _ = env.reset(seed=1)
obs = build_obs(state, env.target)

print(state.robots_blue[0], env.target)
print(obs.shape, obs.dtype)     # (35,) float32
print(obs.round(2))
env.close()