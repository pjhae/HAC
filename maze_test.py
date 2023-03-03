import gym
import mujoco_maze
import random
import numpy as np

# Question) 왜 .unwrapped 를 추가 안하면 env.wrapped_env.get_body_com("torso")[:2] 에서 얻어야 하고 mazeEnv 에서 정의된 _get_obs()같은 함수 바로 못쓰는걸까?

env = gym.make("AntSquareRoom-v0").unwrapped
env.reset()
action = np.zeros(8)
print(env._get_obs_space().high)  ## [26 , 2]
print(env._get_obs_space().low)   ## [-2 , -26]
env._get_obs_space().high[1] = 101010
print(env._get_obs_space().high[2:])
print(env.action_space.high)  ## [30. 30. 30. 30. 30. 30. 30. 30.]
print(env.action_space.low)   ## [-30. -30. -30. -30. -30. -30. -30. -30.]

print(env.action_space.high - env.action_space.low)
print(env._get_obs_space().high - env._get_obs_space().low)

print(env._get_obs().shape[0])  ## 2

state_bounds = np.array((env._get_obs_space().high - env._get_obs_space().low)/2)
state_offset =  np.array((env._get_obs_space().high + env._get_obs_space().low)/2)
    
for i in range(len(state_bounds)):
    if env._get_obs_space().high[i] == float('inf'):
        state_bounds[i] = 100
        state_offset[i] = 0


print(state_bounds)
print(state_offset)

while True:

    for i in range(len(action)):
        action[i] = 60*(random.random()-0.5)
    env.step(action)
    env.render()
   