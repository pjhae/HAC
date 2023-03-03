import random
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
env = HalfCheetahEnv()

while True:
	env.render()
	action = [random.randint(0,1) for _ in range(env.action_space.shape[0])]
	_, _, _, _ = env.step(action)

env.close()
