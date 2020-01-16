# case 1 - 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_underwater.envs.underwater_env import UnderwaterEnv
import numpy as np

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: UnderwaterEnv()])
model = PPO2(MlpPolicy, env, verbose=1)

model = PPO2.load("girona500_betaInter")

obs1 = env.reset()
obs2 = env.reset()
obs3 = env.reset()

# print(obs)

testGoal1 = [.15,.15,.15,0,0,1.57,0,0,0,0,0,0]
testGoal2 = [.05,.35,-.25,0,0,1.57,0,0,0,0,0,0]
testGoal3 = [1.20,.05,-.15,0,0,1.57,0,0,0,0,0,0]

obs1[0,:] = np.asarray(testGoal1)
obs2[0,:] = np.asarray(testGoal2)
obs3[0,:] = np.asarray(testGoal3)

print("case1")
print("before ",obs1)
for i in range(5):
	action, _states = model.predict(obs1)
	print("action ", action)
	obs1, rewards, done, info = env.step(action)
	print("after ",obs1)
	env.render()

print("-----------------------------")

print("case2")
print("before ",obs2)
for i in range(5):
	action, _states = model.predict(obs2)
	print("action ", action)
	obs2, rewards, done, info = env.step(action)
	print("after ",obs2)
	env.render()

print("-----------------------------")

print("case3")
print("before ",obs3)
for i in range(15):
	action, _states = model.predict(obs3)
	print("action ", action)
	obs3, rewards, done, info = env.step(action)
	print("after ",obs3)
	env.render()

