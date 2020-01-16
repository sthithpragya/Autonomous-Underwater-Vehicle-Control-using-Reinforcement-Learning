# case 1 - 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from gym_underwater.envs.underwater_env import UnderwaterEnv
import numpy as np

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: UnderwaterEnv()])
model = PPO2(MlpPolicy, env, verbose=1)

#betaPoor - 5 mil
#betaInter - 2 mil
#betaGood - 5 mil
#betaBetter - 2 mil
#betaBest - 2 mil




model.learn(total_timesteps=1000000) # training for 1 mil steps for each beta limit

model.save("girona500_beta1")
model = PPO2.load("girona500_beta1")

obs = env.reset()

for i in range(500):
  # print("iter ", i)
  action, _states = model.predict(obs)
  # print("action ", action)
  # print("before ",obs)
  obs, rewards, done, info = env.step(action)
  # print("rewards ", rewards)
  # print("after ",obs)
  env.render()





















