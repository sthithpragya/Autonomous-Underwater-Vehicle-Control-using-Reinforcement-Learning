from gym.envs.registration import register

register(
    id='underwater-v0',
    entry_point='gym_underwater.envs:UnderwaterEnv',)