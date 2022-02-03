from gym.envs.registration import register
from .connect_four_env import ConnectFourEnv, ResultType

register(
    id='ConnectFour-v0',
    entry_point='gym_connect_four:ConnectFourEnv',
)
