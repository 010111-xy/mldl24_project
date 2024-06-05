import gym
from env.custom_hopper import CustomHopper

def register_custom_hopper():
  gym.envs.registration.register(
      id='CustomHopper-source-v0',
      entry_point='env.custom_hopper:CustomHopper',
  )