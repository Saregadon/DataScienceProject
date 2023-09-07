#import dependencies
import os
import gym
import pytorch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

