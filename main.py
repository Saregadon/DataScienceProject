#import dependencies
import os
import torch #pytorch
import numpy as np
import gymnasium as gym
import stable_baselines3
import tensorflow as tf
from stable_baselines3 import PPO #Proximal Policy Optimization Algorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = 'CartPole-v1' #maps to preinstalled environment from gym
env = gym.make(env_name) #creating environment
observation, info = env.reset(seed=42)

"""
episodes = 5 #number of loops through the game
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, truncated, info = env.step(action)
        score += reward

        if done or truncated:
            observation, info = env.reset()

        #print(env.reset())
    print('Episode: {} Score: {}'.format(episode, score))
env.close()
"""

# Make directories
log_path = os.path.join('Training', 'Logs') 

env = gym.make(env_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) #MLP = Multilayer Perceptron Policy

model.learn(total_timesteps = 20000) #trains the model

PPO_Path = os.path.join('Training','Saved Models', 'PPO_Model_Cartpole') #saves place of the model
model.save(PPO_Path) #saves model