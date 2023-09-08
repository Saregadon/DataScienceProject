#import dependencies
import os
import torch #pytorch
import numpy as np
import pygame
import gymnasium as gym
import stable_baselines3
import tensorflow as tf
from stable_baselines3 import PPO #Proximal Policy Optimization Algorithm
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

pygame.init()

env_name = 'CartPole-v1' #maps to preinstalled environment from gym
env = gym.make(env_name) #creating environment
observation, info = env.reset(seed=42)

# Make directories
log_path = os.path.join('Training', 'Logs') 

env = gym.make(env_name, render_mode ='human')

#creates the pygame window and display
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("CartPole")

#clock controls frame rate
clock = pygame.time.Clock()

#enviroment wrapper with DummyVecEnv
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path) #MLP = Multilayer Perceptron Policy

model.learn(total_timesteps = 20000) #trains the model

PPO_Path = os.path.join('Training','Saved Models', 'PPO_Model_Cartpole') #saves place of the model
model.save(PPO_Path) #saves model

def render_CartPole(observation):
    #white background for screen
    screen.fill((255, 255, 255))

    cart_x = int(observation[0] * 50) + 200
    pole_x = cart_x + 10
    pole_y = 100

    pygame.draw.rect(screen, (0, 0, 0), (cart_x, 150, 20, 10))
    pygame.draw.line(screen, (0, 0, 0), (pole_x, 150), (pole_x, pole_y), 5)

    pygame.display.update()

def eval_and_render():
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            render_CartPole(obs[0])
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

#evaluates
evaluate_policy(model, env, n_eval_episodes(10), render = True)

pygame.quit()