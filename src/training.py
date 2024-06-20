#!/Users/rushideshmukh/Library/Mobile Documents/com~apple~CloudDocs/RL Projects/PandaPickAndPlace/venv/bin/python
import os
import gymnasium as gym
import panda_gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

env_id = 'PandaPickAndPlaceDense-v3'
env = make_vec_env(env_id, n_envs=4)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Suggest hyperparameters
n_steps = 12
gamma =  0.9567216423439765
learning_rate = 0.003590397817946911
ent_coef = 0.040893253736167264
vf_coef = 0.8728746094624913
max_grad_norm = 0.7648677633846508
gae_lambda =  0.9183112170474744
n_epochs = 3


# Create the model
model = PPO('MultiInputPolicy', env, n_steps=2**n_steps, gamma=gamma, learning_rate=learning_rate,
            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda, n_epochs=n_epochs, verbose=1, device='mps')

model.learn(total_timesteps=10000000,progress_bar=True)

model_name = "a2c-PandaPickAndPlace-v3"
model.save("../model/"+model_name)
env.save("vec_normalize.pkl")