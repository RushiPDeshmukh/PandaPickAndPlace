#!/Users/rushideshmukh/Library/Mobile Documents/com~apple~CloudDocs/RL Projects/PandaPickAndPlace/venv/bin/python
import os
import gymnasium as gym
import panda_gym
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize,DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.her import GoalSelectionStrategy
from stable_baselines3 import HerReplayBuffer

env_id = 'PandaPickAndPlaceDense-v3'
env = make_vec_env(env_id,4)
# Seed the environment
env.seed(42)
# env = TimeFeatureWrapper(env,)


policy_kwargs = dict(net_arch=[512, 512, 512], n_critics=2)
model = TQC("MultiInputPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs,replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=dict(goal_selection_strategy=GoalSelectionStrategy.FUTURE, n_sampled_goal=4,),batch_size=2048,buffer_size=100000,gamma=0.95,learning_rate=0.001,tau=0.05,learning_starts=1000,device='mps')
model.learn(total_timesteps=1000000, log_interval=4, progress_bar=True)
model_name = "a2c-PandaPickAndPlace-v3"
model.save("../model/"+model_name)
env.save("vec_normalize.pkl")