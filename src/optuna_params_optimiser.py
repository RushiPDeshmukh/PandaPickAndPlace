import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym
import panda_gym

# Define the RL environment
ENV_NAME = 'PandaPickAndPlaceJointsDense-v3'

def objective(trial):
    # Suggest hyperparameters
    n_steps = trial.suggest_int('n_steps', 6, 13)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 1.0)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.95)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    

    # Create the environment
    env = make_vec_env(ENV_NAME, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Create the model
    model = PPO('MultiInputPolicy', env, n_steps=2**n_steps, gamma=gamma, learning_rate=learning_rate,
                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                gae_lambda=gae_lambda, n_epochs=n_epochs, verbose=0)

    # Train the model
    model.learn(total_timesteps=100000)  

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    # We want to maximize the mean reward
    return mean_reward

# Setup the study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

# Start optimization
study.optimize(objective, n_trials=100,show_progress_bar=True)  

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Plotting results
optuna.visualization.plot_parallel_coordinate(study).show()
optuna.visualization.plot_param_importances(study).show()
