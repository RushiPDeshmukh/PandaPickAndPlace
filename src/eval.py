from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaPickAndPlace-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

#  do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

model_name = "a2c-PandaPickAndPlace-v3"
# Load the agent
model = A2C.load(model_name)

mean_reward, std_reward = evaluate_policy(model, eval_env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")