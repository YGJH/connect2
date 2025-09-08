import torch
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs.registration import register

# Register environment
register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')
env = gym.make('ConnectFour-v0')
obs, _ = env.reset()

# Simulate Kaggle observation format
kaggle_obs = {
    'board': obs['board'].tolist(),  # Convert to list to match Kaggle
    'mark': int(obs['mark'].item())  # Convert to int
}
config = {'rows': 6, 'columns': 7, 'inarow': 4}

# Load SB3 model
model = PPO.load("temp.zip")
sb3_action, _ = model.predict(obs, deterministic=True)
print(f"SB3 action: {sb3_action}")


import importlib.util
import sys
spec = importlib.util.spec_from_file_location("agent_module", "opponents/submission_vMega.py")
module = importlib.util.module_from_spec(spec)
sys.modules["agent_module"] = module
class Config:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4
spec.loader.exec_module(module)
agent_action = module.agent(kaggle_obs, Config())
print(f"SB3 action: {sb3_action}, Agent.py action: {agent_action}")
