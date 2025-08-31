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
model = PPO.load("checkpoints/ppo_connectfour_best_-100.747")
sb3_action, _ = model.predict(obs, deterministic=True)
print(f"SB3 action: {sb3_action}")

# Get SB3 logits
# with torch.no_grad():
#     features = model.policy.extract_features(obs)
#     latent_pi, _, _ = model.policy._get_latent(features)
#     distribution = model.policy._get_action_dist_from_latent(latent_pi)
#     action_mask = obs.get("action_mask", None)
#     if action_mask is not None:
#         action_mask = action_mask.to(distribution.distribution.logits.device)
#         all_zero = (action_mask.sum(dim=1) == 0)
#         if all_zero.any():
#             action_mask[all_zero] = 1.0
#         logits = distribution.distribution.logits
#         logits = logits + (action_mask - 1) * 1e9
#     sb3_logits = logits.numpy()[0]

# Load agent.py
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("agent_module", "submission.py")
module = importlib.util.module_from_spec(spec)
sys.modules["agent_module"] = module
class Config:
    def __init__(self):
        self.columns = 7
        self.rows = 6
        self.inarow = 4
spec.loader.exec_module(module)
agent_action = module.agent(kaggle_obs, Config())
# Get agent.py logits
# board = np.array(kaggle_obs['board'], dtype=np.float32)
# board = torch.from_numpy(board).unsqueeze(0).detach()
# mark = torch.tensor([float(kaggle_obs['mark'])], dtype=torch.float32).unsqueeze(0).detach()
# action_mask = np.zeros(Config().columns, dtype=np.float32)
# board2d = board.view(6, 7)
# for col in range(7):
#     if board2d[0, col] == 0:
#         action_mask[col] = 1.0
# obs_dict = {"board": board, "mark": mark, "action_mask": torch.from_numpy(action_mask).unsqueeze(0).detach()}
# with torch.no_grad():
#     dumped_logits = module.model(obs_dict)[0].numpy()
#     dumped_masked_logits = dumped_logits + (action_mask - 1) * 1e9

print(f"SB3 action: {sb3_action}, Agent.py action: {agent_action}")
# print(f"SB3 logits: {sb3_logits}")
# print(f"Dumped logits: {dumped_logits}")
# print(f"Dumped masked logits: {dumped_masked_logits}")
# print(f"Action mask: {action_mask}")
# print(f"Actions match: {int(sb3_action) == agent_action}")