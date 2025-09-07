import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import time
from stable_baselines3.common.distributions import CategoricalDistribution
import os
import requests
import gc
import gymnasium.spaces as spaces
from stable_baselines3.common.vec_env import VecNormalize
import math
from typing import Callable

def cosine_annealing_schedule(initial_value: float):
    """
    Cosine annealing learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        fraction = 1.0 - progress_remaining
        return initial_value * 0.5 * (1.0 + math.cos(math.pi * fraction))

    return func

def make_env(rank, seed=0, render_mode=None):
    def _init():
        try:
            env = gym.make('SelfPlayConnectFourEnv')
        except gym.error.NameNotFound:
            register(id='SelfPlayConnectFourEnv', entry_point='temp:SelfPlayConnectFourEnv')  # Adjust entry_point if needed
            env = gym.make('SelfPlayConnectFourEnv')
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    register(id='SelfPlayConnectFourEnv', entry_point='temp:SelfPlayConnectFourEnv')  # Adjust if needed
    num_cpu = 4

    if num_cpu == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=cosine_annealing_schedule(3e-4),
        verbose=1,
    )   

    # 訓練循環：每隔幾輪更新對手模型
    for iteration in range(10):  # 例如 10 個迭代
        model.learn(total_timesteps=10000, progress_bar=True)  # 訓練一輪

        # 更新對手為當前模型的複製 (或舊版本存檔)
        opponent_model = deepcopy(model)
        env.opponent_model = opponent_model
        
        # 可選：存檔舊模型到 opponent pool，隨機選舊版對手避免過擬合
        # 例如：opponent_pool.append(deepcopy(model))，然後 env.opponent_model = random.choice(opponent_pool)

    model.save("chess_ppo_selfplay")

if __name__ == '__main__':
    main()