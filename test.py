import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import time


# 自訂 Transformer 特徵提取器
class DictTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
        super().__init__(observation_space, features_dim)
        board_dim = observation_space.spaces["board"].shape[0]
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.mark_embedding = nn.Embedding(2, mark_emb_dim)
        self.input_projection = nn.Linear(board_dim + mark_emb_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        board = observations["board"]
        mark = observations["mark"]
        mark_idx = (mark.long() - 1).squeeze(-1)
        mark_emb = self.mark_embedding(mark_idx)
        x = torch.cat([board, mark_emb], dim=1)
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.out(x)

class DictTransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DictTransformerExtractor,
            features_extractor_kwargs=dict(features_dim=128, d_model=64, num_layers=14, num_heads=16),
            **kwargs
        )

def make_env(rank, seed=0):
    """
    創建環境的工廠函數
    
    Args:
        rank: 進程編號
        seed: 隨機種子
    """

    def _init():
        # 為每個進程設置不同的隨機種子
        try:
            env = gym.make('ConnectFour-v0')
        except gym.error.NameNotFound:
            register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')

        env = gym.make('ConnectFour-v0')
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# 自訂回調以記錄平均獎勵並保存模型
class EvaluationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EvaluationCallback, self).__init__(verbose)
        self.best_mean_reward = -float('inf')
        self.rewards = []

    def _on_step(self):
        self.rewards.extend(self.locals['rewards'])
        # 每 1000 步檢查一次平均獎勵並保存最佳模型
        if len(self.rewards) >= 1000:
            mean_reward = np.mean(self.rewards[-1000:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"ppo_connectfour_reward_{mean_reward:.2f}.pt")
                print(f"Saved model with mean reward: {mean_reward:.2f}")
        return True

    def _on_training_end(self):
        mean_reward = np.mean(self.rewards)
        print(f"Average reward: {mean_reward}")
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(f"ppo_connectfour_reward_{mean_reward:.2f}.pt")
            print(f"Saved model with mean reward: {mean_reward:.2f}")



def send_telegram(msg: str):
    import os
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or "6166024220"
    if not token or not chat_id:
        print("未設置 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID，略過訊息通知。")
        return
    try:
        import requests
        base = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg}
        r = requests.post(base, data=payload, timeout=3.0)
        print("已發送 Telegram 通知。")
    except Exception as e:
        print(f"Telegram 發送失敗: {e}")


def visualize_model(model, num_episodes=5):
    """使用單個環境進行可視化"""
    env = gym.make('ConnectFour-v0', render_mode='human')

    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        print(f"Visualization Episode {ep+1}")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.5)
        print(f"Episode {ep+1} reward={total_reward} steps={steps}")
    env.close()

def main():
    # 註冊環境
    register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')
    
    # 設置參數
    num_cpu = 1  # 使用的 CPU 核心數
    total_timesteps = 10
    
    print(f"使用 {num_cpu} 個並行環境進行訓練")
    
    # 創建多進程向量化環境
    if num_cpu == 1:
        # 單進程版本（調試用）
        env = DummyVecEnv([make_env(0)])
    else:
        # 多進程版本
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # 創建模型
    model = PPO(
        DictTransformerPolicy,
        env,
        verbose=1,
        n_steps=20 // num_cpu,  # 每個環境的步數
        batch_size=256,           # 批次大小
        n_epochs=10,              # 每次更新的 epoch 數
        learning_rate=3e-4,       # 學習率
        clip_range=0.2,           # PPO clip 範圍
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    
    # 創建回調
    callback = EvaluationCallback()
    
    print("開始訓練...")
    start_time = time.time()
    
    # 訓練模型
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print(f"訓練完成！耗時: {time.time() - start_time:.2f} 秒")
    
    # 保存最終模型
    model.save("ppo_connect4_multiprocessing_final.zip")
    print("最終模型已保存")
    
    # 關閉環境
    env.close()
    
    # 可視化測試（可選）
    print("開始可視化測試...")
    visualize_model(model, num_episodes=3)
    send_telegram('Connect Four 模型訓練完成並已保存！')


if __name__ == "__main__":
    main()