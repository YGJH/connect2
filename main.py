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

import connectFour

# 自訂 Transformer 特徵提取器
class DictTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024, d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
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

class EvaluationCallback(BaseCallback):
    def __init__(self, verbose=0, eval_freq=1000, save_freq=5000):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -float('inf')
        self.rewards = []                     # step rewards (all envs)
        self.episode_rewards = []             # finished episodes total reward
        self.current_episode_rewards = None   # per-env accumulating buffer
        self.episode_lengths = []
        self.win_rates = []
        self.game_results = {'win': 0, 'loss': 0, 'draw': 0}   # 修正 key
        self.opponent_stats = {}
        self.episode_count = 0
        self.last_eval_step = 0

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs

    def _on_step(self):
        infos = self.locals.get('infos', [])
        step_rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])

        # 累積全域 step reward
        self.rewards.extend(step_rewards)

        # 累積每個環境的 episode reward
        for i, r in enumerate(step_rewards):
            self.current_episode_rewards[i] += r

        # 處理 done
        for i, (info, done) in enumerate(zip(infos, dones)):
            if done:
                ep_r = self.current_episode_rewards[i]
                self.episode_rewards.append(ep_r)
                self.current_episode_rewards[i] = 0.0
                self.episode_count += 1

                game_result = info.get('game_result')
                if game_result in self.game_results:
                    self.game_results[game_result] += 1

                opponent = str(info.get('opponent_type', 'unknown'))
                if opponent not in self.opponent_stats:
                    self.opponent_stats[opponent] = {'games': 0, 'wins': 0}
                self.opponent_stats[opponent]['games'] += 1
                if game_result == 'win':
                    self.opponent_stats[opponent]['wins'] += 1

                if 'episode_length' in info:
                    self.episode_lengths.append(info['episode_length'])
                if 'win_rate' in info:
                    self.win_rates.append(info['win_rate'])

        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self._evaluate_and_log()
            self.last_eval_step = self.num_timesteps
        return True

    def _evaluate_and_log(self):
        if len(self.rewards) < 100:
            return

        recent_rewards = self.rewards[-1000:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)

        total_games = sum(self.game_results.values())
        if total_games > 0:
            win_rate = self.game_results['win'] / total_games
            draw_rate = self.game_results['draw'] / total_games
            loss_rate = self.game_results['loss'] / total_games
        else:
            win_rate = draw_rate = loss_rate = 0.0

        avg_episode_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
        avg_episode_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0

        log_info = f"""
=== 訓練統計 (Step: {self.num_timesteps}) ===
📊 Reward:
  - Mean (last 1000 steps): {mean_reward:.3f} ± {std_reward:.3f}
  - Best step mean: {self.best_mean_reward:.3f}
  - Mean episode reward (last 100 eps): {avg_episode_reward:.3f}

🎮 Results:
  - Total finished games: {total_games}
  - Win: {self.game_results['win']}  Draw: {self.game_results['draw']}  Loss: {self.game_results['loss']}
  - Win Rate: {win_rate:.3f}  Draw Rate: {draw_rate:.3f}  Loss Rate: {loss_rate:.3f}

⏱ Episodes:
  - Finished episodes: {self.episode_count}
  - Avg length (last 100): {avg_episode_length:.1f}

🤖 Opponents:"""

        for opponent, stats in self.opponent_stats.items():
            if stats['games'] > 0:
                opp_wr = stats['wins'] / stats['games']
                log_info += f"\n  - {opponent}: {opp_wr:.3f} ({stats['wins']}/{stats['games']})"

        print(log_info)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            model_name = f"ppo_connectfour_best_{mean_reward:.3f}.zip"
            self.model.save(model_name)
            send_telegram(f"新最佳模型\nMean(step1000)={mean_reward:.3f} WinRate={win_rate:.3f}")
            print(f"[Saved] {model_name}")

        if self.num_timesteps % self.save_freq == 0:
            ckpt = f"ppo_connectfour_checkpoint_{self.num_timesteps}.zip"
            self.model.save(ckpt)
            print(f"[Checkpoint] {ckpt}")

    def _on_training_end(self):
        self._evaluate_and_log()
        final_name = "ppo_connectfour_final.zip"
        self.model.save(final_name)
        total_games = sum(self.game_results.values())
        win_rate = self.game_results['win'] / total_games if total_games else 0
        send_telegram(f"訓練結束 steps={self.num_timesteps} games={total_games} win_rate={win_rate:.3f}")


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Path to the output file (optional).')
    args = parser.parse_args()
    model_path = args.model

    # 註冊環境
    register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')
    
    # 設置參數
    num_cpu = 5
    total_timesteps = 3_000_000

    print(f"使用 {num_cpu} 個並行環境進行訓練")
    
    # 創建多進程向量化環境
    if num_cpu == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    layer = [512] * 6
    model = PPO(
        DictTransformerPolicy,
        env,
        verbose=1,
        n_steps=2048 // num_cpu,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=layer),
    )
    
    # 創建改進的回調
    callback = EvaluationCallback(verbose=1, eval_freq=1000, save_freq=10000)

    if model_path:
        model.load(model_path)
        print(f"已載入模型: {model_path}")
    
    print("開始訓練...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print(f"訓練完成！耗時: {time.time() - start_time:.2f} 秒")
    env.close()
    
    print("開始可視化測試...")
    visualize_model(model, num_episodes=3)

if __name__ == '__main__':
    main()