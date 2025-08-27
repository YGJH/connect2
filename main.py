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



import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 自訂 Transformer 特徵提取器 with CNN backbone
class DictTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, d_model=128, num_layers=4, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[16, 32]):
        super().__init__(observation_space, features_dim)
        board_dim = observation_space.spaces["board"].shape[0]
        assert board_dim == height * width, f"Board dimension mismatch: expected {height*width}, got {board_dim}"
        self.height = height  # 6 rows
        self.width = width   # 7 columns
        self.seq_len = height * width  # 6*7=42
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # CNN backbone for spatial feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = 3  # One-hot for 0=empty, 1=player1, 2=player2
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        
        # CNN output: (B, cnn_channels[-1], height, width)
        self.cnn_out_dim = cnn_channels[-1]  # e.g., 32
        self.seq_len = height * width  # No downsampling: 6*7=42
        
        # Projection from CNN features to d_model
        self.cnn_projection = nn.Linear(self.cnn_out_dim, d_model)
        
        # Mark embedding
        self.mark_embedding = nn.Embedding(2, mark_emb_dim)
        self.mark_projection = nn.Linear(mark_emb_dim, d_model)
        
        # Positional encoding for sequence
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=256, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.out = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        board = observations["board"]  # (B, 42)
        mark = observations["mark"]    # (B, 1)
        B = board.shape[0]
        
        # Reshape board to 2D and one-hot encode: (B, 42) -> (B, 3, 6, 7)
        board_int = board.long()
        board_onehot = torch.zeros(B, 3, self.height, self.width, device=board.device)
        board_onehot.scatter_(1, board_int.view(B, 1, self.height, self.width), 1)
        x = board_onehot  # (B, 3, 6, 7)
        
        # Apply CNN layers
        for layer in self.cnn_layers:
            x = layer(x)  # (B, 32, 6, 7)
        
        # Flatten to sequence: (B, 32, 6, 7) -> (B, 42, 32)
        x = x.flatten(2).transpose(1, 2)  # (B, 6*7, 32)
        
        # Project to d_model
        x = self.cnn_projection(x)  # (B, 42, d_model)
        
        # Mark embedding and broadcast
        mark_idx = (mark.long() - 1).squeeze(-1)  # (B,)
        mark_emb = self.mark_embedding(mark_idx)  # (B, mark_emb_dim)
        mark_emb = self.mark_projection(mark_emb).unsqueeze(1)  # (B, 1, d_model)
        x += mark_emb  # Broadcast to all tokens: (B, 42, d_model)
        
        # Add positional encoding
        x += self.pos_encoding  # (B, 42, d_model)
        
        # Transformer
        x = self.transformer(x)  # (B, 42, d_model)
        
        # Global mean pooling
        x = x.mean(dim=1)  # (B, d_model)
        
        # Output
        return self.out(x)  # (B, features_dim)

class DictTransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DictTransformerExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                d_model=128,
                num_layers=4,
                num_heads=8,
                height=6,
                width=7,
                cnn_channels=[16, 32]
            ),
            **kwargs
        )

    def forward(self, obs, deterministic: bool = False):
        # Extract features
        features = self.extract_features(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Apply action mask (1=valid, 0=invalid)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(distribution.distribution.logits.device)
            # Handle all-zero masks to avoid NaN
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0

            # Mask invalid actions
            logits = distribution.distribution.logits
            logits = logits + (action_mask - 1) * 1e9
            distribution.distribution.logits = logits

        # Sample actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _get_latent(self, features: torch.Tensor):
        """
        Adapted from ActorCriticPolicy to unpack MlpExtractor latent vectors.
        """
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_sde = None  # No SDE
        return latent_pi, latent_vf, latent_sde

    def _predict(self, observation, deterministic: bool = False):
        """
        Used by SB3 for rollout. Returns only actions, consistent with forward.
        """
        features = self.extract_features(observation)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action_mask = observation.get("action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(distribution.distribution.logits.device)
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0
            logits = distribution.distribution.logits
            logits = logits + (action_mask - 1) * 1e9
            distribution.distribution.logits = logits

        return distribution.get_actions(deterministic=deterministic)



def make_env(rank, seed=0, render_mode=None):
    def _init():
        try:
            env = gym.make('ConnectFour-v0', render_mode=render_mode)
        except gym.error.NameNotFound:
            register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')
        env = gym.make('ConnectFour-v0', render_mode=render_mode)
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



    def visualize_model(model, num_episodes=1):
        env = gym.make('ConnectFour-v0', render_mode='human')
        try:
            for ep in range(num_episodes):
                obs, _ = env.reset()
                terminated = False
                truncated = False
                total_reward = 0.0
                steps = 0
                print(f"Visualization Episode {ep+1}")
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action.item())
                    mask = obs["action_mask"]
                    valid_actions = np.where(mask == 1)[0]
                    if mask[action] == 0:
                        if len(valid_actions) > 0:
                            action = int(np.random.choice(valid_actions))
                        else:
                            terminated = True
                            break
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    env.render()
                    time.sleep(0.5)
            print(f"Episode {ep+1} reward={total_reward} steps={steps}")
        finally:
            env.close()  # 確保無論如何都 close
            time.sleep(1)  # 給 Pygame quit 時間
        
    
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
  - 平均 reward (last 1000 steps): {mean_reward:.3f} ± {std_reward:.3f}
  - 最佳平均 reward: {self.best_mean_reward:.3f}
  - 平均 episode reward (last 100 eps): {avg_episode_reward:.3f}

🎮 結果:
  - 總共執行的遊戲: {total_games}
  - 勝利: {self.game_results['win']}  Draw: {self.game_results['draw']}  Loss: {self.game_results['loss']}
  - 勝率: {win_rate:.3f}  Draw Rate: {draw_rate:.3f}  Loss Rate: {loss_rate:.3f}

⏱ 進度:
  - 總共執行的遊戲: {self.episode_count}
  - 平均長度 (last 100): {avg_episode_length:.1f}
"""

        for opponent, stats in self.opponent_stats.items():
            if stats['games'] > 0:
                opp_wr = stats['wins'] / stats['games']
                log_info += f"\n  - {opponent}: {opp_wr:.3f} ({stats['wins']}/{stats['games']})"

        print(log_info)
        visualize_model(self.model, num_episodes=1)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            import os

            model_name = f"ppo_connectfour_best_{mean_reward:.3f}.zip"
            self.model.save(os.path.join('checkpoints', model_name))
            send_telegram(f"新最佳模型\nMean(step1000)={mean_reward:.3f} WinRate={win_rate:.3f}")
            print(f"[Saved] {model_name}")


        # if self.num_timesteps % self.save_freq == 0:
        #     ckpt = f"ppo_connectfour_checkpoint_{self.num_timesteps}.zip"
        #     self.model.save(ckpt)
        #     print(f"[Checkpoint] {ckpt}")

    def _on_training_end(self):
        self._evaluate_and_log()
        final_name = "ppo_connectfour_final.zip"
        import os

        self.model.save(os.path.join('checkpoints', final_name))
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
            # SB3 predict，obs 是 dict，policy 会用 action_mask 过滤
            action, _ = model.predict(obs, deterministic=True)
            action = int(action.item())  # Convert to scalar int
            # Check action mask for validity
            mask = obs["action_mask"]
            valid_actions = np.where(mask == 1)[0]
            if mask[action] == 0:
                if len(valid_actions) > 0:
                    action = int(np.random.choice(valid_actions))
                    print(f"[visualize_model] Invalid action {action} predicted, fallback to random valid action={action}")
                else:
                    print("[visualize_model] No valid actions available (board likely full), forcing termination")
                    terminated = True
                    break
            # Log action details
            # print(f"[visualize_model] Step {steps+1}: current_player={env.current_player}, action={action}, mask={mask}")
            # 交给环境
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            # 调用带 mode 的 render，强制走 pygame 路径
            env.render()
            print(f"[visualize_model] Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
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
    num_cpu = 6
    
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
    callback = EvaluationCallback(verbose=1, eval_freq=1000)

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
    visualize_model(model, num_episodes=10)

if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')  # 強制用 spawn，避免 fork 問題
    main()