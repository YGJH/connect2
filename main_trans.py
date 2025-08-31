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

import torch.nn.functional as F


import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DictTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, d_model=128, num_layers=12, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[128, 128]):
        super().__init__(observation_space, features_dim)
        board_dim = observation_space.spaces["board"].shape[0]
        assert board_dim == height * width, f"Board dimension mismatch: expected {height*width}, got {board_dim}"
        self.height = height
        self.width = width
        self.seq_len = height * width
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # CNN backbone
        self.cnn_layers = nn.ModuleList()
        in_channels = 3
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.cnn_layers.append(nn.BatchNorm2d(out_channels))
            self.cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.cnn_out_dim = cnn_channels[-1]
        self.cnn_projection = nn.Linear(self.cnn_out_dim, d_model)
        
        # Mark embedding
        self.mark_embedding = nn.Embedding(2, mark_emb_dim)
        self.mark_projection = nn.Linear(mark_emb_dim, d_model)
        
        # Positional encoding
        def positional_encoding(seq_len, d_model, device='cpu'):
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0).to(device)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pos_encoding = nn.Parameter(positional_encoding(self.seq_len, d_model, device=device), requires_grad=False)
        
        # Transformer with residual blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=512,
                dropout=0.1, batch_first=True, norm_first=True
            ))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        # Output layer
        self.out = nn.Linear(d_model * 2, features_dim)

    def forward(self, observations):
        board = observations["board"].to(self.cnn_layers[0].weight.device)
        mark = observations["mark"].to(self.cnn_layers[0].weight.device)
        B = board.shape[0]
        
        # Assert mark validity
        assert torch.all((mark == 1) | (mark == 2)), "Mark must be 1 or 2"
        
        # Reshape and one-hot encode
        board_int = board.long().view(B, self.height, self.width)
        board_onehot = F.one_hot(board_int, num_classes=3).permute(0, 3, 1, 2).float()
        x = board_onehot
        
        # CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Flatten and project
        x = x.flatten(2).transpose(1, 2)
        x = self.cnn_projection(x)
        
        # Positional encoding
        x += self.pos_encoding  # 現在 pos_encoding 已與 x 在同一設備
        
        # Transformer blocks with residual and LayerNorm
        for block, norm in zip(self.transformer_blocks, self.norms):
            residual = x
            x = block(x)
            x = norm(x + residual)
        
        # Global mean pooling
        pooled_x = x.mean(dim=1)
        
        # Mark embedding and concat
        mark_idx = (mark.long() - 1).squeeze(-1)
        mark_emb = self.mark_embedding(mark_idx)
        mark_emb = self.mark_projection(mark_emb)
        combined_x = torch.cat([pooled_x, mark_emb], dim=1)
        
        # Output
        return self.out(combined_x)

class DictTransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DictTransformerExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                d_model=128,
                num_layers=12,
                num_heads=8,
                height=6,
                width=7,
                cnn_channels=[128, 128]  # Sync with Extractor
            ),
            **kwargs
        )

    def _apply_action_mask(self, distribution, action_mask):
        if action_mask is not None:
            action_mask = action_mask.to(distribution.distribution.logits.device)
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0  # Consider raise error if terminal
            logits = distribution.distribution.logits
            logits = logits + (action_mask - 1) * 1e9
            distribution.distribution.logits = logits
        return distribution

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action_mask = obs.get("action_mask", None)
        distribution = self._apply_action_mask(distribution, action_mask)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def _get_latent(self, features: torch.Tensor):
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_sde = None
        return latent_pi, latent_vf, latent_sde

    def _predict(self, observation, deterministic: bool = False):
        features = self.extract_features(observation)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        action_mask = observation.get("action_mask", None)
        distribution = self._apply_action_mask(distribution, action_mask)

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

        # if mean_reward > self.best_mean_reward:
        self.best_mean_reward = mean_reward
        import os
        model_name = f"ppo_connectfour_best_{mean_reward:.3f}.zip"
        self.model.save(os.path.join('checkpoints', model_name))
        send_telegram(f"新最佳模型\nMean(step1000)={mean_reward:.3f} WinRate={win_rate:.3f}")
        print(f"[Saved] {model_name}")
        cmd = ['uv',
        'run',
        'dump_weight.py',
        '--model_path',
        'checkpoints/'+model_name,
        '--output',
        'checkopponents/dump_weight.py']
        cmd = ' '.join(cmd)
        try:
            import subprocess
            print(f'cmd {cmd}')
            subprocess.run(cmd , check=True, shell=True, capture_output=True)
            # update opponent_list
            self.training_env.env_method('update_opponents')
        except Exception as e:
            print(f"Error occurred while dumping weights: {e}")
            raise Exception("Weight dumping failed")




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
    parser.add_argument('--total_step', default=1000, type=int, help='total_step to train')
    parser.add_argument('--num_cpu', default=8, type=int, help='cpu cores')
    parser.add_argument('--eval_freq', default=1001, type=int, help='eval_freq')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_steps', default=2048, type=int, help='n_steps')
    parser.add_argument('--n_epochs', default=20, type=int, help='n_epochs')
    parser.add_argument('--ent_coef', default=0.01, type=float, help='ent_coef')
    parser.add_argument('--vf_coef', default=0.5, type=float, help='vf_coef')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')

    args = parser.parse_args()
    model_path = args.model
    num_cpu = int(args.num_cpu)
    learning_rate = float(args.lr)
    n_steps = int(args.n_steps) // num_cpu
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    ent_coef = float(args.ent_coef)
    vf_coef = float(args.vf_coef)
    # 註冊環境
    register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')
    
    # 設置參數

    total_timesteps = args.total_step
    print(f"使用 {num_cpu} 個並行環境進行訓練")
    
    # 創建多進程向量化環境
    if num_cpu == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = PPO(
        DictTransformerPolicy,
        env,
        learning_rate=learning_rate,  # 提高學習率
        n_steps=n_steps,
        batch_size=batch_size,      # 增大 batch
        n_epochs=n_epochs,         # 增加 epoch
        ent_coef=ent_coef,       # 增加熵係數鼓勵探索
        vf_coef=vf_coef,         # 價值損失權重
        policy_kwargs=dict(net_arch=[256, 256, 128]),  # 更深網路
    )
    
    # 創建改進的回調
    callback = EvaluationCallback(verbose=1, eval_freq=args.eval_freq)

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
    visualize_model(model, num_episodes=5)

if __name__ == '__main__':
    main()
