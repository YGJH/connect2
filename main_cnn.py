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
from connectFour import ConnectFourEnv  # 直接匯入類別
import multiprocessing as mp
import sys
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def make_env(rank, seed=0, render_mode=None, folder_path='checkopponents'):
    def _init():
        try:
            env = gym.make('ConnectFourEnv', render_mode=render_mode, folder_path=folder_path)
        except gym.error.NameNotFound:
            register(id='ConnectFourEnv', entry_point=ConnectFourEnv)  # 直接使用類別
            env = gym.make('ConnectFourEnv', render_mode=render_mode, folder_path=folder_path)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ConnectFourExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.height = 6
        self.width = 7
        n_channels = 64
        self.cnn = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            ResBlock(n_channels),
            ResBlock(n_channels),
            ResBlock(n_channels),
            ResBlock(n_channels),
            ResBlock(n_channels),
            ResBlock(n_channels),
            ResBlock(n_channels),
            nn.Flatten(),
        )
        # Compute flattened size dynamically
        with torch.no_grad():
            sample_obs = observation_space.sample()
            sample_tensor = self._prepare_sample(sample_obs)
            n_flatten = self.cnn(sample_tensor).shape[1]
        print(f'Flattened size: {n_flatten}')
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        board = observations['board'].reshape(-1, self.height, self.width)
        mark = observations['mark'][:, 0]
        
        player_plane = (board == mark[:, None, None]).float()
        opponent_plane = (board == (3 - mark)[:, None, None]).float()
        # 修復：使用 expand_as(board) 而非 expand(-1, ...)
        turn_plane = ((mark - 1).float()[:, None, None].expand_as(board))
        x = torch.stack([player_plane, opponent_plane, turn_plane], dim=1)
        x = self.cnn(x)
        x = self.linear(x)
        return x


    def _prepare_sample(self, obs_dict):
        board = torch.tensor(obs_dict['board']).float().reshape(self.height, self.width)  # (6, 7)
        mark = torch.tensor(obs_dict['mark']).float()
        
        player_plane = (board == mark).float()
        opponent_mark = 3 - mark
        opponent_plane = (board == opponent_mark).float()
        turn_plane = (mark - 1.0).expand_as(board)
        
        stacked = torch.stack([player_plane, opponent_plane, turn_plane], dim=0)  # (5, 6, 7)
        return stacked.unsqueeze(0)  # (1, 5, 6, 7)

class CustomAlphaZeroPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=ConnectFourExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[64, 64, 64], vf=[64, 64, 64]),
            activation_fn=nn.ReLU,
        )

    def get_distribution(self, obs):
        # 先拿 feature
        features = super().extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)


        distribution = self.action_dist.proba_distribution(action_logits=logits)
        return distribution



class ValueScatterCallback(BaseCallback):
    def __init__(self, check_freq=1, save_path="scatter", verbose=0):
        super(ValueScatterCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_rollout_end(self):
        rollout_buffer = self.model.rollout_buffer
        values = rollout_buffer.values.flatten()
        returns = rollout_buffer.returns.flatten()

        # correlation coefficient
        corr = np.corrcoef(values, returns)[0, 1]

        if self.n_calls % self.check_freq == 0:
            plt.figure(figsize=(6, 6))

            # 散點圖
            plt.scatter(returns, values, alpha=0.4, s=12, c="blue", label="Samples")

            # 理想線 y=x
            min_val = min(returns.min(), values.min())
            max_val = max(returns.max(), values.max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal y=x")

            # 軸範圍統一
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)

            plt.xlabel("Returns (target)")
            plt.ylabel("Value Predictions (critic)")
            plt.title(f"Value vs Returns\ncorr={corr:.3f}, step={self.num_timesteps}, n={len(values)}")
            plt.legend()
            plt.grid(True)

            save_file = os.path.join(self.save_path, f"value_scatter_{self.num_timesteps}.png")
            plt.savefig(save_file)
            plt.close()

            if self.verbose > 0:
                print(f"[ValueScatterCallback] Saved scatter plot to {save_file}, corr={corr:.3f}")
    def _on_step(self) -> bool:
        # 必須要有，不然不能被實例化
        return True


class EvaluationCallback(BaseCallback):
    def __init__(self, verbose=0, eval_freq=1000, save_path="scatter" , save_freq=5000, visualize_model=None, max_steps=1e6 , coef_fun=None):
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
        self.eval_count = 0
        self.last_eval_step = 0
        self.visualize_model = visualize_model

        os.makedirs(save_path, exist_ok=True)

        # entropy tracking
        self.max_steps = max_steps
        self.coef_fun = coef_fun

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs

    def _on_step(self):

        if self.n_calls % self.eval_freq == 0:
            model_path = "selfplay_model.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Updated self-play model path: {model_path}")

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


        self.eval_count += 1
        recent_rewards = self.rewards[-1000:]
        import numpy as np
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
        self.model.ent_coef = self.coef_fun(self.num_timesteps)

        for opponent, stats in self.opponent_stats.items():
            if stats['games'] > 0:
                opp_wr = stats['wins'] / stats['games']
                log_info += f"\n  - {opponent}: {opp_wr:.3f} ({stats['wins']}/{stats['games']})"

        print(log_info)
        import os
        model_name = f"ppo_connectfour_best_cnn_{mean_reward:.3f}.zip"
        self.model.save(os.path.join('checkpoints', model_name))
        if mean_reward > self.best_mean_reward:
            send_telegram(f"新最佳模型\nMean(step1000)={mean_reward:.3f} WinRate={win_rate:.3f}")
            self.best_mean_reward = mean_reward
        
        
        
        if self.eval_count > 49 and self.visualize_model is not None:
            print(f"[Saved] {os.path.join('checkpoints', model_name)}")
            most_saved_path = f'ppo_connectfour_best_cnn_{mean_reward:.3f}.py'
            cmd = [
                # 'uv run dump_weight_cnn.py',
                './dist/dump_weight_cnn/dump_weight_cnn',
                '--model_path',
                os.path.join('checkpoints', model_name),
                '--output',
                os.path.join('checkopponents', most_saved_path),
            ]
            cmd_strong = [
                # 'uv run dump_weight_cnn.py',
                './dist/dump_weight_cnn_strong/dump_weight_cnn_strong',
                '--model_path',
                os.path.join('checkpoints', model_name),
                '--output',
                os.path.join('checkopponents', most_saved_path),
            ]


            import gc
            gc.collect()
            cmd = ' '.join(cmd)
            cmd_strong = ' '.join(cmd_strong)
            if os.path.exists('checkopponents'):
                fs = [f for f in os.listdir('checkopponents') if f.endswith('.py')]
                # if len(fs) > 5:
                import random
                live = random.sample(fs, 5)
                for f in os.listdir('checkopponents'):
                    if f not in live or not f.endswith('.py'):
                        print(f'remove {f}')
                        os.remove(os.path.join('checkopponents', f))

            self.opponent_stats.clear()
            try:
                import numpy as np
                import subprocess
                if np.random.rand() < 0.7:
                    print()
                    subprocess.run(cmd_strong , shell=True, capture_output=True)
                else:
                    subprocess.run(cmd , shell=True, capture_output=True)
                # update opponent_list
                self.training_env.env_method('update_opponents', most_saved=most_saved_path)
            except Exception as e:
                print(f"Error occurred while dumping weights: {e}")
                raise Exception("Weight dumping failed")



            self.visualize_model(self.model, num_episodes=1)
            self.eval_count = 0


    def _on_training_end(self):
        self._evaluate_and_log()
        final_name = "ppo_connectfour_cnn_final.zip"
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
    env = None
    try:
        env = gym.make('ConnectFourEnv', render_mode='human', folder_path='checkopponents')
        for ep in range(num_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            print(f"Visualization Episode {ep+1}")
            print(f"Initial observation: keys={obs.keys()}, board_shape={obs['board'].shape}, mark_shape={obs['mark'].shape}")
            while not (terminated or truncated):

                action, _ = model.predict(obs, deterministic=True)  # Pass the full dict obs here
                action = int(action.item())
                mask = obs["action_mask"]
                valid_actions = np.where(mask == 1)[0]
                if mask[action] == 0:
                    if len(valid_actions) > 0:
                        action = int(np.random.choice(valid_actions))
                        print(f"[visualize_model] Invalid action {action} predicted, using random valid action={action}")
                    else:
                        print("[visualize_model] No valid actions, terminating")
                        terminated = True
                        break
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                env.render()
                print(f"[visualize_model] Step {steps}: Action={action}, Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")
                time.sleep(0.5)
            print(f"Episode {ep+1} reward={total_reward} steps={steps}")
    except Exception as e:
        print(f"[visualize_model] Error: {e}")

    finally:
        if env is not None:
            try:
                env.close()  # 確保總是關閉
            except Exception as close_error:
                print(f"[visualize_model] Error closing env: {close_error}")
            finally:
                del env  # 明確刪除引用
            print("[visualize_model] Environment closed.")

def warmup_cosine_annealing_schedule(initial_value: float, total_timesteps: int, warmup_fraction: float = 0.03, min_lr: float = 1e-8):
    """
    Returns a scheduler function for stable-baselines3 that applies:
      - linear warmup from 0 -> initial_value over warmup_steps
      - cosine annealing from initial_value -> min_lr over remaining steps

    initial_value: starting lr after warmup peak
    total_timesteps: total training timesteps (used to compute warmup length)
    warmup_fraction: fraction of total_timesteps used for warmup (0..1)
    min_lr: lower bound for learning rate
    """
    warmup_steps = max(1, int(total_timesteps * float(warmup_fraction)))

    def lr_fn(progress_remaining: float) -> float:
        # SB3 passes progress_remaining that goes from 1.0 -> 0.0 during training
        steps_done_fraction = 1.0 - progress_remaining
        # Convert fraction to absolute step index (approximate)
        approx_step = int(steps_done_fraction * total_timesteps)

        if approx_step < warmup_steps:
            # linear warmup from min_lr to initial_value
            t = approx_step / float(warmup_steps)
            lr = min_lr + t * (initial_value - min_lr)
            return max(lr, min_lr)
        else:
            # cosine annealing on remaining steps
            remaining_steps = total_timesteps - warmup_steps
            if remaining_steps <= 0:
                return max(min_lr, initial_value)
            t = (approx_step - warmup_steps) / float(remaining_steps)
            # cosine from 0 -> pi: value goes initial_value -> min_lr
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
            lr = min_lr + (initial_value - min_lr) * cos_factor
            return max(lr, min_lr)

    return lr_fn

def linear_schedule(number, min_bound=1e-6):
    """
    Returns a scheduler function for stable-baselines3 that applies:
      - linear warmup from 0 -> initial_value over warmup_steps
      - cosine annealing from initial_value -> min_lr over remaining steps

    initial_value: starting lr after warmup peak
    total_timesteps: total training timesteps (used to compute warmup length)
    warmup_fraction: fraction of total_timesteps used for warmup (0..1)
    min_lr: lower bound for learning rate
    """

    def lr_fn(progress_remaining: float) -> float:
        return number * (0.1 + 0.9 * progress_remaining)

        # return max(number * (0.1 + 0.9 * progress_remaining), min_bound)

    return lr_fn

def linear_schedule(number, min_bound=1e-6):
    """
    Returns a scheduler function for stable-baselines3 that applies:
      - linear warmup from 0 -> initial_value over warmup_steps
      - cosine annealing from initial_value -> min_lr over remaining steps

    initial_value: starting lr after warmup peak
    total_timesteps: total training timesteps (used to compute warmup length)
    warmup_fraction: fraction of total_timesteps used for warmup (0..1)
    min_lr: lower bound for learning rate
    """

    def lr_fn(progress_remaining: float) -> float:
        return number * (0.1 + 0.9 * progress_remaining)

        # return max(number * (0.1 + 0.9 * progress_remaining), min_bound)

    return lr_fn


def cosine_annealing_schedule(total_timesteps: int, end_coef: float, start_coef: float, min_ent: float = 0.01):
    def annealine_fun(step):
        progress = min(step / total_timesteps, 1.0)
        ret_ent = end_coef + 0.5 * (start_coef - end_coef) * (
            1 + np.cos(np.pi * progress)
        )
        return max(ret_ent, min_ent)
    return annealine_fun


# This is the policy_function; it will be set on the env(s)
def agent_policy_function(model, vec_normalize, obs, config):
    gym_obs = kaggle_to_gym_obs(obs, config)
    normalized_obs = vec_normalize.normalize_obs(gym_obs)
    action, _ = model.predict(normalized_obs, deterministic=False)
    return int(action.item())  # Return scalar int action




def main():
    import connectFour
    register(id='ConnectFourEnv', entry_point='connectFour:ConnectFourEnv')  # Adjust if needed

    print("\033[1;32m")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, help='Path to the output file (optional).')
    parser.add_argument('--total_step', default=1000000, type=int, help='total_step to train')
    parser.add_argument('--num_cpu', default=5, type=int, help='cpu cores')
    parser.add_argument('--eval_freq', default=1001, type=int, help='eval_freq')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_steps', default=1000, type=int, help='n_steps')
    parser.add_argument('--n_epochs', default=20, type=int, help='n_epochs')
    parser.add_argument('--ent_coef', default=0.05, type=float, help='ent_coef')
    parser.add_argument('--vf_coef', default=1.0, type=float, help='vf_coef')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    parser.add_argument('--end_coef', default=0.01, type=float, help='end_coef')
    args, _ = parser.parse_known_args()
    model_path = args.model
    num_cpu = int(args.num_cpu)
    learning_rate = float(args.lr)
    n_steps = int(args.n_steps) // num_cpu
    batch_size = int(args.batch_size)
    n_epochs = int(args.n_epochs)
    ent_coef = float(args.ent_coef)
    vf_coef = float(args.vf_coef)
    total_step = int(args.total_step)
    end_coef = float(args.end_coef)

    print(f"Model path: {model_path}")
    print(f'total_step: {total_step}')
    print(f"Number of CPU cores: {num_cpu}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of steps: {n_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Entropy coefficient: {ent_coef}")
    print(f"Value function coefficient: {vf_coef}")
    print(f"End entropy coefficient: {end_coef}")



    os.makedirs('checkpoints', exist_ok=True)
    if os.path.exists('checkopponents'):
        fs = [f for f in os.listdir('checkopponents') if f.endswith('.py')]
        if len(fs) > 5:
            import random
            live = random.sample(fs, 5)
            for f in fs:
                if f not in live:
                    print(f'remove {f}')
                    os.remove(os.path.join('checkopponents', f))


    print(f"Training with {num_cpu} parallel environments")

    if num_cpu == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)

    model = MaskablePPO(
        CustomAlphaZeroPolicy,
        env,
        learning_rate=warmup_cosine_annealing_schedule(initial_value=learning_rate, total_timesteps=total_step, warmup_fraction=0.03, min_lr=1e-6),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=0.2,           # 確保裁剪範圍適當
        clip_range_vf=0.2,        # 添加 value function 裁剪
        max_grad_norm=0.5,        # 梯度裁剪
        target_kl=0.04,           # 目標 KL 散度
        gamma = 0.99,
        gae_lambda = 0.95,
        verbose=1,
        tensorboard_log="/tmp/sb3_logs/",
    )

    if model_path:
        model.load(model_path)
        print(f"Loaded model: {model_path}")

    print("Starting training...")
    model.save('temp.zip')
    start_time = time.time()


    callback = EvaluationCallback(
        verbose=1,
        eval_freq=args.eval_freq,
        visualize_model=visualize_model,
        coef_fun=cosine_annealing_schedule(
            total_timesteps=total_step,
            end_coef=end_coef,
            start_coef=ent_coef
        ),
        max_steps=total_step
    )

    import psutil

    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage before training: {mem_usage:.2f} MB")

    scatter_callback = ValueScatterCallback(check_freq=5, save_path="scatter", verbose=1)


    model.learn(
        total_timesteps=total_step,
        callback=[callback, scatter_callback],
        progress_bar=True,
        tb_log_name="ppo_cartpole_r1"
    )
    print(f"Memory usage after training: {mem_usage:.2f} MB")
    print(f"Training completed! Time taken: {time.time() - start_time:.2f} seconds")
    env.close()

    print("Starting visualization test...")
    visualize_model(model, num_episodes=10)

if __name__ == '__main__':
    # Support frozen executables (PyInstaller) and avoid forkserver spawn issues
    try:
        mp.freeze_support()
    except Exception:
        pass
    try:
        # Force fork start method to avoid forkserver/spawn introducing extra CLI flags
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # start method already set; ignore
        pass
    main()