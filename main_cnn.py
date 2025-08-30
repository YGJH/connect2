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




def make_env(rank, seed=0, render_mode=None):
    def _init():
        try:
            register(id='ConnectFour-v0', entry_point='connectFour:ConnectFourEnv')  # Adjust entry_point if needed
            env = gym.make('ConnectFour-v0', render_mode=render_mode)
        except gym.error.NameNotFound:
            env = gym.make('ConnectFour-v0', render_mode=render_mode)
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
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
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
            ResBlock(n_channels),  # Add one more for better capacity
            nn.Flatten(),
        )
        # Compute flattened size dynamically
        with torch.no_grad():
            sample_obs = observation_space.sample()
            sample_tensor = self._prepare_sample(sample_obs)
            n_flatten = self.cnn(sample_tensor).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        board = observations['board'].reshape(-1, self.height, self.width)
        mark = observations['mark'][:, 0]  # (batch,)
        player_plane = (board == mark[:, None, None]).float()
        opponent_mark = 3 - mark
        opponent_plane = (board == opponent_mark[:, None, None]).float()
        turn_plane = ((mark - 1).float()[:, None, None].expand(-1, self.height, self.width))  # 0.0 if mark=1, 1.0 if mark=2
        x = torch.stack([player_plane, opponent_plane, turn_plane], dim=1)  # (batch, 3, height, width)
        x = self.cnn(x)
        x = self.linear(x)
        return x

    def _prepare_sample(self, obs_dict):
        board = torch.tensor(obs_dict['board']).float().reshape(1, self.height, self.width)
        mark = torch.tensor(obs_dict['mark']).float() # scalar for sample
        # print(mark)
        player_plane = (board == mark).float()
        opponent_mark = 3 - mark
        opponent_plane = (board == opponent_mark).float()
        turn_plane = torch.full_like(board, (mark - 1).item())
        return torch.stack([player_plane, opponent_plane, turn_plane], dim=1)  # (1, 3, height, width)

class CustomAlphaZeroPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=ConnectFourExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.ReLU,
        )

    def get_distribution(self, obs):
        features = super().extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        action_mask = obs["action_mask"]
        logits = logits + (1 - action_mask) * -1e9  # Mask invalid actions by setting logits to -inf
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        return distribution

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
    env = gym.make('ConnectFour-v0', render_mode='human')
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            print(f"Visualization Episode {ep+1}")
            print(f"Initial observation: keys={obs.keys()}, board_shape={obs['board'].shape}, mark_shape={obs['mark'].shape}, action_mask_shape={obs['action_mask'].shape}")
            while not (terminated or truncated):
                if not isinstance(obs, dict):
                    raise ValueError(f"Expected dict observation, got {type(obs)}")
                assert "board" in obs and "mark" in obs and "action_mask" in obs, \
                    f"Invalid observation keys: {obs.keys()}"
                assert obs["board"].shape == (42,), f"Invalid board shape: {obs['board'].shape}"
                assert obs["mark"].shape == (1,), f"Invalid mark shape: {obs['mark'].shape}"
                assert obs["action_mask"].shape == (7,), f"Invalid action_mask shape: {obs['action_mask'].shape}"

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
    finally:
        env.close()
        time.sleep(1)

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

    register(id='ConnectFour-v0', entry_point='__name__:ConnectFourEnv')  # Adjust if needed

    print(f"Training with {num_cpu} parallel environments")

    if num_cpu == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    model = PPO(
        CustomAlphaZeroPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1,
    )

    callback = EvaluationCallback(verbose=1, eval_freq=args.eval_freq)

    if model_path:
        model.load(model_path)
        print(f"Loaded model: {model_path}")

    print("Starting training...")
    start_time = time.time()

    model.learn(
        total_timesteps=args.total_step,
        callback=callback,
        progress_bar=True
    )

    print(f"Training completed! Time taken: {time.time() - start_time:.2f} seconds")
    env.close()

    print("Starting visualization test...")
    visualize_model(model, num_episodes=10)

if __name__ == '__main__':
    main()