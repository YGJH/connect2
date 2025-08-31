import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy  # 如果需要

# 定義 standalone PyTorch 模型（基於你提供的 ConnectFourExtractor 和 policy 結構）
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

class ConnectFourExtractor(nn.Module):
    def __init__(self, features_dim: int = 256):
        super().__init__()
        self.height = 6
        self.width = 7
        n_channels = 256
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
        n_flatten = n_channels * self.height * self.width
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

class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = ConnectFourExtractor(features_dim=256)
        self.pi_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(128, 7)  # 輸出 7 個動作的 logits

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi = self.pi_net(features)
        logits = self.action_net(latent_pi)
        action_mask = obs["action_mask"]
        logits = logits + (1 - action_mask) * -1e9  # Mask 無效動作
        return logits


def main():


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='ppo_connectfour.zip')
    parser.add_argument('--output', default='submission.py')

    args = parser.parse_args()
    # 加載你的 PPO 模型（從 .zip 檔）
    trained_model = PPO.load(args.model_path)  # 替換成你的 .zip 檔路徑

    # 創建 standalone 模型
    standalone_model = ConnectFourPolicy()

    # 複製權重（假設你的 trained_model.policy 是 CustomAlphaZeroPolicy 或 ActorCriticPolicy）
    # 注意：如果你的 policy 結構不同，請調整對應的鍵
    standalone_model.features_extractor.load_state_dict(trained_model.policy.features_extractor.state_dict())
    standalone_model.pi_net.load_state_dict(trained_model.policy.mlp_extractor.policy_net.state_dict())
    standalone_model.action_net.load_state_dict(trained_model.policy.action_net.state_dict())

    # 保存 state_dict 到 bytes，並 base64 編碼
    buffer = io.BytesIO()
    torch.save(standalone_model.state_dict(), buffer)
    buffer.seek(0)
    encoded_weights = base64.b64encode(buffer.read()).decode('utf-8')

    # 現在 encoded_weights 就是你的嵌入權重字串，複製它用在下一步
    # print(encoded_weights)  # 輸出後複製這個長字串



    agent_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
import warnings
warnings.filterwarnings("ignore")

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

class ConnectFourExtractor(nn.Module):
    def __init__(self, features_dim: int = 256):
        super().__init__()
        self.height = 6
        self.width = 7
        n_channels = 256
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
        n_flatten = n_channels * self.height * self.width
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

class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = ConnectFourExtractor(features_dim=256)
        self.pi_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(128, 7)  # 輸出 7 個動作的 logits

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi = self.pi_net(features)
        logits = self.action_net(latent_pi)
        action_mask = obs["action_mask"]
        logits = logits + (1 - action_mask) * -1e9  # Mask 無效動作
        return logits


# 嵌入的 base64 權重（從步驟 2 複製過來）
ENCODED_WEIGHTS = "{encoded_weights}"
# 加載模型
model = ConnectFourPolicy()
if ENCODED_WEIGHTS:
    bytes_io = io.BytesIO(base64.b64decode(ENCODED_WEIGHTS))
    model.load_state_dict(torch.load(bytes_io, map_location=torch.device('cpu')))
model.eval()
def agent(observation, configuration):
    # 從 Kaggle 環境構建 obs
    board = np.array(observation['board'], dtype=np.float32).reshape(6, 7)  # 注意：board 是 list of 42，需 reshape
    mark = np.array([observation['mark']], dtype=np.float32)

    # 計算 action_mask：檢查每列頂部是否為 0
    action_mask = np.zeros(7, dtype=np.float32)
    for col in range(7):
        if board[0, col] == 0:  # 頂部是 row 0（假設 row-major，從上到下）
            action_mask[col] = 1.0
    
    obs_dict = {{
        "board": board.flatten(),  # 展平回 42
        "mark": mark,
        "action_mask": action_mask
    }}
    
    # 轉成 torch tensor，加 batch dim
    obs = {{k: torch.from_numpy(np.array(v)).unsqueeze(0) for k, v in obs_dict.items()}}
    
    with torch.no_grad():
        logits = model(obs)
        action = torch.argmax(logits, dim=1).item()
    
    # 確保 action 是有效的（如果 mask 沒用，fallback）
    if action_mask[action] == 0:
        valid_actions = np.where(action_mask == 1)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
    
    return int(action)
    '''


    with open(args.output, 'w') as f:
        f.write(agent_code)

if __name__ == '__main__':
    main()