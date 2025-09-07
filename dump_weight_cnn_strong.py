import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy  # 如果需要
from gymnasium import spaces
from sb3_contrib import MaskablePPO

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
    def __init__(self, features_dim: int = 64):
        super().__init__()
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
        # from gymnasium import spaces
        

        with torch.no_grad():
            sample_obs = {
                "board": np.zeros((self.height * self.width,), dtype=np.float32),
                "mark": [0],
                "action_mask": np.ones((self.width,), dtype=np.float32),
            }
            sample_tensor = self._prepare_sample(sample_obs)
            n_flatten = self.cnn(sample_tensor).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def _prepare_sample(self, obs_dict):
        board = torch.tensor(obs_dict['board']).float().reshape(self.height, self.width)  # (6, 7)
        mark = torch.tensor(obs_dict['mark']).float()
        
        player_plane = (board == mark).float()
        opponent_mark = 3 - mark
        opponent_plane = (board == opponent_mark).float()
        turn_plane = (mark - 1.0).expand_as(board)
        
        stacked = torch.stack([player_plane, opponent_plane, turn_plane], dim=0)  # (5, 6, 7)
        return stacked.unsqueeze(0)  # (1, 5, 6, 7)
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


class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = ConnectFourExtractor(features_dim=64)
        self.pi_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(64, 7)  # 輸出 7 個動作的 logits

    def _is_winner(self, board, player):
        b = board
        H, W, C = 6, 7, 4
        for r in range(H):
            for c in range(W - C + 1):
                if np.all(b[r, c:c+C] == player):
                    return True
        for c in range(W):
            for r in range(H - C + 1):
                if np.all(b[r:r+C, c] == player):
                    return True
        for r in range(H - C + 1):
            for c in range(W - C + 1):
                if all(b[r+i, c+i] == player for i in range(C)):
                    return True
        for r in range(H - C + 1):
            for c in range(C - 1, W):
                if all(b[r+i, c-i] == player for i in range(C)):
                    return True
        return False

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi = self.pi_net(features)
        logits = self.action_net(latent_pi)
        action_mask = obs["action_mask"]
        logits = logits + (1 - action_mask) * -1e8  # Mask 無效動作
        return logits


def main():


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='ppo_connectfour.zip')
    parser.add_argument('--output', default='submission.py')

    args = parser.parse_args()
    # 加載你的 PPO 模型（從 .zip 檔）
    trained_model = MaskablePPO.load(args.model_path)

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
    def __init__(self, features_dim: int = 64):
        super().__init__()
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
        self.linear = nn.Sequential(
            nn.Linear(2688, features_dim),
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

class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = ConnectFourExtractor(features_dim=64)
        self.pi_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.action_net = nn.Linear(64, 7)  # 輸出 7 個動作的 logits


    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi = self.pi_net(features)
        logits = self.action_net(latent_pi)
        action_mask = obs["action_mask"]
        logits = logits + (1 - action_mask) * -1e8  # Mask 無效動作
        return logits

# 嵌入的 base64 權重（從步驟 2 複製過來）
ENCODED_WEIGHTS = "{encoded_weights}"
# 加載模型
model = ConnectFourPolicy()
if ENCODED_WEIGHTS:
    bytes_io = io.BytesIO(base64.b64decode(ENCODED_WEIGHTS))
    model.load_state_dict(torch.load(bytes_io, map_location=torch.device('cpu')))
model.eval()


def _is_winner(board, player):
    b = board
    H, W, C = 6, 7, 4
    for r in range(H):
        for c in range(W - C + 1):
            if np.all(b[r, c:c+C] == player):
                return True
    for c in range(W):
        for r in range(H - C + 1):
            if np.all(b[r:r+C, c] == player):
                return True
    for r in range(H - C + 1):
        for c in range(W - C + 1):
            if all(b[r+i, c+i] == player for i in range(C)):
                return True
    for r in range(H - C + 1):
        for c in range(C - 1, W):
            if all(b[r+i, c-i] == player for i in range(C)):
                return True
    return False


def _next_open_row(board, col):
    for r in range(6 - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return -1

def agent(observation, configuration):
    # 從 Kaggle 環境構建 obs
    board = np.array(observation['board'], dtype=np.float32).reshape(6, 7)  # (6,7)
    # mark 保持成 (1,1) (batch,1)
    mark = np.array([observation['mark']], dtype=np.float32).reshape(1, 1)  # (1,1)
    mark_scalar = int(mark[0, 0])

    # 先做簡單的 immediate win 檢查（使用 scalar）
    for c in range(7):
        if board[0, c] == 0:
            row = _next_open_row(board, c)
            board[row, c] = mark_scalar
            if _is_winner(board, mark_scalar):
                return int(c)
            board[row, c] = 0


    for c in range(7):
        if board[0, c] == 0:
            row = _next_open_row(board, c)
            board[row, c] = 3 - mark_scalar
            if _is_winner(board, 3 - mark_scalar):
                return int(c)
            board[row, c] = 0

    # action mask
    action_mask = np.zeros(7, dtype=np.float32)
    for col in range(7):
        if board[0, col] == 0:
            action_mask[col] = 1.0



    # 構建 obs：明確轉成 torch tensor 並控制各欄位 shape
    obs = {{
        "board": torch.from_numpy(board.flatten().astype(np.float32)).unsqueeze(0),        # (1,42)
        "mark": torch.from_numpy(mark.astype(np.float32)),                                # (1,1)
        "action_mask": torch.from_numpy(action_mask.astype(np.float32)).unsqueeze(0),     # (1,7)
    }}

    with torch.no_grad():
        logits = model(obs)
        action = torch.argmax(logits, dim=1).item()

    # fallback
    if action_mask[action] == 0:
        valid_actions = np.where(action_mask == 1)[0]
        action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 0

    return int(action)
'''


    with open(args.output, 'w') as f:
        f.write(agent_code)

if __name__ == '__main__':
    main()