import torch
import torch.nn as nn
from stable_baselines3 import PPO
import io
import base64
import argparse
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# DictTransformerExtractor class (matches training code, hard-coded board_dim=42)
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
            features_extractor_kwargs=dict(features_dim=128, d_model=16*24, num_layers=14, num_heads=16),
            **kwargs
        )

    def forward(self, obs, deterministic: bool = False):
        """
        obs: dict of tensors (已由 SB3 處理成 torch.Tensor)
        回傳: actions, values, log_prob (與 ActorCriticPolicy 約定一致)
        """
        # 取得 latent
        features = self.extract_features(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # 取得 action mask (1=合法, 0=非法)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(distribution.distribution.logits.device)
            # 若整行全 0，避免 -inf 全部變 NaN → fallback 全設為 1
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0

            # 將非法行動 logits 大幅降權
            # distribution.distribution.logits shape: (batch, n_actions)
            logits = distribution.distribution.logits
            logits = logits + (action_mask - 1) * 1e9
            distribution.distribution.logits = logits

        # 取動作
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob
    def _get_latent(self, features: torch.Tensor):
        """
        拷貝自 ActorCriticPolicy，unpack MlpExtractor 的 policy/value latent。
        """
        # mlp_extractor 回傳 (latent_pi, latent_vf)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # 如果不用 SDE，可以回 None
        latent_sde = None
        return latent_pi, latent_vf, latent_sde

    def _predict(self, observation, deterministic: bool = False):
        """
        SB3 用來在 rollout 時快速取得動作。
        確保與 forward 一致（只取動作，不回傳 value/log_prob）。
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

def dump_policy_to_py(model_path, output_path):
    # Load PPO model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    policy = model.policy

    # 只保留 policy 相關參數，並改 key 以匹配 submission.py 裡的 nn.Module
    full_state_dict = policy.state_dict()
    filtered_state_dict = {}
    for key, value in full_state_dict.items():
        # 跳過 value net 及其 extractor
        if key.startswith('value_net') or key.startswith('vf_features_extractor'):
            continue

        # features extractor
        if key.startswith('pi_features_extractor.'):
            new_key = key.replace('pi_features_extractor.', 'features_extractor.')
            filtered_state_dict[new_key] = value
            continue

        # policy mlp layers: mlp_extractor.policy_net.X → policy_net.X
        if key.startswith('mlp_extractor.policy_net.'):
            new_key = key.replace('mlp_extractor.policy_net.', 'policy_net.')
            filtered_state_dict[new_key] = value
            continue

        # action head
        if key.startswith('action_net.'):
            filtered_state_dict[key] = value
            continue
    # Save filtered state_dict to bytes buffer
    buffer = io.BytesIO()
    torch.save(filtered_state_dict, buffer)
    buffer.seek(0)
    state_dict_bytes = buffer.read()

    # Base64 encode the state dict
    try:
        base64_encoded = base64.b64encode(state_dict_bytes).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode state_dict to base64: {e}")

    # Generate agent.py content
    agent_code = f"""\
# Kaggle ConnectX agent generated from PPO model
import torch
import torch.nn as nn
import numpy as np
import base64
import io

# DictTransformerExtractor class
class DictTransformerExtractor(nn.Module):
    def __init__(self, features_dim=128, d_model=128, num_layers=4, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[16, 32]):
        super().__init__()
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

# Simple policy as nn.Module
class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # extractor 與訓練時一樣
        self.features_extractor = DictTransformerExtractor(
            features_dim=128, d_model=128, num_layers=4, num_heads=8, height=6, width=7, cnn_channels=[16, 32]
        )
        # net_arch=[512]*6
        dims = [512] * 6
        layers = []
        in_dim = 128
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.policy_net = nn.Sequential(*layers)
        # action head: 512 → 7
        self.action_net = nn.Linear(512, 7)

    def forward(self, observations):
        features = self.features_extractor(observations)
        latent_pi = self.policy_net(features)
        return self.action_net(latent_pi)
# Load model weights
try:
    model = ConnectFourPolicy()
    base64_data = \"\"\"{base64_encoded}\"\"\"
    state_dict_bytes = base64.b64decode(base64_data)
    buffer = io.BytesIO(state_dict_bytes)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model weights: {{e}}")

# Kaggle ConnectX agent function
def agent(observation, configuration):
    # Prepare obs dict
    board = torch.tensor(observation['board'], dtype=torch.float).unsqueeze(0)
    mark = torch.tensor([observation['mark']], dtype=torch.float).unsqueeze(0)
    action_mask = np.zeros(configuration.columns, dtype=np.float32)
    # build mask
    board2d = board.view(6,7)
    for col in range(7):
        if board2d[0, col] == 0:
            action_mask[col] = 1.0
    obs_dict = {{"board": board, "mark": mark}}

    # forward
    with torch.no_grad():
        logits = model(obs_dict)[0]  # (7,)
    # mask invalid
    masked = logits.numpy() + (action_mask - 1)*1e9
    action = int(masked.argmax())
    return action
"""

    # Write to output file
    try:
        with open(output_path, 'w') as f:
            f.write(agent_code)
        print(f"Generated {output_path} with embedded policy weights.")
    except Exception as e:
        raise IOError(f"Failed to write to {output_path}: {e}")

    # Verify the generated file
    try:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("agent_module", output_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["agent_module"] = module
        spec.loader.exec_module(module)
        if not hasattr(module, 'agent'):
            raise ValueError("Generated agent.py does not contain an 'agent' function")
        # Test loading the model
        model = module.ConnectFourPolicy()
        state_dict_bytes = base64.b64decode(base64_encoded)
        buffer = io.BytesIO(state_dict_bytes)
        state_dict = torch.load(buffer, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("Verified: agent.py contains a callable 'agent' function and model loads successfully.")
    except Exception as e:
        raise ValueError(f"Failed to verify: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump PPO policy to self-contained agent.py for Kaggle ConnectX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .zip model file")
    parser.add_argument("--output", type=str, default="agent.py", help="Output path for the agent.py file")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    dump_policy_to_py(args.model_path, args.output)