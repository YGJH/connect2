import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
import io
import base64
import argparse
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# DictTransformerExtractor class (matches training code with num_layers=12)
class DictTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, d_model=128, num_layers=12, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[256, 256, 128]):
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
        x += self.pos_encoding
        
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

# DictTransformerPolicy (for loading PPO model)
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
                cnn_channels=[256, 256, 128]
            ),
            **kwargs
        )

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf, latent_sde = self._get_latent(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            action_mask = action_mask.to(distribution.distribution.logits.device)
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0
            logits = distribution.distribution.logits
            logits = logits + (action_mask - 1) * 1e9
            distribution.distribution.logits = logits
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
    # Debug: Print state_dict keys to verify
    print("State dict keys:", list(policy.state_dict().keys()))
    # Filter state dict to include only policy-related weights
    full_state_dict = policy.state_dict()
    filtered_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('value_net') or key.startswith('vf_features_extractor'):
            continue
        if key.startswith('pi_features_extractor.'):
            new_key = key.replace('pi_features_extractor.', 'features_extractor.')
            filtered_state_dict[new_key] = value
            continue
        if key.startswith('mlp_extractor.policy_net.'):
            new_key = key.replace('mlp_extractor.policy_net.', 'policy_net.')
            filtered_state_dict[new_key] = value
            continue
        if key.startswith('action_net.'):
            filtered_state_dict[key] = value
            continue
    # Debug: Print filtered state_dict keys
    print("Filtered state dict keys:", list(filtered_state_dict.keys()))
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
import torch.nn.functional as F
import numpy as np
import base64
import io

# DictTransformerExtractor class
class DictTransformerExtractor(nn.Module):
    def __init__(self, features_dim=256, d_model=128, num_layers=12, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[128, 128]):
        super().__init__()
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
        def positional_encoding(seq_len, d_model):
            pe = torch.zeros(seq_len, d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0)
        self.pos_encoding = nn.Parameter(positional_encoding(self.seq_len, d_model), requires_grad=False)
        
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
        board = observations["board"]
        mark = observations["mark"]
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
        x += self.pos_encoding
        
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

# Simple policy as nn.Module
class ConnectFourPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = DictTransformerExtractor(
            features_dim=256, d_model=128, num_layers=12, num_heads=8, height=6, width=7, cnn_channels=[128, 128]
        )
        dims = [256, 256, 128]  # Match net_arch=[128, 128] from main.py
        layers = []
        in_dim = 256
        for out_dim in dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(128, 7)  # Match output to 7 actions

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
    board = torch.tensor(observation['board'], dtype=torch.float).unsqueeze(0)
    mark = torch.tensor([observation['mark']], dtype=torch.float).unsqueeze(0)
    action_mask = np.zeros(configuration.columns, dtype=np.float32)
    board2d = board.view(6, 7)
    for col in range(7):
        if board2d[0, col] == 0:
            action_mask[col] = 1.0
    obs_dict = {{"board": board, "mark": mark}}
    with torch.no_grad():
        logits = model(obs_dict)[0]
    masked = logits.numpy() + (action_mask - 1) * 1e9
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