import torch
import torch.nn as nn
from stable_baselines3 import PPO
import io
import base64
import argparse
import os

# DictTransformerExtractor class (matches training code, hard-coded board_dim=42)
class DictTransformerExtractor(nn.Module):
    def __init__(self, observation_space=None, features_dim=128, d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
        super().__init__()
        board_dim = 42  # 6x7 Connect Four board
        self.d_model = d_model
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
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

# ConnectFourPolicy class (matches DictTransformerPolicy from training)
class ConnectFourPolicy(nn.Module):
    def __init__(self, observation_space=None, action_space=None, net_arch=[128, 128], features_dim=128,
                 d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
        super().__init__()
        action_dim = 7  # Connect Four has 7 columns
        self.features_extractor = DictTransformerExtractor(
            observation_space=observation_space, features_dim=features_dim, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads, mark_emb_dim=mark_emb_dim
        )
        self.mlp_extractor = nn.ModuleDict({
            'policy_net': nn.Sequential(
                nn.Linear(features_dim, net_arch[0]),
                nn.ReLU(),
                nn.Linear(net_arch[0], net_arch[1]),
                nn.ReLU()
            )
        })
        self.action_net = nn.Linear(net_arch[-1], action_dim)

    def forward(self, observations):
        features = self.features_extractor(observations)
        latent_pi = self.mlp_extractor.policy_net(features)
        return self.action_net(latent_pi)

def dump_policy_to_py(model_path, output_path):
    # Load PPO model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    policy = model.policy

    # Filter state_dict to include only policy-related keys and map prefixes
    full_state_dict = policy.state_dict()
    filtered_state_dict = {}
    for key, value in full_state_dict.items():
        if 'value_net' in key or 'vf_features_extractor' in key:
            continue
        
        # Map pi_features_extractor to features_extractor
        if key.startswith('pi_features_extractor.'):
            new_key = key.replace('pi_features_extractor.', 'features_extractor.')
            filtered_state_dict[new_key] = value
        # Map numerical indices to named layers
        elif key == 'mlp_extractor.policy_net.0.weight':
            filtered_state_dict['mlp_extractor.policy_net.0.weight'] = value
        elif key == 'mlp_extractor.policy_net.0.bias':
            filtered_state_dict['mlp_extractor.policy_net.0.bias'] = value
        elif key == 'mlp_extractor.policy_net.2.weight':
            filtered_state_dict['mlp_extractor.policy_net.2.weight'] = value
        elif key == 'mlp_extractor.policy_net.2.bias':
            filtered_state_dict['mlp_extractor.policy_net.2.bias'] = value
        else:
            filtered_state_dict[key] = value


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
import base64
import io

# DictTransformerExtractor class
class DictTransformerExtractor(nn.Module):
    def __init__(self, observation_space=None, features_dim=128, d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
        super().__init__()
        board_dim = 42  # 6x7 Connect Four board
        self.d_model = d_model
        assert d_model % num_heads == 0, f"d_model ({{d_model}}) must be divisible by num_heads ({{num_heads}})"
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

# ConnectFourPolicy class
class ConnectFourPolicy(nn.Module):
    def __init__(self, observation_space=None, action_space=None, net_arch=[128, 128], features_dim=128,
                 d_model=64, num_layers=14, num_heads=16, mark_emb_dim=8):
        super().__init__()
        action_dim = 7  # Connect Four has 7 columns
        self.features_extractor = DictTransformerExtractor(
            observation_space=observation_space, features_dim=features_dim, d_model=d_model,
            num_layers=num_layers, num_heads=num_heads, mark_emb_dim=mark_emb_dim
        )
        self.mlp_extractor = nn.ModuleDict({{
            'policy_net': nn.Sequential(
                nn.Linear(features_dim, net_arch[0]),
                nn.ReLU(),
                nn.Linear(net_arch[0], net_arch[1]),
                nn.ReLU()
            )
        }})
        self.action_net = nn.Linear(net_arch[-1], action_dim)

    def forward(self, observations):
        features = self.features_extractor(observations)
        latent_pi = self.mlp_extractor.policy_net(features)
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

# Agent function for Kaggle ConnectX
def agent(observation, configuration):
    try:
        # Convert observation to tensors
        board = torch.tensor(observation['board'], dtype=torch.float).unsqueeze(0)  # Shape: (1, 42)
        mark = torch.tensor([observation['mark']], dtype=torch.float).unsqueeze(0)  # Shape: (1, 1)
        obs_dict = {{"board": board, "mark": mark}}
        
        # Get action logits
        with torch.no_grad():
            logits = model(obs_dict)[0]  # Shape: (7,)
        
        # Mask invalid actions (columns that are full)
        board_2d = torch.tensor(observation['board']).reshape(6, 7)
        valid_actions = [i for i in range(7) if board_2d[0, i] == 0]
        if not valid_actions:
            return 0  # Fallback to column 0 if no valid actions
        
        # Apply mask to logits
        masked_logits = torch.full_like(logits, float('-inf'))
        for action in valid_actions:
            masked_logits[action] = logits[action]
        
        # Select action with highest logit
        action = torch.argmax(masked_logits).item()
        
        # Ensure action is valid
        if action not in valid_actions:
            action = valid_actions[0]  # Fallback to first valid action
        return action
    except Exception as e:
        print(f"Agent error: {{e}}")
        return 0  # Fallback action
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
        raise ValueError(f"Failed to verify agent.py: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump PPO policy to self-contained agent.py for Kaggle ConnectX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .zip model file")
    parser.add_argument("--output", type=str, default="agent.py", help="Output path for the agent.py file")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    dump_policy_to_py(args.model_path, args.output)