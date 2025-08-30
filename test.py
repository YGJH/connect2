import torch
from stable_baselines3 import PPO

# Load your trained SB3 model
model = PPO.load("checkpoints/ppo_connectfour_best_-0.233")

# Extract the policy network
policy = model.policy
state_dict = policy.state_dict()

# Save the policy weights
torch.save(state_dict, "policy_weights.pth")

# Optional: Save as base64 for embedding (if file size is small enough)
import base64
with open("policy_weights.pth", "rb") as f:
    weights_data = f.read()
    weights_b64 = base64.b64encode(weights_data).decode('utf-8')
    # print(weights_b64)  # Copy this for embedding in submission.py

agent_code = f'''
import torch
import torch.nn as nn
import base64
import io

# Custom feature extractor (from your training script)
class DictTransformerExtractor(nn.Module):
    def __init__(self, features_dim=256, d_model=128, num_layers=42, num_heads=8, mark_emb_dim=8, height=6, width=7, cnn_channels=[126, 126]):
        super().__init__()
        self.height = height
        self.width = width
        self.seq_len = height * width
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # CNN backbone
        self.cnn_layers = nn.ModuleList()
        in_channels = 3  # One-hot for 0=empty, 1=player1, 2=player2
        for out_channels in cnn_channels:
            self.cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.cnn_out_dim = cnn_channels[-1]
        self.cnn_projection = nn.Linear(self.cnn_out_dim, d_model)
        
        # Mark embedding
        self.mark_embedding = nn.Embedding(2, mark_emb_dim)
        self.mark_projection = nn.Linear(mark_emb_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.01)
        
        # Transformer with residual blocks
        self.transformer_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(0, num_layers, 2):
            layers = []
            for _ in range(2):
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads, dim_feedforward=256,
                    dropout=0.2, batch_first=True, norm_first=True
                )
                layers.append(layer)
            self.transformer_blocks.append(nn.Sequential(*layers))
            self.norms.append(nn.LayerNorm(d_model))
        
        # Output layer
        self.out = nn.Linear(d_model, features_dim)

    def forward(self, observations):
        board = observations["board"]  # (B, 42)
        mark = observations["mark"]    # (B, 1)
        B = board.shape[0]
        
        # Reshape and one-hot encode
        board_int = board.long()
        board_onehot = torch.zeros(B, 3, self.height, self.width, device=board.device)
        board_onehot.scatter_(1, board_int.view(B, 1, self.height, self.width), 1)
        x = board_onehot  # (B, 3, 6, 7)
        
        # CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Flatten and project
        x = x.flatten(2).transpose(1, 2)  # (B, 42, cnn_out_dim)
        x = self.cnn_projection(x)  # (B, 42, d_model)
        
        # Mark embedding
        mark_idx = (mark.long() - 1).squeeze(-1)  # (B,)
        mark_emb = self.mark_embedding(mark_idx)  # (B, mark_emb_dim)
        mark_emb = self.mark_projection(mark_emb).unsqueeze(1)  # (B, 1, d_model)
        x += mark_emb  # Broadcast: (B, 42, d_model)
        
        # Positional encoding
        x += self.pos_encoding
        
        # Transformer blocks with residual and LayerNorm
        for block, norm in zip(self.transformer_blocks, self.norms):
            residual = x
            x = block(x)
            x = x + residual
            x = norm(x)
        
        # Global mean pooling
        x = x.mean(dim=1)  # (B, d_model)
        
        # Output
        return self.out(x)  # (B, features_dim)

# Simplified policy network without SB3
class DictTransformerPolicy(nn.Module):
    def __init__(self, action_dim=7, features_dim=256):
        super().__init__()
        self.features_extractor = DictTransformerExtractor(
            features_dim=features_dim,
            d_model=128,
            num_layers=42,
            num_heads=8,
            height=6,
            width=7,
            cnn_channels=[126, 126]
        )
        # MLP for action logits
        self.mlp_actor = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs):
        features = self.features_extractor(obs)
        logits = self.mlp_actor(features)
        
        # Apply action mask
        action_mask = obs.get("action_mask", None)
        if action_mask is not None:
            logits = logits + (action_mask - 1) * 1e9  # Mask invalid actions
        
        return logits

# Load model weights (replace with your base64 string or file path)
MODEL_WEIGHTS = "{weights_b64}"
# Decode and load weights
weights_data = base64.b64decode(MODEL_WEIGHTS)
weights_buffer = io.BytesIO(weights_data)
state_dict = torch.load(weights_buffer, map_location=torch.device('cpu'))

# Initialize policy and load weights
policy = DictTransformerPolicy(action_dim=7, features_dim=256)
policy.load_state_dict(state_dict)
policy.eval()  # Set to evaluation mode

def get_action_mask(board, config):
    """Generate action mask for valid moves (1 for valid, 0 for invalid)."""
    board = np.array(board).reshape(config.rows, config.columns)
    action_mask = np.zeros(config.columns, dtype=np.float32)
    for col in range(config.columns):
        if board[0, col] == 0:  # Top row is empty, column is valid
            action_mask[col] = 1.0
    return action_mask

def agent(obs, config):
    """Kaggle ConnectX agent function."""
    # Convert Kaggle observation to model-compatible format
    board = np.array(obs.board, dtype=np.int64).reshape(1, -1)  # (1, 42)
    mark = np.array([obs.mark], dtype=np.int64).reshape(1, 1)  # (1, 1)
    action_mask = get_action_mask(obs.board, config).reshape(1, -1)  # (1, 7)

    # Create observation dictionary
    observation = {{
        "board": torch.tensor(board, dtype=torch.int64),
        "mark": torch.tensor(mark, dtype=torch.int64),
        "action_mask": torch.tensor(action_mask, dtype=torch.float32)
    }}

    # Predict action
    with torch.no_grad():
        logits = policy(observation)
        action = torch.argmax(logits, dim=1).item()  # Deterministic action

    # Fallback to random valid action if predicted action is invalid
    valid_actions = np.where(action_mask[0] == 1)[0]
    if action_mask[0, action] == 0 and len(valid_actions) > 0:
        action = np.random.choice(valid_actions)

    return int(action)
'''

with open("submission.py", "w") as f:
    f.write(agent_code)