# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Basic building blocks
# -------------------------------

class SqueezeAndExcitation(nn.Module):
    """
    Standard Squeeze-and-Excitation block.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)       # [B, C]
        y = self.fc(y).view(b, c, 1, 1)        # [B, C, 1, 1]
        return x * y

class ResBlock(nn.Module):
    """
    A residual block with two 3x3 convolutions and a squeeze-and-excitation layer.
    """
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.se = SqueezeAndExcitation(channels, reduction=16)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + residual

class DoubleConeBlock(nn.Module):
    """
    The "double cone" block:
      1. Downsampling with a 4x4 conv (stride=4) and GELU.
      2. A sequence of ResBlocks.
      3. Two consecutive upsampling conv-transpose layers.
      4. A skip connection adds the block input.
    """
    def __init__(self, channels=128, num_resblocks=6):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=4, stride=4)
        self.down_act = nn.GELU()
        self.mid_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_resblocks)])
        self.up1 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_act1 = nn.GELU()
        self.up2 = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_act2 = nn.GELU()

    def forward(self, x):
        skip = x
        x = self.down_act(self.down(x))
        x = self.mid_blocks(x)
        x = self.up_act1(self.up1(x))
        x = self.up_act2(self.up2(x))
        return x + skip

# -------------------------------
# Map and Unit Encoders
# -------------------------------

class MapEncoder(nn.Module):
    """
    Modified double-cone map encoder that outputs a single feature vector.
    """
    def __init__(self, in_channels=10, hidden_dim=64, out_dim=128, num_res_pre=2, num_res_mid=5, num_res_post=2):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.initial_act = nn.GELU()
        self.resblocks_pre = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_pre)])
        self.double_cone = DoubleConeBlock(channels=hidden_dim, num_resblocks=num_res_mid)
        self.resblocks_post = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_post)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_act(x)
        x = self.resblocks_pre(x)
        x = self.double_cone(x)
        x = self.resblocks_post(x)
        x = x.mean(dim=[2, 3])  # Global average pooling over H,W
        x = self.head(x)
        return x  # [B, out_dim]

class UnitEncoder(nn.Module):
    """
    Fully connected network for encoding the 9-dimensional unit state.
    """
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Behavior Cloning Actor & Model
# -------------------------------

class BC_Actor(nn.Module):
    """
    The actor network for behavior cloning. It takes a map encoding and a unit state,
    concatenates them, and outputs logits over discrete actions.
    """
    def __init__(self, map_encoding_dim, unit_feature_dim, n_actions):
        super().__init__()
        self.unit_enc = UnitEncoder(in_dim=unit_feature_dim, out_dim=32)
        self.policy_head = nn.Sequential(
            nn.Linear(map_encoding_dim + 32, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, n_actions)  # Logits output (no softmax here)
        )

    def forward(self, map_encoding, unit_input):
        unit_feats = self.unit_enc(unit_input)
        # If map_encoding is [1, D] and we have multiple units, repeat the map encoding.
        if map_encoding.shape[0] == 1 and unit_feats.shape[0] > 1:
            map_encoding = map_encoding.repeat(unit_feats.shape[0], 1)
        combined = torch.cat([map_encoding, unit_feats], dim=1)
        logits = self.policy_head(combined)
        return logits

class BC_Model(nn.Module):
    """
    The full behavior cloning model that fuses the map and unit states
    to predict action logits.
    """
    def __init__(self, map_channels_input, unit_feature_dim, action_dim):
        super().__init__()
        self.map_encoding_dim = 128
        self.mapencoder = MapEncoder(in_channels=map_channels_input, out_dim=self.map_encoding_dim)
        self.actor = BC_Actor(map_encoding_dim=self.map_encoding_dim,
                              unit_feature_dim=unit_feature_dim,
                              n_actions=action_dim)

    def forward(self, unit_states, map_states):
        """
        Args:
            unit_states: Tensor of shape [batch, unit_feature_dim]
            map_states: Tensor of shape [batch, map_channels, H, W]
        Returns:
            logits: Tensor of shape [batch, action_dim]
        """
        map_encoding = self.mapencoder(map_states)
        logits = self.actor(map_encoding, unit_states)
        return logits

class BC_Agent:
    """
    Behavior Cloning Agent wraps the BC_Model and provides helper functions.
    """
    def __init__(self, env_cfg, device=None):
        self.env_cfg = env_cfg
        # The unit state is 9-dimensional (see get_state) and the map has 10 channels.
        self.state_dim = 9  
        # Expert actions are integers in 0 to 5 (6 discrete actions).
        self.action_dim = 6  
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BC_Model(map_channels_input=10, unit_feature_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        
    def predict(self, unit_states, map_state):
        """
        Predicts the actions given the state inputs.
        Args:
            unit_states: list or tensor with shape [batch, 9]
            map_state: tensor with shape [1, map_channels, H, W] (or [batch, map_channels, H, W])
        Returns:
            actions: predicted actions as a NumPy array.
        """
        self.model.eval()
        with torch.no_grad():
            unit_states_t = torch.tensor(unit_states, dtype=torch.float32, device=self.device)
            map_state_t = map_state.to(self.device)
            logits = self.model(unit_states_t, map_state_t)
            actions = torch.argmax(logits, dim=-1)
        return actions.cpu().numpy()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
