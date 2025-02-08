# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from map_processing import Playing_Map
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions
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
    def act(self, unit_states, map_stack):
        """
        unit_states: shape [num_units, unit_feature_dim]
        map_stack:   shape [1, map_channels, H, W], 
                     or [num_units, map_channels, H, W]
        """
        unit_states_t = torch.as_tensor(unit_states, dtype=torch.float32)

        with torch.no_grad():
            # sample action using old policy
            map_encoding = self.mapencoder(map_stack)
            logits = self.actor(map_encoding, unit_states_t)
        return logits
    
#Acting
def get_state(unit_pos, nearest_relic_node_position, unit_energy,env_cfg):
    return [
        unit_pos[0] / 11.5 - 1,
        unit_pos[1] / 11.5 - 1,
        nearest_relic_node_position[0] / 11.5 - 1,
        nearest_relic_node_position[1] / 11.5 - 1,
        unit_energy / 200 - 1,
        env_cfg['unit_sensor_range']/5,
        env_cfg['unit_move_cost']/10,
        env_cfg['unit_sap_cost']/50,
        env_cfg['unit_sap_range']/5,
        
    ]
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
class Agent:
    """
    Behavior Cloning Agent wraps the BC_Model and provides helper functions.
    """
    def __init__(self, player: str, env_cfg):
        device=None
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.play_map = Playing_Map(
            player_id=self.team_id, 
            map_size=env_cfg["map_width"],
            unit_channels=2,
            map_channels=5,
            relic_channels=3
        )
        self.env_cfg = env_cfg
        # The unit state is 9-dimensional (see get_state) and the map has 10 channels.
        self.state_dim = 9  
        # Expert actions are integers in 0 to 5 (6 discrete actions).
        self.action_dim = 6  
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BC_Model(map_channels_input=10, unit_feature_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        path="bc_model.pth"
        try:
            self.load_model(path=path)
        except FileNotFoundError:
            pass
        self.last_obs=None
        
    def act(self, step, obs, remainingOverageTime=60):
        """
        Predicts the actions given the state inputs.
        Args:
            unit_states: list or tensor with shape [batch, 9]
            map_state: tensor with shape [1, map_channels, H, W] (or [batch, map_channels, H, W])
        Returns:
            actions: predicted actions as a NumPy array.
        """
        self.model.eval()
        # extract data from obs
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"])
        
        # map update
        self.play_map.update_map(obs,self.last_obs)
        self.last_obs=obs
        # which units can act?
        available_unit_ids = np.where(unit_mask)[0]

        # record newly discovered relics
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                self.relic_node_positions.append(observed_relic_node_positions[rid])
                self.play_map.add_relic(observed_relic_node_positions[rid])

        # Build the single map stack [1, 4, H, W]
        
        map_data_single = self.play_map.map_stack().unsqueeze(0).to(self.device)

        # build unit states
        states_list = []
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nrp = min(self.relic_node_positions, key=lambda pos: manhattan_distance(unit_pos, pos))
            else:
                nrp = np.array([13,13])
            st = get_state(unit_pos, nrp, unit_energy,self.env_cfg)
            states_list.append(st)
        if len(available_unit_ids) == 0:
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        logits = self.model.act(
            unit_states=states_list,
            map_stack=map_data_single,
        )

        # Convert logits to probabilities with softmax.
        probs = F.softmax(logits, dim=-1)

        # Create a categorical distribution and sample an action.
        dist = torch.distributions.Categorical(probs)
        sampled_action_np = dist.sample().cpu().numpy()
        
        # build final action array
        actions_array = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(available_unit_ids):
            actions_array[unit_id, 0] = sampled_action_np[i]
            actions_array[unit_id, 1] = 0
            actions_array[unit_id, 2] = 0
        
        return actions_array
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
