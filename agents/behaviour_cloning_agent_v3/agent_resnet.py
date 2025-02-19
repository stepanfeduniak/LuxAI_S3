
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import numpy as np
from map_processing import Playing_Map

# -------------------------------
# Squeeze-and-Excitation Block
# -------------------------------
class SqueezeAndExcitation(nn.Module):
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
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# -------------------------------
# Residual Block with SE
# -------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.se = SqueezeAndExcitation(channels, reduction=16)
    def forward(self, x):
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return out + residual

# -------------------------------
# Double Cone Block
# -------------------------------
class DoubleConeBlock(nn.Module):
    """
    Downsample via a conv (kernel=4, stride=4), process with several ResBlocks,
    then upsample twice (overall upsampling factor=4), and add a skip connection.
    """
    def __init__(self, channels, num_resblocks=4):
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
# Efficient Map Encoder
# -------------------------------
class MapEncoder(nn.Module):
    """
    Encodes the global map into a fixed-length vector.
    Uses:
      1. An initial conv to get to hidden_dim channels.
      2. A few ResBlocks (pre-cone).
      3. A DoubleConeBlock.
      4. A few ResBlocks (post-cone).
      5. Global average pooling followed by an MLP head that outputs a 128-dim vector.
    """
    def __init__(self, in_channels=10, hidden_dim=64, out_dim=128, 
                 num_res_pre=2, num_res_mid=4, num_res_post=2):
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
        # x: [B, in_channels, H, W]
        x = self.initial_act(self.initial_conv(x))
        x = self.resblocks_pre(x)
        x = self.double_cone(x)
        x = self.resblocks_post(x)
        x = x.mean(dim=[2, 3])  # Global average pooling => [B, hidden_dim]
        x = self.head(x)        # => [B, out_dim]
        return x

# -------------------------------
# Unit Encoder
# -------------------------------
class UnitEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
            nn.GELU()
        )
    def forward(self, x):
        # x: [B, in_dim]
        return self.fc(x)

# -------------------------------
# Actor with FiLM Fusion (Output remains same as original)
# -------------------------------
class Actor(nn.Module):
    def __init__(self, map_encoding_dim, unit_feature_dim, n_actions):
        super().__init__()
        # Process raw unit features to a higher-dimensional embedding.
        self.unit_enc = UnitEncoder(in_dim=unit_feature_dim, out_dim=64)
        # FiLM: use map encoding to compute scale (gamma) and shift (beta) for the unit encoding.
        self.film_gamma = nn.Linear(map_encoding_dim, 64)
        self.film_beta = nn.Linear(map_encoding_dim, 64)
        # Final policy head: we concatenate the original map encoding (of size map_encoding_dim)
        # with the modulated unit encoding (64 dims) to produce the final logits.
        fc_input_dim = map_encoding_dim + 64
        self.policy_head = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_actions)
        )
    def forward(self, map_encoding, unit_input):
        """
        map_encoding: [B, map_encoding_dim] (or [1, map_encoding_dim] if shared)
        unit_input:   [B, unit_feature_dim]
        """
        unit_feats = self.unit_enc(unit_input)         # [B, 64]
        gamma = self.film_gamma(map_encoding)            # [B, 64]
        beta = self.film_beta(map_encoding)              # [B, 64]
        modulated_unit = gamma * unit_feats + beta       # FiLM conditioning
        combined = torch.cat([map_encoding, modulated_unit], dim=1)  # [B, map_encoding_dim + 64]
        logits = self.policy_head(combined)
        return logits

# -------------------------------
# Complete Efficient Agent Model (Output as Original)
# -------------------------------
class ResNetBC_Model(nn.Module):
    def __init__(self, map_channels_input=10, unit_feature_dim=10, n_actions=6, map_out_dim=128):
        super().__init__()
        self.map_enc = MapEncoder(in_channels=map_channels_input,
                                  hidden_dim=128,
                                  out_dim=map_out_dim,
                                  num_res_pre=4,
                                  num_res_mid=6,
                                  num_res_post=4)
        self.actor = Actor(map_encoding_dim=map_out_dim,
                           unit_feature_dim=unit_feature_dim,
                           n_actions=n_actions)
    def forward(self, map_input, unit_input):
        """
        map_input: [B, in_map_channels, H, W]
        unit_input: [B, unit_feature_dim]
        Returns:
            logits: [B, n_actions]
        """
        map_encoding = self.map_enc(map_input)   # [B, map_out_dim]
        if map_encoding.size(0) == 1 and unit_input.size(0) > 1:
            map_encoding = map_encoding.expand(unit_input.size(0), -1)
        logits = self.actor(map_encoding, unit_input)
        return logits
    def act(self, unit_states, map_stack):
        device = map_stack.device
        unit_states_t = torch.as_tensor(unit_states, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(map_stack, unit_states_t)
        return logits

# -------------------------------
# Helper Functions (unchanged from your original code)
# -------------------------------

def get_state(unit_pos, nearest_relic_node_position, unit_energy, env_cfg, team):
    if team == 0:
        return [
            unit_pos[0] / 11.5 - 1,
            unit_pos[1] / 11.5 - 1,
            nearest_relic_node_position[0] / 11.5 - 1,
            nearest_relic_node_position[1] / 11.5 - 1,
            unit_energy / 200 - 1,
            env_cfg['unit_sensor_range'] / 5,
            env_cfg['unit_move_cost'] / 10,
            env_cfg['unit_sap_cost'] / 50,
            env_cfg['unit_sap_range'] / 5,
            team
        ]
    else:
        return [
            (23 - unit_pos[1]) / 11.5 - 1,
            (23 - unit_pos[0]) / 11.5 - 1,
            (23 - nearest_relic_node_position[1]) / 11.5 - 1,
            (23 - nearest_relic_node_position[0]) / 11.5 - 1,
            unit_energy / 200 - 1,
            env_cfg['unit_sensor_range'] / 5,
            env_cfg['unit_move_cost'] / 10,
            env_cfg['unit_sap_cost'] / 50,
            env_cfg['unit_sap_range'] / 5,
            team
        ]

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -------------------------------
# Agent Class using the New ResNet Model
# -------------------------------

class Agent_resnet:
    """
    Behavior Cloning Agent using the new ResNet–style network.
    """
    def __init__(self, player: str, env_cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
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
        
        self.unit_sap_range = env_cfg["unit_sap_range"]
        # The unit state is 10-dimensional (see get_state) and the map has 10 channels.
        self.state_dim = 10  
        # Expert actions are integers in 0 to 5 (6 discrete actions).
        self.action_dim = 6  
        self.device = device
        
        # Use the new ResNet–style model.
        # (Note: the map_channels_input here is 10 since you had 10 channels in your map_stack.)
        self.model = ResNetBC_Model(map_channels_input=10,
                                    unit_feature_dim=self.state_dim,
                                    n_actions=self.action_dim).to(self.device)
        path = "bc_model_resnet.pth"
        try:
            self.load_model(path=path)
        except FileNotFoundError:
            pass
        self.last_obs = None
        
    def act(self, step, obs, remainingOverageTime=60):
        """
        Predicts the actions given the state inputs.
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
        
        # Update the playing map.
        self.play_map.update_map(obs, self.last_obs)
        self.last_obs = obs
        
        # Determine which units can act.
        available_unit_ids = np.where(unit_mask)[0]

        # Record newly discovered relic nodes.
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                self.relic_node_positions.append(observed_relic_node_positions[rid])
                self.play_map.add_relic(observed_relic_node_positions[rid])
        
        # Build the single map stack [1, 10, H, W] and move to device.
        map_data_single = self.play_map.map_stack().unsqueeze(0).to(self.device)
        
        # Build unit states.
        states_list = []
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nrp = min(self.relic_node_positions, key=lambda pos: manhattan_distance(unit_pos, pos))
            else:
                nrp = np.array([-1, -1])
            st = get_state(unit_pos, nrp, unit_energy, self.env_cfg, self.team_id)
            states_list.append(st)
        if len(available_unit_ids) == 0:
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        # Get logits from the ResNet model.
        logits = self.model.act(unit_states=states_list, map_stack=map_data_single)
        # Optionally, scale logits if needed.
        logits = logits * torch.tensor([1, 1, 1, 1, 1, 1], device=self.device)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled_action_np = dist.sample().cpu().numpy()
        
        # Build final action array.
        # direction mapping (0 = center, 1 = up, 2 = right, 3 = down, 4 = left, 5 = sap)
        reverse_action_map = {0:0, 1:3, 2:4, 3:1, 4:2, 5:5}
        actions_array = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(available_unit_ids):
            if self.team_id == 1:
                actions_array[unit_id, 0] = reverse_action_map[sampled_action_np[i]]
            else:
                actions_array[unit_id, 0] = sampled_action_np[i]
            actions_array[unit_id, 1] = 0
            actions_array[unit_id, 2] = 0
            
            if sampled_action_np[i] == 5:
                # Sap action: pick first valid enemy.
                opp_positions = obs["units"]["position"][1-self.team_id]
                opp_mask = obs["units_mask"][1-self.team_id]
                valid_targets = [
                    pos for opp_id, pos in enumerate(opp_positions)
                    if opp_mask[opp_id] and pos[0] != -1
                ]
                if valid_targets:
                    unit_pos = unit_positions[unit_id]
                    target_pos = min(valid_targets, key=lambda pos: manhattan_distance(unit_pos, pos))
                    if (manhattan_distance(target_pos, unit_pos) <= self.unit_sap_range and 
                        unit_energys[unit_id] >= self.env_cfg['unit_sap_cost']):
                        actions_array[unit_id] = [5, target_pos[0], target_pos[1]]
                    else:
                        actions_array[unit_id] = [0, 0, 0]
                else:
                    actions_array[unit_id] = [0, 0, 0]  # Stay if no valid targets.
        return actions_array

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        new_state_dict = {}
        prefix = "_orig_mod."
        for key, value in state_dict.items():
            new_key = key[len(prefix):] if key.startswith(prefix) else key
            new_state_dict[new_key] = value
        self.model.load_state_dict(new_state_dict)

