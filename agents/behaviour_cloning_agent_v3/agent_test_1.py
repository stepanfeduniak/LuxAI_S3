import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import numpy as np
from map_processing import Playing_Map
from lux.utils import direction_to
import matplotlib.pyplot as plt
import random
# -------------------------------
# Basic building blocks for the U-Net
# -------------------------------

class ConvBlock(nn.Module):
    """
    A simple convolutional block: Conv2d (kernel=3, padding=1, optional stride), BatchNorm2d, and GELU.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -------------------------------
# U-Net Style Behavior Cloning Model
# -------------------------------

class UNetBC_Model(nn.Module):
    """
    A U-Net–style architecture for behavior cloning.
    
    The map (with shape [1, map_channels, H, W]) is first encoded.
    At the bottleneck (lowest spatial resolution) the per–unit state (a vector)
    is injected via a learned linear projection (and broadcast over the spatial dimensions).
    Then a symmetric decoder upsamples (using skip–connections) to yield a spatial feature map.
    Finally, a 1x1 convolution and global spatial average pooling produces a [B, n_actions] logits output.
    
    During inference the map is shared across units and (if necessary) repeated to match the number of units.
    """
    def __init__(self, map_channels_input, unit_feature_dim, n_actions, base_channels=64, dropout_p=0.1):
        super().__init__()
        # --- Encoder ---
        # Level 1 (no downsampling)
        self.enc1 = ConvBlock(map_channels_input, base_channels, stride=1)
        # Level 2 (downsample by 2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        # Level 3 (downsample by 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        # Bottleneck (further downsample by 2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, stride=2)
        
        # --- Conditioning at the bottleneck ---
        # Project the per–unit state (a 1D vector) into a feature map that can be concatenated.
        self.unit_fc = nn.Linear(unit_feature_dim, base_channels * 8)
        # After concatenation (bottleneck + tiled unit state) the channels become base_channels*8 * 2.
        # Fuse them back to base_channels*8 channels.
        self.bottleneck_fuse = ConvBlock(base_channels * 8 * 2, base_channels * 8, stride=1)
        
        # --- Decoder ---
        # Decoder stage 1: upsample to match enc3 resolution.
        self.dec3_up = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                          kernel_size=2, stride=2)
        # After upsampling, we concatenate the skip from enc3 (which has base_channels*4 channels)
        self.dec3_conv = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4, stride=1)
        
        # Decoder stage 2: upsample to match enc2 resolution.
        self.dec2_up = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                          kernel_size=2, stride=2)
        self.dec2_conv = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2, stride=1)
        
        # Decoder stage 3: upsample to match enc1 resolution.
        self.dec1_up = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                          kernel_size=2, stride=2)
        self.dec1_conv = ConvBlock(base_channels + base_channels, base_channels, stride=1)
        
        # Final head: produce n_actions feature maps and then use global average pooling.
        self.final_conv = nn.Conv2d(base_channels, 32, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1=nn.Linear(32,n_actions)
        
    def forward(self, map_input, unit_state):
        """
        Args:
            map_input: Tensor of shape [B, map_channels, H, W]. (If B==1, the same global map can be repeated.)
            unit_state: Tensor of shape [B, unit_feature_dim] (one per unit).
        Returns:
            logits: Tensor of shape [B, n_actions]
        """
        # If the map_input comes with batch size 1 (i.e. one global map) but we have multiple unit states,
        # then repeat the map for each unit.
        B_units = unit_state.shape[0]
        if map_input.shape[0] == 1 and B_units > 1:
            map_input = map_input.repeat(B_units, 1, 1, 1)
            
        # --- Encoder (store skip connections) ---
        e1 = self.enc1(map_input)      # shape: [B, base_channels, H, W]
        e2 = self.enc2(e1)             # shape: [B, base_channels*2, H/2, W/2]
        e3 = self.enc3(e2)             # shape: [B, base_channels*4, H/4, W/4]
        bn = self.bottleneck(e3)       # shape: [B, base_channels*8, H/8, W/8]
        
        # --- Inject unit state at the bottleneck ---
        # Project unit_state to shape [B, base_channels*8]
        unit_proj = self.unit_fc(unit_state)  # [B, base_channels*8]
        # Reshape to [B, base_channels*8, 1, 1] then expand to match bn's spatial size.
        unit_proj = unit_proj.unsqueeze(-1).unsqueeze(-1)  # [B, base_channels*8, 1, 1]
        unit_proj = unit_proj.expand_as(bn)                # [B, base_channels*8, H/8, W/8]
        # Concatenate along the channel dimension.
        bn_cat = torch.cat([bn, unit_proj], dim=1)         # [B, base_channels*8*2, H/8, W/8]
        bn_fused = self.bottleneck_fuse(bn_cat)            # [B, base_channels*8, H/8, W/8]
        
        # --- Decoder ---
        # Decoder stage 1: upsample bn_fused to match e3.
        d3 = self.dec3_up(bn_fused)                        # [B, base_channels*4, H/4, W/4]
        # Concatenate with the corresponding encoder skip connection.
        d3 = torch.cat([d3, e3], dim=1)                    # [B, base_channels*4*2, H/4, W/4]
        d3 = self.dec3_conv(d3)                            # [B, base_channels*4, H/4, W/4]
        
        # Decoder stage 2: upsample to match e2.
        d2 = self.dec2_up(d3)                              # [B, base_channels*2, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)                    # [B, base_channels*2*2, H/2, W/2]
        d2 = self.dec2_conv(d2)                            # [B, base_channels*2, H/2, W/2]
        
        # Decoder stage 3: upsample to match e1.
        d1 = self.dec1_up(d2)                              # [B, base_channels, H, W]
        d1 = torch.cat([d1, e1], dim=1)                    # [B, base_channels*2, H, W]
        d1 = self.dec1_conv(d1)                            # [B, base_channels, H, W]
        
        # --- Final head ---
        out = self.final_conv(d1)# [B, n_actions, H, W]
        out = self.dropout(out)
        # Global average pooling over the spatial dimensions to get [B, 32]
        out = out.mean(dim=[2, 3])
        logits=self.fc1(out)
        return logits
    
    def act(self, unit_states, map_stack):
        """
        Args:
            unit_states: list or array-like with shape [num_units, unit_feature_dim]
            map_stack:   Tensor with shape [1, map_channels, H, W]
        Returns:
            logits: Tensor of shape [num_units, n_actions]
        """
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
# Agent Class using the New Model
# -------------------------------

class Agent_test_1:
    """
    Behavior Cloning Agent using the new U-Net style network.
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
        self.opp_team_id = 1 - self.team_id
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
        
        # Use the new U-Net style model.
        # (Note: the map_channels_input here is 10 since you had 10 channels in your map_stack.)
        self.model = UNetBC_Model(map_channels_input=10,
                                  unit_feature_dim=self.state_dim,
                                  n_actions=self.action_dim).to(self.device)
        path = "bc_model_v2.pth"
        try:
            self.load_model(path=path)
        except FileNotFoundError:
            pass
        self.last_obs = None
        
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
        self.play_map.update_map(obs, self.last_obs)
        self.last_obs = obs
        # which units can act?
        available_unit_ids = np.where(unit_mask)[0]

        # record newly discovered relics
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            if rid not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(rid)
                self.relic_node_positions.append(observed_relic_node_positions[rid])
                self.play_map.add_relic(observed_relic_node_positions[rid])

        # Build the single map stack [1, 10, H, W]
        map_data_single = self.play_map.map_stack().unsqueeze(0).to(self.device)

        # build unit states
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
        
        logits = self.model.act(
            unit_states=states_list,
            map_stack=map_data_single,
        )
        # Scale logits if needed (the original code scales the first logit by 0.1)
        #logits = logits * torch.tensor([0.4, 0.8, 1, 1, 0.8, 0.5], device=self.device)
        # Convert logits to probabilities with softmax.
        probs = F.softmax(logits, dim=-1)
        # Create a categorical distribution and sample an action.
        dist = torch.distributions.Categorical(probs)
        sampled_action_np = dist.sample().cpu().numpy()
        
        # build final action array
        # direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
        reverse_action_map={0:0,1:3,2:4,3:1,4:2,5:5}
        actions_array = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(available_unit_ids):
            chosen_action=sampled_action_np[i]
            if self.team_id==1:
                actions_array[unit_id, 0] = reverse_action_map[chosen_action]
                actions_array[unit_id, 1] = 0
                actions_array[unit_id, 2] = 0
                
            else:
                actions_array[unit_id, 0] = chosen_action
                actions_array[unit_id, 1] = 0
                actions_array[unit_id, 2] = 0
            if sampled_action_np[i] == 5:
                # Sap action: pick first valid enemy
                opp_positions = obs["units"]["position"][self.opp_team_id]
                opp_mask = obs["units_mask"][self.opp_team_id]
                valid_targets = [
                    pos for opp_id, pos in enumerate(opp_positions)
                    if opp_mask[opp_id] and pos[0] != -1
                ]

                if valid_targets:
                    unit_pos = unit_positions[unit_id]
                    target_pos = min(valid_targets, key=lambda pos: manhattan_distance(unit_pos, pos))
                    if manhattan_distance(target_pos, unit_pos) <= self.unit_sap_range and unit_energys[unit_id] >= self.env_cfg['unit_sap_cost']:
                        #print(f"Successful attack at {target_pos}")
                        #self.show_map(step)
                        actions_array[unit_id] = [5, target_pos[0]-unit_pos[0], target_pos[1]-unit_pos[1]]
                    else:
                        actions_array[unit_id] = [0, 0, 0]
                else:
                    actions_array[unit_id] = [0, 0, 0]  # Stay if no valid targets

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
    def show_map(self,step,showmap=False):
        print(step)
        if showmap:
            play_map=self.play_map.map_stack()
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            axes = axes.flatten()
            for i in range(10):
                # Convert the i-th map (which is a torch tensor) to a NumPy array.
                heatmap = play_map[i, :, :].numpy()
                # Display the heatmap. You can change the colormap ('cmap') as desired.
                im = axes[i].imshow(heatmap, cmap='viridis')
                axes[i].set_title(f"Row {i}")
                axes[i].axis('off')  # Optional: Hide axis ticks
                
                # Optionally add a colorbar to each subplot
                fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
