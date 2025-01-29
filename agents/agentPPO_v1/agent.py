import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import torch.nn.functional as F
from lux.utils import direction_to

class MapEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # flatten to [batch_size, 32*H*W]
        )
        # after flatten, reduce dimension to out_dim
        # (adjust 32*H*W below to match your actual map size, e.g. 32*24*24 if H=W=24)
        self.fc = nn.Sequential(
            nn.Linear(24 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        x shape: [batch_size, in_channels, H, W]
        """
        x = self.conv(x)
        x = self.fc(x)
        return x  # shape: [batch_size, out_dim]
    

class UnitEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        x shape: [batch_size, in_dim]
        """
        return self.fc(x)  # shape: [batch_size, out_dim]
class Actor(nn.Module):
    def __init__(self, map_channels, unit_feature_dim, n_actions):
        super().__init__()
        
        # Encoders
        self.map_enc = MapEncoder(in_channels=map_channels, out_dim=64)
        self.unit_enc = UnitEncoder(in_dim=unit_feature_dim, out_dim=64)
        
        # Combine map + unit enc outputs => final policy layer
        self.policy_head = nn.Sequential(
            nn.Linear(64 + 64, 64),  # concat -> 256
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, map_input, unit_input):
        """
        map_input:  [batch_size, map_channels, H, W]
        unit_input: [batch_size, unit_feature_dim]
        """
        map_feats = self.map_enc(map_input)       # [batch_size, 128]
        unit_feats = self.unit_enc(unit_input)    # [batch_size, 128]
        combined = torch.cat([map_feats, unit_feats], dim=1)  # [batch_size, 256]
        return self.policy_head(combined)         # [batch_size, n_actions]
class Critic(nn.Module):
    def __init__(self, map_channels):
        super().__init__()
        self.map_enc = MapEncoder(in_channels=map_channels, out_dim=64)
        
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, map_input):
        map_feats = self.map_enc(map_input)     
        value = self.value_head(map_feats)       # shape [batch_size, 1]
        return value

class MultiAgentMemoryPPO:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)          # (map_input, unit_input)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    
class Playing_Map():
    def __init__(self,player_id,size,unit_channels=2,map_channels=4,relic_channels=3):
        self.player_id=player_id
        self.size=size
        self.map_channels=map_channels #
        self.unit_channels=unit_channels
        self.relic_channels=relic_channels
        self.channels=map_channels+unit_channels+relic_channels
        self.map_map=torch.zeros((size,size,map_channels))
        self.unit_map=torch.zeros((size,size,unit_channels))
        self.relic_map=torch.zeros((size,size,relic_channels))
    def add_relic(self,pos):
        x,y=pos
        self.relic_map[max(0,x-2):min(23,x+2),max(0,y-2):min(23,y+2),0]=1
        self.relic_map[x,y,2]=1
    def locate_new_reward_source(self,obs,last_obs):
        unit_us=obs["units"]["position"][self.player_id]
        unit_mask_us=obs["units_mask"][self.player_id]
        valid_positions_us = unit_us[unit_mask_us].T
        expected_reward=sum(self.relic_map[valid_positions_us[0],valid_positions_us[1],0]>=2)
        actual_reward=obs["team_points"][self.player_id]-last_obs["team_points"][self.player_id]
        self.relic_map[valid_positions_us[0],valid_positions_us[1],0]>0 +=0.5*(actual_reward-expected_reward)
        
    def update_map(self,obs):
        #Visibility, right now
        visibility=torch.from_numpy(obs["sensor_mask"])
        #Update map
        rows, cols = torch.where(visibility)
        self.map_map[rows, cols,1:4] = torch.nn.functional.one_hot(torch.from_numpy(obs['map_features']['tile_type'])[visibility].long(), num_classes=3).float()
        self.map_map[:,:,0]-visibility.int()
        #Update units
        unit_us=obs["units"]["position"][self.player_id]
        unit_mask_us=obs["units_mask"][self.player_id]
        valid_positions_us = unit_us[unit_mask_us].T
        values_us = torch.ones(valid_positions_us.shape[1], dtype=torch.float32)
        self.unit_map[:,:,0]=0
        self.unit_map[:,:,1]/=2
        self.unit_map[:,:,0].index_put_((torch.tensor(valid_positions_us[0], dtype=torch.long), torch.tensor(valid_positions_us[1], dtype=torch.long)),torch.tensor(values_us, dtype=torch.float32),accumulate=True)

        unit_mask_them=obs["units_mask"][1-self.player_id]
        unit_them=obs["units"]["position"][1-self.player_id]
        valid_positions_them = unit_them[unit_mask_them].T
        values_them = torch.ones(valid_positions_them.shape[1], dtype=torch.float32)
        self.unit_map[:,:,1].index_put_((torch.tensor(valid_positions_them[0], dtype=torch.long), torch.tensor(valid_positions_them[1], dtype=torch.long)),torch.tensor(values_them, dtype=torch.float32),accumulate=True)
        #Update relics
        relics=obs["relic_nodes"]
        relics_mask=obs["relic_nodes_mask"]
        relics_pos = relics[relics_mask].T
        self.relic_map[relics_pos[0],relics_pos[1],0]=1


    def map_stack(self):
        map_stack = torch.cat(
                                [
                                self.map_map,    # shape = [H, W, map_channels]
                                self.unit_map,   # shape = [H, W, unit_channels]
                                self.relic_map,  # shape = [H, W, relic_channels]
                                ], 
                                dim=-1  # last dimension, so new shape = (H, W, total_channels)
                            )
        map_stack = map_stack.permute(2, 0, 1)
        return map_stack
        


class Agent:
    def __init__(self, player: str, env_cfg, training=True) -> None:
        self.env_cfg=env_cfg
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.training = training
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.max_units=self.env_cfg["max_units"]
        # Initialize map
        self.map=Playing_Map(self.team_id,24)
        # Net parameters
        self.unit_feature_dim = 4  # unit_pos(2) + closest_relic(2) + unit_energy(1)+ step(1)+unit_id(1)
        self.action_size = 6 # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.map_channels=self.map.channels
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate_actor = 0.0006
        self.learning_rate_critic =  0.0005
        self.random_location=np.array([[4, 8], [8, 0], [6, 16], [4, 6], [20, 3], [23, 1], [19, 7], [6, 8],[3, 7], [19, 15], [23, 13], [0, 13], [5, 5], [10, 16], [20, 15], [7, 23]])
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(self.map_channels, self.unit_feature_dim,n_actions=6)
        self.critic = Critic(self.map_channels)
        self.memory = MultiAgentMemoryPPO(16)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        

        



        if not training:
            self.load_model()
            self.epsilon = 0.0
        elif os.path.exists(f'modelA2C_{self.player}.pth'):
            self.load_model()
    
    def _state_representation(self, unit_pos, unit_energy, unit_id):

        state = np.concatenate([
            unit_pos-12,
            [unit_energy/400],
            [unit_id/16]   # Make sure this is in a list or array
        ])
        return torch.FloatTensor(state).to(self.device)
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # Gather basic observations
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])

        # Update map
        self.map.update_map(obs)

        # Prepare arrays for storing final actions & log_probs
        log_probs = [torch.zeros([], dtype=torch.float32) for _ in range(self.max_units)]
        actions = np.zeros((self.max_units, 3), dtype=int)

        # Identify which units are alive
        available_units = np.where(unit_mask)[0]
        num_units = len(available_units)

        # Build a batch of states for all alive units
        # shape => [num_units, self.unit_feature_dim]
        #Can be more efficient(Parallel)
        unit_states = []
        for unit_id in available_units:
            unit_state = self._state_representation(
                unit_pos=unit_positions[unit_id],
                unit_energy=unit_energys[unit_id],
                unit_id=unit_id
            )
            unit_states.append(unit_state)
        #Dont Understand
        if num_units == 0:
            # No units alive => no actions
            # Use the critic anyway (usually returns shape [1, 1])
            # Just return zeros or skip. For simplicity, let's skip:
            map_data = self.map.map_stack().unsqueeze(0).to(self.device)
            values = self.critic(map_data)
            return actions, log_probs, values

        # Stack all states into [num_units, unit_feature_dim]
        unit_states = torch.stack(unit_states, dim=0).to(self.device)

        # Prepare map_data as [1, channels, H, W]
        map_data_single = self.map.map_stack().unsqueeze(0).to(self.device)
        # Repeat it for each unit => [num_units, channels, H, W]
        map_data_batch = map_data_single.repeat(num_units, 1, 1, 1)

        # 1) Forward pass for all units in one go
        # logits => [num_units, n_actions]
        logits = self.actor(map_data_batch, unit_states)

        # 2) Sample an action for each unit
        dist = torch.distributions.Categorical(logits)
        actions_tensor = dist.sample()  # shape => [num_units]
        log_probs_tensor = dist.log_prob(actions_tensor)  # [num_units]

        # 3) Convert to NumPy, then assign each unit's action & log_prob
        actions_array = actions_tensor.cpu().numpy()
        for i, unit_id in enumerate(available_units):
            a = int(actions_array[i])
            log_probs[unit_id] = log_probs_tensor[i]
            
            # Convert the discrete action to (action_type, x, y)
            if a == 5:
                # Sap action: pick first valid enemy
                opp_positions = obs["units"]["position"][self.opp_team_id]
                opp_mask = obs["units_mask"][self.opp_team_id]
                valid_targets = [
                    pos for opp_id, pos in enumerate(opp_positions)
                    if opp_mask[opp_id] and pos[0] != -1
                ]
                if valid_targets:
                    target_pos = valid_targets[0]
                    actions[unit_id] = [5, target_pos[0], target_pos[1]]
                else:
                    actions[unit_id] = [0, 0, 0]  # Stay if no valid targets
            else:
                # 0 = center, 1=up, 2=right, 3=down, 4=left
                actions[unit_id] = [a, 0, 0]

        # 4) Single forward pass through Critic for global value
        # map_data for critic is typically just shape => [1, channels, H, W]
        values = self.critic(map_data_single)  # shape => [1, 1]

        return actions, log_probs, values

def train_on_episode_memory(self):
    """
    Perform a single A2C update after one rollout (episode).
    """
    # Convert stored data into tensors
    values = torch.stack(self.memory.values)  # shape: [T, 1]
    T = values.size(0)

    # For advantage computation, we need the "next value" at each step
    # We'll do this in reverse, as usual in A2C.
    # We'll keep separate returns for each agent if rewards are separate.
    # But the critic is centralized => it produces a single state-value for the global state.
    # We'll compute G_t (the return) for each agent individually.
    # Then advantage = G_t - V(s_t).
    returns_per_agent = [np.zeros((T, ), dtype=np.float32) for _ in range(self.memory.n_agents)]

    next_value = 0.0  # because we assume episode is done, so V(s_{T+1})=0
    for i_agent in range(self.memory.n_agents):
        running_return = 0.0
        for t in reversed(range(T)):
            if self.memory.dones[t]:
                next_value = 0.0
                running_return = 0.0
            # G_t = r_t + gamma * V(s_{t+1}) (for advantage),
            # but we accumulate in running_return
            running_return = self.memory.rewards[i_agent][t] + self.gamma * running_return
            returns_per_agent[i_agent][t] = running_return

    # Convert to Torch
    all_advantages = []
    for i_agent in range(self.memory.n_agents):
        # shape [T]
        returns = torch.from_numpy(returns_per_agent[i_agent])
        # shape [T,1]
        advantages = returns.unsqueeze(-1) - values
        all_advantages.append(advantages)

    # Critic loss is computed from the perspective of the global state,
    # but each agent's return is its own. Often in cooperative settings,
    # you might use the sum or average of rewards. For demonstration,
    # we do a simple sum of MSE across agents.
    # shape: [T,1]
    stacked_advantages = torch.stack(all_advantages, dim=0)  # [n_agents, T, 1]
    critic_loss = (stacked_advantages ** 2).mean()

    self.critic_optim.zero_grad()
    critic_loss.backward()
    self.critic_optim.step()
    actor_loss = 0
    for i_agent in range(self.memory.n_agents):
        log_probs_i = torch.stack(self.memory.log_probs[i_agent])  # shape [T]
        advantages_i = all_advantages[i_agent].detach()       # shape [T,1]
        # We need them to be consistent shape for multiplication
        # log_probs_i -> [T,1], broadcast with advantages_i
        log_probs_i = log_probs_i.unsqueeze(-1)
        actor_loss += (- log_probs_i * advantages_i).mean()

    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()

    return actor_loss.item(), critic_loss.item()

def save_model(self):
    torch.save({
        'actor': self.actor.state_dict(),
        'critic': self.critic.state_dict(),
        'critic_optim': self.critic_optim.state_dict(),
        'actor_optim': self.actor_optim.state_dict()
    }, f'modelA2C_{self.player}.pth')

def load_model(self):
    try:
        checkpoint = torch.load(f'modelA2C_{self.player}.pth')
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
    except FileNotFoundError:
        raise FileNotFoundError(f"No trained model found for {self.player}")