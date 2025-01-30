from lux.utils import direction_to
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from map_processing import Playing_Map
class ShipMemory:
    def __init__(self, ship_id, device=torch.device("cpu")):
        self.ship_id = ship_id
        self.device = device
        self.clear_memory()
        self.steps_collected = 0
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        # for convenience if you want
        self.next_value = 0.0

    def get_batch(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'is_terminals': self.is_terminals,
            'values': self.values
        }
class FleetMemory:
    def __init__(self, max_ships, device=torch.device("cpu")):
        self.max_ships = max_ships
        self.ships = [ShipMemory(i,device)for i in range(max_ships)]
        
    def clear_memory(self):
        for ship in self.ships:
            ship.clear_memory()
    def get_trajectories(self):
        trajectories = []
        for ship in self.ships:
            if len(ship.states) > 0:
                trajectory = {
                    'states': torch.stack(ship.states).to(ship.device),
                    'actions': torch.stack(ship.actions).to(ship.device),
                    'logprobs': torch.stack(ship.logprobs).to(ship.device),
                    'rewards': torch.tensor(ship.rewards).to(ship.device),
                    'is_terminals': torch.tensor(ship.is_terminals).to(ship.device),
                    'values': torch.tensor(ship.values).to(ship.device)
                }
                trajectories.append(trajectory)
        return trajectories
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
        return self.fc(x)

class Actor(nn.Module):
    def __init__(self, map_encoding_dim, unit_feature_dim, n_actions):
        super().__init__()
        
        # Encoders
        
        self.unit_enc = UnitEncoder(in_dim=unit_feature_dim, out_dim=64)
        
        # Combine map + unit enc outputs => final policy layer
        self.policy_head = nn.Sequential(
            nn.Linear(map_encoding_dim + 64, 64),  # concat -> 256
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, map_encoding, unit_input):
        """
        map_input:  [batch_size, map_channels, H, W]
        unit_input: [batch_size, unit_feature_dim]
        """
        unit_feats = self.unit_enc(unit_input)# [batch_size, 128]
        map_feats = map_encoding.repeat(unit_feats.shape[0], 1)
            
        combined = torch.cat([map_feats, unit_feats], dim=1)  # [batch_size, 256]
        action_probs=self.policy_head(combined)      # [batch_size, n_actions]
        return  action_probs       # [batch_size, n_actions]
class Critic(nn.Module):
    def __init__(self, map_encoding_dim):
        super().__init__()        
        self.value_head = nn.Sequential(
            nn.Linear(map_encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, map_encoding):    
        value = self.value_head(map_encoding)       # shape [batch_size, 1]
        return value
class ActorCritic(nn.Module):
    def __init__(self,map_channels_input, unit_feature_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.map_encoding_dim = 128
        self.mapencoder=MapEncoder(4,self.map_encoding_dim)
        # Separate actor and critic networks
        self.actor = Actor(map_channels=map_channels_input, unit_feature_dim=unit_feature_dim, n_actions=action_dim)
        self.critic = Critic(self.map_encoding_dim)
    def encode_map(self, map_input):
        self.map_encoding = self.mapencoder(map_input) 
        
    def forward(self, unit_states):
        
        # Extract map and unit features from states
        map_input = self.map_encoding  # Reshape for map channels
        unit_input = unit_states  # Unit features
        
        # Get policy and value
        policy = self.actor(map_input, unit_input)
        value = self.critic(map_input)
        
        return policy, value

    def evaluate(self,map_state, state, action):
        map_encoding = self.mapencoder(map_state) 
        policy= self.actor(map_encoding, state)
        value = self.critic(map_encoding)
        
        # Categorical distribution
        dist = torch.distributions.Categorical(probs=policy)
        
        # Calculate log probabilities
        logprobs = dist.log_prob(action)
        
        # Entropy
        dist_entropy = dist.entropy()
        
        return logprobs, value, dist_entropy
    
    def get_action(self, states):
        policy, value = self.forward(states)
        
        # Categorical distribution 
        dist = torch.distributions.Categorical(probs=policy)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), dist, value

class PPO_Model(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.95,lam=0.95,eps_clip=0.15, lr=0.0001, K_epochs=5, update_timestep=1,device=torch.device("cpu"),entropy_coef=0.001,agent=None):
        super(PPO_Model, self).__init__()
        self.gamma = gamma
        self.lam=lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.update_timestep = update_timestep
        self.step_count=0
        self.policy = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    def compute_gae(self, trajectory):
        rewards = trajectory.rewards
        values = trajectory.values
        dones = trajectory.is_terminals

        # Convert to python list or torch
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        values.append(trajectory.next_value)  # bootstrap

        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            # we treat is_terminals as done
            mask = 1 - float(dones[step])
            delta = rewards[step] + self.gamma * values[step+1] * mask - values[step]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        # optionally normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + torch.FloatTensor(values[:-1])
        return advantages, returns
    
    def update(self, memory):
        advantages, returns = self.compute_gae(memory)
        
        old_unit_states = torch.stack(memory.states).to(self.device).detach()
        old_map_states = torch.stack(memory.map_states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        returns = returns.to(self.device).detach()
        advantages = advantages.to(self.device).detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_unit_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) \
                   + 0.2 * F.mse_loss(state_values.squeeze(-1), returns) \
                   - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.step_count+=1
        if self.step_count>=self.update_timestep:
            self.update_policy()
    def update_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.step_count=0
    def act(self, state, memory):
        """
        If you want to store transitions in memory each time you get an action.
        """
        state_t = torch.FloatTensor(state).to(self.device)
        action, log_prob, dist, value = self.policy_old.get_action(state_t)

        # store
        memory.states.append(state_t)
        memory.actions.append(torch.tensor(action))
        memory.logprobs.append(log_prob)
        memory.values.append(value)

        return action

    
def get_state(unit_pos, nearest_relic_node_position, unit_energy):
    state = [unit_pos[0] / 11.5 - 1, unit_pos[1] / 11.5 - 1,
             nearest_relic_node_position[0] / 11.5 - 1, nearest_relic_node_position[1] / 11.5 - 1,
             unit_energy / 200 - 1]
    return state 




def manhattan_distance(unit_pos,nearest_relic_node_position):
    return abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.state_dim = 5
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.device=torch.device("cpu")
        self.memory=ShipMemory(0,device=self.device)
        #map
        self.play_map=Playing_Map(self.team_id,env_cfg["map_width"],unit_channels=2,map_channels=4,relic_channels=3)
        #model params and model
        self.action_dim = 5
        self.ppo = PPO_Model(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        try:
            self.load_model()
        except:
            pass
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        self.play_map.update_map(obs)
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
                self.relic_node_positions.append(23-observed_relic_node_positions[id])

        ###Try on one unit  
        if len(available_unit_ids) > 0:
            unit_0 = available_unit_ids[0]
            unit_pos = unit_positions[unit_0]
            unit_energy = unit_energys[unit_0]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
            else:
                nearest_relic_node_position = np.array([0,0])
            # get state
            state = get_state(unit_pos, nearest_relic_node_position, unit_energy)
            # get PPO action
            action_idx = self.ppo.act(state, self.memory)  

            actions[unit_0] = [action_idx, 0, 0]


        for unit_id in available_unit_ids[1:]:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                m_d=manhattan_distance(unit_pos,nearest_relic_node_position)
                
                # if close to the relic node we want to hover around it and hope to gain points
                if m_d <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
        if step % 100 == 1 and len(self.memory.states) > 2:
            # you might want to set memory.next_value = <some bootstrap value>
            self.memory.next_value = 0.0
            self.ppo.update(self.memory)
            self.memory.clear_memory()
            

        return actions
    def calculate_rewards_and_dones(self,obs,last_obs,env_cfg,new_points,done):
        
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        new_visible_tiles = np.sum(np.bitwise_xor(obs["sensor_mask"], last_obs["sensor_mask"]))
        visibility_coef=1/((env_cfg["unit_sensor_range"]*2+1)**2)
        visible_tiles=visibility_coef*np.sum(obs["sensor_mask"])/(24*24)
        reward=0
        reward+=visible_tiles
        raw_award=reward
        """if obs['units']['energy'][self.team_id,0]>50:
            reward+=0.05
        else:
            reward-=0.05"""
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)
        return raw_award

        
    def save_model(self):
        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict()
        }, f'modelPPO.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'modelPPO.pth')
            self.ppo.policy.load_state_dict(checkpoint['policy'])
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.agent.player}")
