import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import torch.nn.functional as F
from lux.utils import direction_to
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        # Common feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)  # outputs a single scalar V(s)
        )
        
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size)  # outputs advantage for each action
        )

    def forward(self, x):
        # Common features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        values = self.value_stream(features)           # shape: [batch_size, 1]
        advantages = self.adv_stream(features)         # shape: [batch_size, output_size]
        
        # Combine them into Q-values
        # Q(s,a) = V(s) + ( A(s,a) - mean(A(s,a) over a) )
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        q_values = values + (advantages - advantages_mean)
        
        return q_values
# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
"""def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, player: str, env_cfg, training=True) -> None:
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
        # DQN parameters
        self.state_size = 6  # unit_pos(2) + closest_relic(2) + unit_energy(1)+ step(1)+unit_id(1)
        self.action_size = 6 # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.96
        self.epsilon = 1
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.random_location=np.array([[4, 8], [8, 0], [6, 16], [4, 6], [20, 3], [23, 1], [19, 7], [6, 8],[3, 7], [19, 15], [23, 13], [0, 13], [5, 5], [10, 16], [20, 15], [7, 23]])
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(5000)



        if not training:
            self.load_model()
            self.epsilon = 0.0
        elif os.path.exists(f'dqn_model_{self.player}.pth'):
            self.load_model()
    
    def _state_representation(self, unit_pos, unit_energy, relic_node_positions, unit_id):
        # relic_node_positions is currently a list in your code.
        # Let's convert it to NumPy right here:
        relic_array = np.array(relic_node_positions)

        if len(relic_array) == 0:
            closest_relic = self.random_location[unit_id]
        else:
            distances = np.linalg.norm(relic_array - unit_pos, axis=1)
            closest_relic = relic_array[np.argmin(distances)]
        state = np.concatenate([
            unit_pos-12,
            closest_relic-12,
            [unit_energy/100],
            [unit_id/16]   # Make sure this is in a list or array
        ])
        return torch.FloatTensor(state).to(self.device)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        relic_nodes = np.array(obs["relic_nodes"])
        relic_mask = np.array(obs["relic_nodes_mask"])
        self.score = np.array(obs["team_points"][self.team_id])
        observed_relic_node_positions = np.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
            # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
        #if step % 500 == 0:
         #   print(f"memory:  {len(self.memory)}")
        #if step%100==99:
        #    print(f"Relics found by {self.player} :{self.relic_node_positions}")
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        available_units = np.where(unit_mask)[0]
        for unit_id in available_units:
            state = self._state_representation(
                unit_positions[unit_id],
                unit_energys[unit_id],
                self.relic_node_positions,
                unit_id
            )
            state = state.unsqueeze(0)
            if random.random() < self.epsilon and self.training:
                unit_pos = unit_positions[unit_id]
                unit_energy = unit_energys[unit_id]
                if len(self.relic_node_positions) > 0:
                    nearest_relic_node_position = (state[0][2]+12,state[0][3]+12)
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                    
                    # if close to the relic node we want to hover around it and hope to gain points
                    if manhattan_distance <= 3:
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



            else:
                with torch.no_grad():
                    q_values = self.policy_net(state)
                    action_type = q_values.argmax().item()
                    #print(f"Q-values: {q_values}")
                if action_type == 5:  # Sap action
                    # Find closest enemy unit
                    opp_positions = obs["units"]["position"][self.opp_team_id]
                    opp_mask = obs["units_mask"][self.opp_team_id]
                    valid_targets = []

                    for opp_id, pos in enumerate(opp_positions):
                        if opp_mask[opp_id] and pos[0] != -1:
                            valid_targets.append(pos)

                    if valid_targets:
                        target_pos = valid_targets[0]  # Choose first valid target
                        actions[unit_id] = [5, target_pos[0], target_pos[1]]
                    else:
                        actions[unit_id] = [0, 0, 0]  # Stay if no valid targets
                else:
                    actions[unit_id] = [action_type, 0, 0]

    
        #print(f (Actions: {actions}")
        
        return actions










    def learn(self, step, last_obs, actions, obs, rewards, dones):
        if not self.training or len(self.memory) < self.batch_size:
          return
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device)
            
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if step % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #print(f"Loss: {loss.item()} Epsilon: {self.epsilon} Score: {rewards} Step: {step}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'dqn_model_{self.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'dqn_model_{self.player}.pth')
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.player}")