from lux.utils import direction_to
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class ShipMemory:
    def __init__(self, ship_id, device=torch.device("cpu")):
        self.ship_id = ship_id
        self.device = device
        self.clear_memory()
        
    def clear_memory(self):
        self.states = []
        self.action_types = []
        self.detail_actions = []
        self.type_logprobs = []
        self.detail_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.returns = None
        self.advantages = None
        self.steps_collected = 0

class PPO_Model(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99,lam=0.95,eps_clip=0.15, lr=0.0003, K_epochs=4, update_timestep=2000,device=torch.device("cpu"),entropy_coef=0.01,agent=None):
        super(PPO_Model, self).__init__()
        self.gamma = gamma
        self.lam=lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.policy = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.agent = agent
    def compute_gae(self, transitions, next_value):
        values = [t.value.item() for t in transitions]
        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        values.append(next_value)
        
        gae = 0
        advantages = []
        for step in reversed(range(len(transitions))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        # Convert to torch Tensor
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Returns (target for value function)
        returns = advantages + torch.FloatTensor(values[:-1])
        return advantages, returns
    def finish_rollout(self, next_value=0.0):
        
        transitions = self.memory
        
        # Arrays for each piece of data
        values = [t.value.item() for t in transitions]
        rewards = [t.reward for t in transitions]
        dones = [t.done for t in transitions]
        log_probs = [t.log_prob for t in transitions]
        states = [t.state for t in transitions]
        actions = [t.action for t in transitions]
        
        # We'll append the next_value at the end for advantage calculation
        values.append(next_value)

        # GAE / Advantage calculation
        gae = 0
        advantages = []
        for step in reversed(range(len(transitions))):
            delta = rewards[step] + GAMMA * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        # Convert lists to tensors
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages for numerical stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Returns (target for value function)
        returns = advantages + torch.FloatTensor(values[:-1])
        
        self.memory = []  # Clear memory
        return states, actions, log_probs, returns, advantages


    def update(self, states, actions, old_log_probs, returns, advantages):
        pass
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Common feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor = nn.Linear(64, action_dim)
        
        # Critic head
        self.critic = nn.Linear(64, 1)
        
    def forward(self, state):
        x = self.shared(state)
        
        # Policy logits and value
        policy_logits = self.actor(x)
        value = self.critic(x)
        
        return policy_logits, value
    
    def get_action(self, state):
        policy_logits, value = self.forward(state)
        
        # Categorical distribution
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), dist, value


    
def get_state(unit_pos,nearest_relic_node_position,unit_energy):
    state = torch.FloatTensor([unit_pos[0],unit_pos[1],nearest_relic_node_position[0],nearest_relic_node_position[1]],unit_energy)
    return state






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
        self.ppo = PPO_Model(
            state_dim=self.state_dim,
            lr=0.0002,
            gamma=0.99,
            eps_clip=0.15,
            K_epochs=4,
            device=self.device,
            target_steps=2048,
            entropy_coef=0.01,
            agent=self  
        )
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


        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
            
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
        state=get_state(unit_pos,nearest_relic_node_position,unit_energy)

        return actions
