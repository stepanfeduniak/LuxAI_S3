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
class PPO_Model(nn.Module):
    def __init__(self, state_dim, action_dim, gamma=0.99,lam=0.95,eps_clip=0.15, lr=0.0003, K_epochs=4, update_timestep=2000,device=torch.device("cpu"),entropy_coef=0.001,agent=None):
        super(PPO_Model, self).__init__()
        self.gamma = gamma
        self.lam=lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim,action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    def compute_gae(self, memory):
        rewards = memory.rewards
        values = memory.values
        dones = memory.is_terminals

        # Convert to python list or torch
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        values.append(memory.next_value)  # bootstrap

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
        
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        returns = returns.to(self.device).detach()
        advantages = advantages.to(self.device).detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) \
                   + 0.5 * F.mse_loss(state_values.squeeze(-1), returns) \
                   - self.entropy_coef * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

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
    def evaluate(self, state, action):
        policy_logits, value = self.forward(state)
        
        # Categorical distribution
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        # Calculate the log probabilities
        logprobs = dist.log_prob(action)
        
        # Entropy
        dist_entropy = dist.entropy()
        
        return logprobs, value, dist_entropy
    
    def get_action(self, state):
        policy_logits, value = self.forward(state)
        
        # Categorical distribution
        dist = torch.distributions.Categorical(logits=policy_logits)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), dist, value


    
def get_state(unit_pos, nearest_relic_node_position, unit_energy):
    state = [unit_pos[0], unit_pos[1],
             nearest_relic_node_position[0], nearest_relic_node_position[1],
             unit_energy]
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
        self.action_dim = 5
        self.ppo = PPO_Model(
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        self.load_model()
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

        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

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
        """for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
            else:
                nearest_relic_node_position = np.array([0,0])
            state=get_state(unit_pos,nearest_relic_node_position,unit_energy)
            action,_,_,_ = self.ppo.policy_old.get_action(state)
            actions[unit_id] = action"""
        if step % 100 == 0 and len(self.memory.states) > 0:
            # you might want to set memory.next_value = <some bootstrap value>
            self.memory.next_value = 0.0
            self.ppo.update(self.memory)
            self.memory.clear_memory()

        return actions
    def calculate_rewards_and_dones(self,obs,new_points,done):
        unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        reward=-manhattan_distance(unit_positions[available_unit_ids[0]],[23,23])
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)


        
    def save_model(self):
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f'modelPPO_{self.agent.player}.pth')

    def load_model(self):
        try:
            checkpoint = torch.load(f'modelPPO_{self.agent.player}.pth')
            self.ppo.policy.load_state_dict(checkpoint['policy'])
            self.ppo.policy_old.load_state_dict(checkpoint['policy_old'])
            self.ppo.optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            raise FileNotFoundError(f"No trained model found for {self.agent.player}")
