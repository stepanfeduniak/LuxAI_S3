import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import torch.nn.functional as F
from lux.utils import direction_to
class Actor(nn.Module):
    """
    Decentralized actor that only sees the local state of its agent.
    """
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)  # important: specify dim for softmax
        )

    def forward(self, x):
        # x shape: [batch_size, obs_dim]
        return self.model(x)
class Critic(nn.Module):
    """
    Centralized critic that sees the global state
    (which can be concatenation of all agents' observations, or any centralized info).
    """
    def __init__(self, global_state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(global_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, global_state_dim]
        return self.model(x)
class MultiAgentMemory:
    """
    We store the transitions for all agents in parallel.
    For each step, we have:
      - log_probs[agent_i]
      - values (the single critic's output, given the global state)
      - rewards[agent_i]
      - done
    """
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.log_probs = [[] for _ in range(n_agents)]
        self.rewards = [[] for _ in range(n_agents)]
        self.values = []
        self.dones = []

    def add(self, log_probs, value, rewards, done):
        """
        log_probs: list of size n_agents (log prob of each agent's chosen action)
        value: single float/ tensor from the critic
        rewards: list of size n_agents (reward for each agent)
        done: bool
        """
        for i in range(self.n_agents):
            self.log_probs[i].append(log_probs[i])
            self.rewards[i].append(rewards[i])
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.rewards = [[] for _ in range(self.n_agents)]
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.values)


# =================================
# 4) A2C Training Function
# =================================

def train_on_episode_memory(memory: MultiAgentMemory,
                           actor: Actor,
                           critic: Critic,
                           actor_optim: optim.Optimizer,
                           critic_optim: optim.Optimizer,
                           gamma=0.99):
    """
    Perform a single A2C update after one rollout (episode).
    """
    # Convert stored data into tensors
    values = torch.stack(memory.values)  # shape: [T, 1]
    T = values.size(0)

    # For advantage computation, we need the "next value" at each step
    # We'll do this in reverse, as usual in A2C.
    # We'll keep separate returns for each agent if rewards are separate.
    # But the critic is centralized => it produces a single state-value for the global state.
    # We'll compute G_t (the return) for each agent individually.
    # Then advantage = G_t - V(s_t).
    returns_per_agent = [np.zeros((T, ), dtype=np.float32) for _ in range(memory.n_agents)]

    next_value = 0.0  # because we assume episode is done, so V(s_{T+1})=0
    for i_agent in range(memory.n_agents):
        running_return = 0.0
        for t in reversed(range(T)):
            if memory.dones[t]:
                next_value = 0.0
                running_return = 0.0
            # G_t = r_t + gamma * V(s_{t+1}) (for advantage),
            # but we accumulate in running_return
            running_return = memory.rewards[i_agent][t] + gamma * running_return
            returns_per_agent[i_agent][t] = running_return

    # Convert to Torch
    all_advantages = []
    for i_agent in range(memory.n_agents):
        # shape [T]
        returns = torch.from_numpy(returns_per_agent[i_agent])
        # shape [T,1]
        advantages = returns.unsqueeze(-1) - values.detach()
        all_advantages.append(advantages)

    # Critic loss is computed from the perspective of the global state,
    # but each agent's return is its own. Often in cooperative settings,
    # you might use the sum or average of rewards. For demonstration,
    # we do a simple sum of MSE across agents.
    # shape: [T,1]
    stacked_advantages = torch.stack(all_advantages, dim=0)  # [n_agents, T, 1]
    critic_loss = (stacked_advantages ** 2).mean()

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Actor loss
    # Combine all agents' losses. Actor for agent i uses log_probs[i].
    # advantage for agent i is all_advantages[i].
    actor_loss = 0
    for i_agent in range(memory.n_agents):
        log_probs_i = torch.stack(memory.log_probs[i_agent])  # shape [T]
        advantages_i = all_advantages[i_agent].detach()       # shape [T,1]
        # We need them to be consistent shape for multiplication
        # log_probs_i -> [T,1], broadcast with advantages_i
        log_probs_i = log_probs_i.unsqueeze(-1)
        actor_loss += (- log_probs_i * advantages_i).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    return actor_loss.item(), critic_loss.item()
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
        self.obs_size = 6  # unit_pos(2) + closest_relic(2) + unit_energy(1)+ step(1)+unit_id(1)
        self.global_obs= 6*16
        self.action_size = 6 # stay, up, right, down, left, sap
        self.hidden_size = 128
        self.batch_size = 64
        self.gamma = 0.96
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(obs_dim=self.obs_size, n_actions=self.action_size)
        self.critic = Critic(global_state_dim=self.global_obs)
        self.memory = MultiAgentMemory(16)
        actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        



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
        
            dist = self.policy_net(state)
            action_type = dist.sample()
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

    def train_on_episode_memory(self,memory: MultiAgentMemory,
                            actor: Actor,
                            critic: Critic,
                            actor_optim: optim.Optimizer,
                            critic_optim: optim.Optimizer,
                            gamma=0.99):
        """
        Perform a single A2C update after one rollout (episode).
        """
        # Convert stored data into tensors
        values = torch.stack(memory.values)  # shape: [T, 1]
        T = values.size(0)

        # For advantage computation, we need the "next value" at each step
        # We'll do this in reverse, as usual in A2C.
        # We'll keep separate returns for each agent if rewards are separate.
        # But the critic is centralized => it produces a single state-value for the global state.
        # We'll compute G_t (the return) for each agent individually.
        # Then advantage = G_t - V(s_t).
        returns_per_agent = [np.zeros((T, ), dtype=np.float32) for _ in range(memory.n_agents)]

        next_value = 0.0  # because we assume episode is done, so V(s_{T+1})=0
        for i_agent in range(memory.n_agents):
            running_return = 0.0
            for t in reversed(range(T)):
                if memory.dones[t]:
                    next_value = 0.0
                    running_return = 0.0
                # G_t = r_t + gamma * V(s_{t+1}) (for advantage),
                # but we accumulate in running_return
                running_return = memory.rewards[i_agent][t] + gamma * running_return
                returns_per_agent[i_agent][t] = running_return

        # Convert to Torch
        all_advantages = []
        for i_agent in range(memory.n_agents):
            # shape [T]
            returns = torch.from_numpy(returns_per_agent[i_agent])
            # shape [T,1]
            advantages = returns.unsqueeze(-1) - values.detach()
            all_advantages.append(advantages)

        # Critic loss is computed from the perspective of the global state,
        # but each agent's return is its own. Often in cooperative settings,
        # you might use the sum or average of rewards. For demonstration,
        # we do a simple sum of MSE across agents.
        # shape: [T,1]
        stacked_advantages = torch.stack(all_advantages, dim=0)  # [n_agents, T, 1]
        critic_loss = (stacked_advantages ** 2).mean()

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        actor_loss = 0
        for i_agent in range(memory.n_agents):
            log_probs_i = torch.stack(memory.log_probs[i_agent])  # shape [T]
            advantages_i = all_advantages[i_agent].detach()       # shape [T,1]
            # We need them to be consistent shape for multiplication
            # log_probs_i -> [T,1], broadcast with advantages_i
            log_probs_i = log_probs_i.unsqueeze(-1)
            actor_loss += (- log_probs_i * advantages_i).mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        return actor_loss.item(), critic_loss.item()

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