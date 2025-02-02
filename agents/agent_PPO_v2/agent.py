import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from lux.utils import direction_to
from map_processing import Playing_Map


class ShipMemory:
    def __init__(self, ship_id, device=torch.device("cpu")):
        self.ship_id = ship_id
        self.device = device
        self.clear_memory()
        self.steps_collected = 0    

    def clear_memory(self):
        self.unit_states = []
        self.map_states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.next_value = 0.0  # for convenience if you want bootstrap

    def get_batch(self):
        return {
            'unit_states': self.unit_states,
            'map_states': self.map_states,
            'actions': self.actions,
            'logprobs': self.logprobs,
            'rewards': self.rewards,
            'is_terminals': self.is_terminals,
            'values': self.values
        }


class FleetMemory:
    def __init__(self, max_ships, device=torch.device("cpu")):
        self.max_ships = max_ships
        self.ships = [ShipMemory(i, device) for i in range(max_ships)]
        
    def clear_memory(self):
        for ship in self.ships:
            ship.clear_memory()


class MapEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(24 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, H, W]
        x = self.conv(x)
        x = self.fc(x)
        return x  # [batch_size, out_dim]


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
        # x: [batch_size, in_dim]
        return self.fc(x)


class Actor(nn.Module):
    def __init__(self, map_encoding_dim, unit_feature_dim, n_actions):
        super().__init__()
        self.unit_enc = UnitEncoder(in_dim=unit_feature_dim, out_dim=64)
        self.policy_head = nn.Sequential(
            nn.Linear(map_encoding_dim + 64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, map_encoding, unit_input):
        """
        map_encoding: [batch_size, map_encoding_dim] or [1, map_encoding_dim]
        unit_input:   [batch_size, unit_feature_dim]
        """
        unit_feats = self.unit_enc(unit_input)  # => [batch_size, 64]

        # If map_encoding is [1, map_dim], repeat it for each unit
        if map_encoding.shape[0] == 1 and unit_feats.shape[0] > 1:
            map_encoding = map_encoding.repeat(unit_feats.shape[0], 1)
        
        combined = torch.cat([map_encoding, unit_feats], dim=1)  # => [batch_size, map_encoding_dim + 64]
        action_probs = self.policy_head(combined)  # => [batch_size, n_actions]
        return action_probs


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
        # map_encoding: [batch_size, map_encoding_dim]
        value = self.value_head(map_encoding)  # => [batch_size, 1]
        return value


class ActorCritic(nn.Module):
    def __init__(self, map_channels_input, unit_feature_dim, action_dim):
        super().__init__()
        self.map_encoding_dim = 128
        
        # use map_channels_input in the encoder
        self.mapencoder = MapEncoder(in_channels=map_channels_input, out_dim=self.map_encoding_dim)
        
        self.actor = Actor(map_encoding_dim=self.map_encoding_dim,
                           unit_feature_dim=unit_feature_dim,
                           n_actions=action_dim)
        self.critic = Critic(map_encoding_dim=self.map_encoding_dim)

    def forward(self, unit_states, map_state):
        # map_state: [batch_size, map_channels_input, H, W]
        # unit_states: [batch_size, unit_feature_dim]
        map_encoding = self.mapencoder(map_state)  # => [batch_size, 128] or [1,128] if you only pass batch=1
        policy = self.actor(map_encoding, unit_states)
        value = self.critic(map_encoding)
        return policy, value

    def evaluate(self, unit_states, map_states, action):
        """
        Evaluate the log prob of an already selected action, the value, etc.
        Re-using 'unit_states' and 'map_states' in a batch form.
        """
        map_encoding = self.mapencoder(map_states)   # => [batch_size, 128]
        policy = self.actor(map_encoding, unit_states)
        value = self.critic(map_encoding)            # => [batch_size, 1]
        
        dist = torch.distributions.Categorical(probs=policy)
        logprobs = dist.log_prob(action)             # => [batch_size]
        dist_entropy = dist.entropy()                # => [batch_size]
        return logprobs, value, dist_entropy
    
    def get_action(self, unit_states, map_states):
        """
        Sample an action. 
        unit_states: [batch_size, unit_feature_dim]
        map_states:  [batch_size, map_channels, H, W] 
                     or [1, map_channels, H, W] if the map is the same for all units
        """
        policy, value = self.forward(unit_states, map_states)
        dist = torch.distributions.Categorical(probs=policy)
        if value.shape[0] == 1 and policy.shape[0] > 1:
            value = value.repeat(policy.shape[0], 1)
        actions = dist.sample()          # [batch_size]
        log_probs = dist.log_prob(actions)  # [batch_size]
        
        return actions, log_probs, dist, value


class PPO_Model(nn.Module):
    def __init__(self,
                 map_channels_input,
                 unit_feature_dim,
                 action_dim,
                 gamma=0.95,
                 lam=0.95,
                 eps_clip=0.15,
                 lr=1e-4,
                 K_epochs=2,
                 update_timestep=1,
                 device=torch.device("cpu"),
                 entropy_coef=0.001,
                 agent=None):
        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.update_timestep = update_timestep
        self.step_count = 0
        
        # build the policy networks
        self.policy = ActorCritic(
            map_channels_input=map_channels_input,
            unit_feature_dim=unit_feature_dim,
            action_dim=action_dim
        ).to(self.device)
        self.policy_old = ActorCritic(
            map_channels_input=map_channels_input,
            unit_feature_dim=unit_feature_dim,
            action_dim=action_dim
        ).to(self.device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.agent = agent

    def compute_gae(self, trajectory):
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['is_terminals']

        # Convert values to floats
        values = [v.item() if isinstance(v, torch.Tensor) else v for v in values]
        values.append(trajectory["next_value"])  # bootstrap value

        advantages = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * values[step+1] * mask - values[step]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        # Use unbiased=False to avoid NaN when tensor has one element
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)
        return advantages, returns

    
    def get_trajectories(self, total_fleet_memory):
        trajectories = []
        for fleet_memory in total_fleet_memory:
            for ship in fleet_memory.ships:
                # fix: was "if len(ship.states) > 0:", should check "unit_states"
                if len(ship.unit_states) > 0:
                    trajectory = {
                        'unit_states': torch.stack(ship.unit_states).to(ship.device),
                        'map_states': torch.stack(ship.map_states).to(ship.device),
                        'actions': torch.stack(ship.actions).to(ship.device),
                        'logprobs': torch.stack(ship.logprobs).to(ship.device),
                        'rewards': torch.tensor(ship.rewards).to(ship.device),
                        'is_terminals': torch.tensor(ship.is_terminals).to(ship.device),
                        'values': torch.stack(ship.values).to(ship.device),  # ensure shape
                        'next_value': ship.next_value
                    }
                    # compute advantages
                    #print(f"Computing GAE for ship {ship.ship_id}")
                    #print(f"length of actions: {len(trajectory['actions'])}, logprobs: {len(trajectory['logprobs'])}, rewards: {len(trajectory['rewards'])},values: {len(trajectory['values'])}, is_terminals: {len(trajectory['is_terminals'])}")
                    advantages, returns = self.compute_gae(trajectory)
                    trajectory['advantages'] = advantages
                    trajectory['returns'] = returns
                    trajectories.append(trajectory)
        #print(f"Collected {len(trajectories)} trajectories")
        return trajectories
    
    @staticmethod
    def trajectories_to_train_data(trajectories):
        """
        Combine all single-ship trajectories into a single batch
        """
        unit_states = torch.cat([traj['unit_states'] for traj in trajectories], dim=0)
        map_states = torch.cat([traj['map_states'] for traj in trajectories], dim=0)
        actions = torch.cat([traj['actions'] for traj in trajectories], dim=0)
        logprobs = torch.cat([traj['logprobs'] for traj in trajectories], dim=0)
        advantages = torch.cat([traj['advantages'] for traj in trajectories], dim=0)
        returns = torch.cat([traj['returns'] for traj in trajectories], dim=0)
        print(returns.shape)
        return unit_states, map_states, actions, logprobs, advantages, returns
    
    @staticmethod
    def ppo_data_loader(unit_states, map_states, actions, logprobs, advantages, returns, batch_size):
        dataset_size = unit_states.size(0)
        indices = torch.randperm(dataset_size)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            yield (unit_states[batch_indices],
                   map_states[batch_indices],
                   actions[batch_indices],
                   logprobs[batch_indices],
                   advantages[batch_indices],
                   returns[batch_indices])

    def update(self, fleet_memory):
        trajectories = self.get_trajectories(fleet_memory)
        if len(trajectories) == 0:
            return  # no data
        (old_unit_states, old_map_states,
         old_actions, old_logprobs,
         all_advantages, all_returns) = self.trajectories_to_train_data(trajectories)

        for _ in range(self.K_epochs):
            for (unit_states_b, map_states_b, actions_b, logprobs_b,
                 advantages_b, returns_b) in self.ppo_data_loader(
                     old_unit_states, old_map_states, old_actions, old_logprobs,
                     all_advantages, all_returns, batch_size=128):

                # Evaluate with current policy
                logprobs_new, state_values, dist_entropy = self.policy.evaluate(
                    unit_states_b, map_states_b, actions_b
                )
                
                # ratio = exp(new_logprob - old_logprob)
                ratios = torch.exp(logprobs_new - logprobs_b)

                # fix: use advantages_b for surr1, surr2
                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_b

                # policy loss
                actor_loss = -torch.min(surr1, surr2)
                # critic loss
                critic_loss = F.mse_loss(state_values.squeeze(-1), returns_b)
                # total loss
                loss = actor_loss + 0.2 * critic_loss - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        self.step_count += 1
        if self.step_count >= self.update_timestep:
            self.update_policy()

    def update_policy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.step_count = 0

    def act(self, unit_states, map_stack, fleet_memory, unit_ids):
        """
        unit_states: shape [num_units, unit_feature_dim]
        map_stack:   shape [1, map_channels, H, W], 
                     or [num_units, map_channels, H, W]
        """
        unit_states_t = torch.as_tensor(unit_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # sample action using old policy
            actions_t, log_probs_t, dist, values_t = self.policy_old.get_action(
                unit_states_t, map_stack
            )
            # actions_t: [num_units]
            # values_t:  [num_units, 1]
        
        actions_np = actions_t.cpu().numpy()  # shape [num_units]

        # record in memory
        for i, unit_id in enumerate(unit_ids):
            fleet_memory.ships[unit_id].unit_states.append(unit_states_t[i].clone())
            fleet_memory.ships[unit_id].map_states.append(map_stack.squeeze(0).clone())
            fleet_memory.ships[unit_id].actions.append(actions_t[i].clone())
            fleet_memory.ships[unit_id].logprobs.append(log_probs_t[i].clone())
            fleet_memory.ships[unit_id].values.append(values_t[i].clone())

        return actions_np


def get_state(unit_pos, nearest_relic_node_position, unit_energy):
    return [
        unit_pos[0] / 11.5 - 1,
        unit_pos[1] / 11.5 - 1,
        nearest_relic_node_position[0] / 11.5 - 1,
        nearest_relic_node_position[1] / 11.5 - 1,
        unit_energy / 200 - 1
    ]


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Agent:
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id
        self.env_cfg = env_cfg

        self.state_dim = 5
        self.action_dim = 5  # 5 discrete actions
        self.device = torch.device("cpu")
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.clear_track_times()
        # memory
        self.fleet_mem = FleetMemory(env_cfg["max_units"], device=self.device)

        # map
        self.play_map = Playing_Map(
            player_id=self.team_id, 
            map_size=env_cfg["map_width"],
            unit_channels=2,
            map_channels=4,
            relic_channels=3
        )

        # PPO model
        self.ppo = PPO_Model(
            map_channels_input=9,
            unit_feature_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        # try loading existing weights
        try:
            self.load_model()
        except FileNotFoundError:
            pass

    def act(self, step, obs, remainingOverageTime=60):
        # extract data from obs
        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])  # shape (max_units, 1)
        observed_relic_node_positions = np.array(obs["relic_nodes"])  # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"])
        
        # map update
        self.play_map.update_map(obs)
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
                nrp = self.relic_node_positions[0]  # just pick first for now
            else:
                nrp = np.array([0,0])
            st = get_state(unit_pos, nrp, unit_energy)
            states_list.append(st)

        if len(available_unit_ids) == 0:
            return np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # get actions from the old policy
        actions_np = self.ppo.act(
            unit_states=states_list,
            map_stack=map_data_single,
            fleet_memory=self.fleet_mem,
            unit_ids=available_unit_ids
        )
        # build final action array
        actions_array = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, unit_id in enumerate(available_unit_ids):
            actions_array[unit_id, 0] = actions_np[i]
            actions_array[unit_id, 1] = 0
            actions_array[unit_id, 2] = 0
        
            


        return actions_array
    def return_memories(self):
        for ship in self.fleet_mem.ships:
            ship.next_value = 0.0
        return self.fleet_mem
    def clear_memories(self):
        self.fleet_mem= FleetMemory(self.env_cfg["max_units"], device=self.device)

    def update_ppo(self,total_memory):
        #print("Updating PPO model...")
        time_start = time.time()
        self.ppo.update(total_memory)
        time_end = time.time()
        self.track_times["update_ppo"]+=time_end-time_start

    def calculate_rewards_and_dones(self, obs, last_obs, env_cfg, new_points, done,old_available_unit_ids):
        """unit_mask = np.array(obs["units_mask"][self.team_id])
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        available_unit_ids = np.where(unit_mask)[0]
        new_visible_tiles = np.sum(np.bitwise_xor(obs["sensor_mask"], last_obs["sensor_mask"]))
        visibility_coef=1/(((env_cfg["unit_sensor_range"]*2+1)**(3/2))*(24**2))
        visible_tiles=np.sum(obs["sensor_mask"])
        shared_reward=0

        shared_reward+=visibility_coef*visible_tiles"""
        exploration_factor = 0.005    # per newly discovered tile
        combat_factor = 0.2           # per enemy unit “destroyed”
        energy_factor = 0.05          # bonus for high normalized energy
        proximity_factor = 0.1        # bonus for being near a relic
        movement_factor = 0.01        # bonus per Manhattan distance moved

        team = self.team_id
        opp_team = self.opp_team_id

        # --- Global rewards based on observations ---
        # (a) Exploration: count only those tiles that have become visible now.
        new_visible = np.logical_and(obs["sensor_mask"], np.logical_not(last_obs["sensor_mask"]))
        visible_tiles = np.sum(obs["sensor_mask"])
        global_exploration_reward = exploration_factor * visible_tiles

        # (b) Combat: count the drop in visible enemy ships.
        enemy_count_last = np.sum(last_obs["units_mask"][opp_team])
        enemy_count_current = np.sum(obs["units_mask"][opp_team])
        enemy_destroyed = max(0, enemy_count_last - enemy_count_current)
        global_combat_reward = enemy_destroyed * combat_factor

        # (c) Relic: reward any increase in team relic points.
        # new_points is assumed to be (current team_points - previous team_points)
        global_relic_reward = new_points

        # Distribute these global rewards evenly among ships that were available last step.
        num_active = len(old_available_unit_ids)
        per_unit_exploration = global_exploration_reward / max(1, num_active)
        per_unit_combat = global_combat_reward / max(1, num_active)
        per_unit_relic = global_relic_reward / max(1, num_active)

        # --- Per-ship individual rewards ---
        team_positions = np.array(obs["units"]["position"][team])
        team_energies = np.array(obs["units"]["energy"][team])
        last_team_positions = np.array(last_obs["units"]["position"][team])
        
        total_reward_sum = 0.0

        for unit_id in old_available_unit_ids:
            unit_reward = 0.0

            # (1) Movement reward: if this ship was active last timestep, reward distance moved.
            if last_obs["units_mask"][team][unit_id]:
                prev_pos = last_team_positions[unit_id]
                curr_pos = team_positions[unit_id]
                move_distance = abs(curr_pos[0] - prev_pos[0]) + abs(curr_pos[1] - prev_pos[1])
                unit_reward += movement_factor * move_distance

            # (2) Energy reward: bonus proportional to current energy (normalized by 400).
            current_energy = team_energies[unit_id]
            unit_reward += (current_energy / 400.0) * energy_factor

            # (3) Proximity to relic reward: if any relics have been discovered, reward being close.
            if len(self.relic_node_positions) > 0:
                # Compute Manhattan distance to each known relic and select the minimum.
                distances = [abs(team_positions[unit_id][0] - relic[0]) + abs(team_positions[unit_id][1] - relic[1])
                            for relic in self.relic_node_positions]
                nearest_distance = min(distances)
                # Reward gets larger as the distance gets smaller.
                unit_reward += proximity_factor * (1 - (nearest_distance / 48.0))
            
            # (4) Add the distributed global rewards.
            unit_reward += per_unit_exploration + per_unit_combat + per_unit_relic

            # Save this reward and the terminal flag.
            self.fleet_mem.ships[unit_id].rewards.append(unit_reward)
            self.fleet_mem.ships[unit_id].is_terminals.append(done)

            total_reward_sum += unit_reward

        # Return (for logging purposes) the average reward over the ships
        return total_reward_sum / max(1, num_active)

    def save_model(self):
        torch.save({
            'policy': self.ppo.policy.state_dict(),
            'optimizer': self.ppo.optimizer.state_dict()
        }, 'modelPPO.pth')

    def load_model(self):
        checkpoint = torch.load('modelPPO.pth')
        self.ppo.policy.load_state_dict(checkpoint['policy'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer'])
    def clear_track_times(self):
        self.track_times={"update_ppo":0}
    def get_track_times(self):
        return self.track_times
