import torch
import numpy as np
class Playing_Map():
    def __init__(self,player_id,map_size,unit_channels=2,map_channels=5,relic_channels=3):
        self.player_id=player_id
        self.size=map_size
        self.map_channels=map_channels #
        self.unit_channels=unit_channels
        self.relic_channels=relic_channels
        self.channels=map_channels+unit_channels+relic_channels
        self.map_map=torch.zeros((map_size,map_size,map_channels))-1
        self.unit_map=torch.zeros((map_size,map_size,unit_channels))
        self.relic_map=torch.zeros((map_size,map_size,relic_channels))-1
    def add_relic(self, pos):
        x, y = pos
        # Use self.size instead of hardcoded numbers so that we cover a 5x5 area properly.
        self.relic_map[max(0, x - 2):min(self.size, x + 3),
                        max(0, y - 2):min(self.size, y + 3), 0] = 1
        #marking potentials
        self.relic_map[max(0, x - 2):min(self.size, x + 3),
                        max(0, y - 2):min(self.size, y + 3), 1] = 1
        self.relic_map[x, y, 2] = 1

    def locate_new_reward_source(self, obs, last_obs):
        """
        Update the relic map's reward estimates based on the discrepancy between the observed reward
        (team points change) and the sum of reward estimates for tiles occupied by our units.

        The relic_map's first channel (index 0) stores the estimated reward value (clipped to [0, 1]) for each tile.
        If a unit is standing on a tile, that tile is assumed to contribute its estimated reward (up to 1) per step.
        
        This function computes:
        - actual_reward: The change in team points between obs and last_obs.
        - expected_reward: The sum of current reward estimates on all tiles where at least one unit is present.
        - discrepancy = actual_reward - expected_reward

        Then, it distributes a small update (delta) to all occupied tiles, nudging their estimates in the direction
        that reduces the discrepancy. This way, over many steps the relic_map learns which board locations are actually
        "rewarding" (up to a maximum of 1 per tile per step).
        
        Args:
            obs (dict): Current observation.
            last_obs (dict): Previous observation.
        """

        # 1. Calculate the actual reward gained this step.
        actual_reward = obs["team_points"][self.player_id] - last_obs["team_points"][self.player_id]
        # 2. Build an occupancy grid for our units.
        occupancy = torch.zeros((self.size, self.size), dtype=torch.float32)
        unit_us=np.array(obs["units"]["position"][self.player_id])
        unit_mask_us=np.array(obs["units_mask"][self.player_id])
        active_positions = torch.from_numpy(unit_us[unit_mask_us].T)
        
        if self.player_id == 1:
            active_positions = torch.stack([23 - active_positions[1], 23 - active_positions[0]])
            if active_positions.dim() == 1:
                active_positions = active_positions.unsqueeze(1)

        active_positions=active_positions.T
        sure_occupancy=torch.zeros((self.size, self.size), dtype=torch.float32)
        for pos in active_positions:
            x, y = pos.tolist()
            # Ensure the position is within bounds.
            if 0 <= x < self.size and 0 <= y < self.size:
                if self.relic_map[x, y, 1] == 1 and self.relic_map[x, y, 0] < 1.2:
                    occupancy[x, y] = 1.0
                elif self.relic_map[x, y, 0] >= 1.2:
                    sure_occupancy[x,y] =1.0

        # 3. Compute the expected reward from our current reward estimates.
        #    We assume self.relic_map[:, :, 0] holds the reward estimate for each tile.
        #    Clip these estimates to [0, 1] (since maximum reward per tile is 1).
        reward_estimates = self.relic_map[:, :, 0].clamp(0, 1)
        # Only consider tiles where a unit is present.
        active_reward_estimates = reward_estimates[occupancy.bool()]
        expected_reward = active_reward_estimates.sum().item()+sure_occupancy.sum().item()

        # 4. Compute the discrepancy between observed and expected reward.
        discrepancy = actual_reward - expected_reward
        # If there are no occupied tiles, there is nothing to update.
        num_active = active_reward_estimates.numel()
        if num_active == 0:
            return

        # 5. Distribute the discrepancy equally among all occupied (active) tiles.
        #    Use a learning rate to control the update magnitude.
        lr = 0.8  # Adjust this value as needed.
        delta = lr * discrepancy / num_active

        # 6. Update the reward estimates on occupied tiles.
        indices = torch.nonzero(occupancy, as_tuple=True)
        self.relic_map[indices[0], indices[1], 0] += delta
        self.relic_map[23-indices[1], 23-indices[0], 0] = torch.transpose(torch.flip(self.relic_map, dims=[0,1]),1,0)[23-indices[1], 23-indices[0], 0]


        self.relic_map[:, :, 0] = self.relic_map[:, :, 0].clamp(-1, 3)




        
    def update_map(self,obs,last_obs=None):
        #Visibility, right now
        
        #Update map
        if self.player_id==1:
            visibility=torch.transpose(torch.flip(torch.from_numpy(np.array(obs["sensor_mask"])), dims=[0,1]),1,0)
            rows, cols = torch.where(visibility)
            self.map_map[rows, cols,1:4] = torch.nn.functional.one_hot(torch.transpose(torch.flip(torch.from_numpy(np.array(obs['map_features']['tile_type'])), dims=[0,1]),1,0)[visibility].long(), num_classes=3).float()
            self.map_map[:,:,0]=visibility.int()
            self.map_map[rows,cols,4]= torch.transpose(torch.flip(torch.from_numpy(np.array(obs['map_features']['energy'])), dims=[0,1]),1,0)[visibility].float()/8
            #Update reverse
            self.map_map[23-cols, 23-rows, 1:4] = torch.transpose(torch.flip(self.map_map, dims=[0,1]),1,0)[23-cols, 23-rows, 1:4]
            # might be incorrect:
            self.map_map[23-cols, 23-rows, 4] = torch.transpose(torch.flip(self.map_map, dims=[0,1]),1,0)[23-cols, 23-rows, 4]
        else:
            visibility=torch.from_numpy(np.array(obs["sensor_mask"]))
            rows, cols = torch.where(visibility)
            self.map_map[rows, cols,1:4] = torch.nn.functional.one_hot(torch.from_numpy(np.array(obs['map_features']['tile_type']))[visibility].long(), num_classes=3).float()
            self.map_map[:,:,0]=visibility.int()
            self.map_map[rows,cols,4]= torch.from_numpy(np.array(obs['map_features']['energy']))[visibility].float()/8
            #Update reverse
            self.map_map[23-cols, 23-rows, 1:4] = torch.transpose(torch.flip(self.map_map, dims=[0,1]),1,0)[23-cols, 23-rows, 1:4]
            # might be incorrect:
            self.map_map[23-cols, 23-rows, 4] = torch.transpose(torch.flip(self.map_map, dims=[0,1]),1,0)[23-cols, 23-rows, 4]
        #Update units
        unit_us=np.array(obs["units"]["position"][self.player_id])
        unit_mask_us=np.array(obs["units_mask"][self.player_id])
        valid_positions_us = torch.from_numpy(unit_us[unit_mask_us].T)
        
        if self.player_id == 1:
            valid_positions_us = torch.stack([23 - valid_positions_us[1], 23 - valid_positions_us[0]])
            if valid_positions_us.dim() == 1:
                valid_positions_us = valid_positions_us.unsqueeze(1)

    
        values_us = torch.ones(valid_positions_us.shape[1], dtype=torch.float32)
        
        self.unit_map[:,:,0]=0
        self.unit_map[:,:,1]/=2
        self.unit_map[:,:,0].index_put_(
                    (torch.as_tensor(valid_positions_us[0], dtype=torch.long),
                    torch.as_tensor(valid_positions_us[1], dtype=torch.long)),
                    torch.as_tensor(values_us, dtype=torch.float32),
                    accumulate=True
)

        unit_mask_them=np.array(obs["units_mask"][1-self.player_id])
        unit_them=np.array(obs["units"]["position"][1-self.player_id])
        valid_positions_them = torch.from_numpy(unit_them[unit_mask_them].T)
        
        
        if self.player_id == 1:
            valid_positions_them = torch.stack([23 - valid_positions_them[1], 23 - valid_positions_them[0]])
            if valid_positions_them.dim() == 1:
                valid_positions_them = valid_positions_them.unsqueeze(1)
        values_them = torch.ones(valid_positions_them.shape[1], dtype=torch.float32)
        self.unit_map[:,:,1].index_put_(
                (torch.as_tensor(valid_positions_them[0], dtype=torch.long),
                torch.as_tensor(valid_positions_them[1], dtype=torch.long)),
                torch.as_tensor(values_them, dtype=torch.float32),
                accumulate=True
)
        relics = torch.from_numpy(np.array(obs["relic_nodes"]))
        relics_mask = torch.from_numpy(np.array(obs["relic_nodes_mask"]))
        relics_pos = relics[relics_mask].T  # shape: [2, N]
        if relics_pos.dim() == 1:
            relics_pos = relics_pos.unsqueeze(0)
        for x, y in zip(relics_pos[0].tolist(), relics_pos[1].tolist()):
            if self.relic_map[x, y, 2] != 1:
                self.add_relic((x, y))
                self.add_relic((23-y, 23-x))
        
        # If a previous observation is provided, adjust the relic map to add new reward sources.
        if last_obs is not None:
            self.locate_new_reward_source(obs, last_obs)



    def map_stack(self):
        map_stack = torch.cat(
                                [
                                self.map_map,    # shape = [H, W, map_channels]
                                self.unit_map,   # shape = [H, W, unit_channels]
                                self.relic_map,  # shape = [H, W, relic_channels]
                                ], 
                                dim=-1  # last dimension, so new shape = (H, W, total_channels)
                            )
        map_stack = map_stack.permute(2, 1, 0)
        return map_stack