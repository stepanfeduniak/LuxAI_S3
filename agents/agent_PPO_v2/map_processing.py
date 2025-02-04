import torch
class Playing_Map():
    def __init__(self,player_id,map_size,unit_channels=2,map_channels=5,relic_channels=3):
        self.player_id=player_id
        self.size=map_size
        self.map_channels=map_channels #
        self.unit_channels=unit_channels
        self.relic_channels=relic_channels
        self.channels=map_channels+unit_channels+relic_channels
        self.map_map=torch.zeros((map_size,map_size,map_channels))
        self.unit_map=torch.zeros((map_size,map_size,unit_channels))
        self.relic_map=torch.zeros((map_size,map_size,relic_channels))
    def add_relic(self, pos):
        x, y = pos
        # Use self.size instead of hardcoded numbers so that we cover a 5x5 area properly.
        self.relic_map[max(0, x - 2):min(self.size, x + 3),
                        max(0, y - 2):min(self.size, y + 3), 0] = 1
        self.relic_map[x, y, 2] = 1

    def locate_new_reward_source(self,obs,last_obs):
        unit_us=obs["units"]["position"][self.player_id]
        unit_mask_us=obs["units_mask"][self.player_id]
        valid_positions_us = unit_us[unit_mask_us].T
        expected_reward=sum(self.relic_map[valid_positions_us[0],valid_positions_us[1],0]>=2)
        actual_reward=obs["team_points"][self.player_id]-last_obs["team_points"][self.player_id]
        # 1) Create mask: which entries are > 0?
        musk = self.relic_map[valid_positions_us[0], valid_positions_us[1], 0] > 0

        # 2) Use this mask to index into relic_map and do the in-place add
        try:
            self.relic_map[valid_positions_us[0][musk], valid_positions_us[1][musk], 0] += 0.5 * (actual_reward - expected_reward)
        except:
            print(valid_positions_us)

        
    def update_map(self,obs,last_obs=None):
        #Visibility, right now
        visibility=torch.from_numpy(obs["sensor_mask"])
        #Update map
        rows, cols = torch.where(visibility)
        self.map_map[rows, cols,1:4] = torch.nn.functional.one_hot(torch.from_numpy(obs['map_features']['tile_type'])[visibility].long(), num_classes=3).float()
        self.map_map[:,:,0]=visibility.int()
        self.map_map[rows,cols,4]= torch.from_numpy(obs['map_features']['tile_type'])[visibility].float()
        #Update units
        unit_us=obs["units"]["position"][self.player_id]
        unit_mask_us=obs["units_mask"][self.player_id]
        valid_positions_us = unit_us[unit_mask_us].T
        values_us = torch.ones(valid_positions_us.shape[1], dtype=torch.float32)
        self.unit_map[:,:,0]=0
        self.unit_map[:,:,1]/=2
        self.unit_map[:,:,0].index_put_(
                    (torch.as_tensor(valid_positions_us[0], dtype=torch.long),
                    torch.as_tensor(valid_positions_us[1], dtype=torch.long)),
                    torch.as_tensor(values_us, dtype=torch.float32),
                    accumulate=True
)

        unit_mask_them=obs["units_mask"][1-self.player_id]
        unit_them=obs["units"]["position"][1-self.player_id]
        valid_positions_them = unit_them[unit_mask_them].T
        values_them = torch.ones(valid_positions_them.shape[1], dtype=torch.float32)
        self.unit_map[:,:,1].index_put_(
                (torch.as_tensor(valid_positions_them[0], dtype=torch.long),
                torch.as_tensor(valid_positions_them[1], dtype=torch.long)),
                torch.as_tensor(values_them, dtype=torch.float32),
                accumulate=True
)
        relics = obs["relic_nodes"]
        relics_mask = obs["relic_nodes_mask"]
        relics_pos = relics[relics_mask].T  # shape: [2, N]
        for x, y in zip(relics_pos[0].tolist(), relics_pos[1].tolist()):
            self.add_relic((x, y))
        
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
        map_stack = map_stack.permute(2, 0, 1)
        return map_stack