import os
import json
import map_processing
import time
import numpy as np
import torch
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
def get_state(unit_pos, nearest_relic_node_position, unit_energy,env_cfg):
    return [
        unit_pos[0] / 11.5 - 1,
        unit_pos[1] / 11.5 - 1,
        nearest_relic_node_position[0] / 11.5 - 1,
        nearest_relic_node_position[1] / 11.5 - 1,
        unit_energy / 200 - 1,
        env_cfg['unit_sensor_range']/5,
        env_cfg['unit_move_cost']/10,
        env_cfg['unit_sap_cost']/50,
        env_cfg['unit_sap_range']/5,
        
    ]
def get_data(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None
file_dir = "./agents/behaviour_cloning_agent/41863713_58807861.json"
replay=get_data(file_dir)

#Trajectory is a class
class Unit_Trajectory():
    def __init__(self,unit_id):
        self.unit_id=unit_id
        self.map_state_trajectories=[]
        self.unit_state_trajectories=[]
        self.unit_action_trajectories=[]



team_id=0
#unroll map function
env_cfg=replay["configuration"]["env_cfg"]
curr_map=map_processing.Playing_Map(player_id=0, 
            map_size=env_cfg["map_width"],
            unit_channels=2,
            map_channels=5,
            relic_channels=3)
last_obs=None
all_trajectories=[]
map_trajectory=[]
unit_trajectories=[Unit_Trajectory(unit_id) for unit_id in range(16)]
time_start=time.time()
relic_node_positions=[]
discovered_relic_nodes_ids = set()

for step in range(500):    
    
    next_obs=json.loads(replay["steps"][step][team_id]["observation"]["obs"])
    if step>0:
        curr_map.update_map(next_obs,last_obs)
    observed_relic_node_positions = np.array(next_obs["relic_nodes"])  # shape (max_relic_nodes, 2)
    observed_relic_nodes_mask = np.array(next_obs["relic_nodes_mask"])  # shape (max_relic_nodes, )
    visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
    for rid in visible_relic_node_ids:
        if rid not in discovered_relic_nodes_ids:
            discovered_relic_nodes_ids.add(rid)
            relic_node_positions.append(observed_relic_node_positions[rid])
            curr_map.add_relic(observed_relic_node_positions[rid])
    if next_obs["match_steps"]==0:
        all_trajectories.append(unit_trajectories)
        unit_trajectories=[Unit_Trajectory(unit_id) for unit_id in range(16)]
        print(f"Units cleared, step:{step}")
    last_obs=next_obs
    unit_mask=np.array(next_obs["units_mask"][team_id])
    available_unit_ids = np.where(unit_mask)[0]
    unit_positions = np.array(next_obs["units"]["position"][team_id])  # shape (max_units, 2)
    unit_energys = np.array(next_obs["units"]["energy"][team_id])  # shape (max_units, 1)
    #locate_relic_nodes
    for unit_id in available_unit_ids:
        unit_trajectories[unit_id].map_state_trajectories.append(curr_map.map_stack())
        unit_pos = unit_positions[unit_id]
        unit_energy = unit_energys[unit_id]
        if len(relic_node_positions) > 0:
            nrp = min(relic_node_positions, key=lambda pos: manhattan_distance(unit_pos, pos))
        else:
            nrp = np.array([13,13])
        st = get_state(unit_pos, nrp, unit_energy,env_cfg)
        unit_trajectories[unit_id].unit_state_trajectories.append(st)
        unit_action=replay["steps"][step][0]["action"][unit_id][0]
        unit_trajectories[unit_id].unit_action_trajectories.append(unit_action)
    map_trajectory.append(curr_map.map_stack())
    
    
    

time_finish=time.time()
#len(replay["steps"])
print(len(map_trajectory))
print(len(all_trajectories))
print(f"Time:{time_finish-time_start}")

