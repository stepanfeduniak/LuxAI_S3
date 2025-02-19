import os
import json
import map_processing
import time
import numpy as np
import torch
import h5py

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def manhattan_distance(a, b):
    """
    Computes the Manhattan (L1) distance between two 2D points.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_state(unit_pos, nearest_relic_node_position, unit_energy, env_cfg,team):
    if team==0:
        return [
            unit_pos[0] / 11.5 - 1,
            unit_pos[1] / 11.5 - 1,
            nearest_relic_node_position[0] / 11.5 - 1,
            nearest_relic_node_position[1] / 11.5 - 1,
            unit_energy / 200 - 1,
            env_cfg['unit_sensor_range'] / 5,
            env_cfg['unit_move_cost'] / 10,
            env_cfg['unit_sap_cost'] / 50,
            env_cfg['unit_sap_range'] / 5,
            team
        ]
    else:
        return [
            (23-unit_pos[1]) / 11.5 - 1,
            (23-unit_pos[0]) / 11.5 - 1,
            (23-nearest_relic_node_position[1]) / 11.5 - 1,
            (23-nearest_relic_node_position[0]) / 11.5 - 1,
            unit_energy / 200 - 1,
            env_cfg['unit_sensor_range'] / 5,
            env_cfg['unit_move_cost'] / 10,
            env_cfg['unit_sap_cost'] / 50,
            env_cfg['unit_sap_range'] / 5,
            team
        ]


def get_data(file_path):
    """
    Loads replay data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict or None: Parsed JSON replay data or None if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

# ---------------------------------------------------------------
# Replay Processing Function
# ---------------------------------------------------------------
# In process_replay (modified)
def process_replay(replay_path, team_id):
    replay = get_data(replay_path)
    if replay is None:
        return
    reverse_action_map={0:0,1:3,2:4,3:1,4:2,5:5}
    env_cfg = replay["configuration"]["env_cfg"]
    curr_map = map_processing.Playing_Map(
        player_id=team_id, 
        map_size=env_cfg["map_width"],
        unit_channels=2,
        map_channels=5,
        relic_channels=3
    )
    last_obs = None
    relic_node_positions = []
    discovered_relic_nodes_ids = set()

    num_steps = len(replay["steps"])
    for step in range(num_steps):
        next_obs = json.loads(replay["steps"][step][team_id]["observation"]["obs"])

        if step > 0:
            curr_map.update_map(next_obs, last_obs)

        # Process relic nodes (same as before) ...
        observed_relic_node_positions = np.array(next_obs["relic_nodes"])
        observed_relic_nodes_mask = np.array(next_obs["relic_nodes_mask"])
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            if rid not in discovered_relic_nodes_ids:
                discovered_relic_nodes_ids.add(rid)
                relic_node_positions.append(observed_relic_node_positions[rid])
                curr_map.add_relic(observed_relic_node_positions[rid])
        # Process team units
        unit_mask = np.array(next_obs["units_mask"][team_id])
        available_unit_ids = np.where(unit_mask)[0]
        unit_positions = np.array(next_obs["units"]["position"][team_id])
        unit_energys = np.array(next_obs["units"]["energy"][team_id])
        # Get the full map state once per step.
        map_state = curr_map.map_stack()
        if isinstance(map_state, torch.Tensor):
            map_state_np = map_state.cpu().numpy()
        else:
            map_state_np = np.array(map_state)
        map_state_np = map_state_np.astype(np.float32)

        # For each active unit in this step, yield a sample.
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            # Find nearest relic node (or default if none)
            if len(relic_node_positions) > 0:
                nrp = min(relic_node_positions, key=lambda pos: abs(unit_pos[0]-pos[0]) + abs(unit_pos[1]-pos[1]))
            else:
                nrp = np.array([-1, -1])
            unit_state = get_state(unit_pos, nrp, unit_energy, env_cfg,team_id)
            unit_state = np.array(unit_state, dtype=np.float32)

            # Get the expert action for this unit.
            unit_action = int(replay["steps"][step][team_id]["action"][unit_id][0])
            # direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            if team_id==1:
                unit_action1=reverse_action_map[unit_action]
                unit_action=unit_action1
            # *** Yield the current step index along with the sample. ***
            yield step, map_state_np, unit_state, unit_action

        last_obs = next_obs

# ---------------------------------------------------------------
# Main Loop: Process All Replays and Save to HDF5
# ---------------------------------------------------------------
def process_all_replays(replay_folder, hdf5_filename, batch_size=1000):
    with h5py.File(hdf5_filename, 'w') as hf:
        # We will create four datasets:
        #   - "unique_map_states": stores each unique map state (shape: [N, channels, H, W])
        #   - "map_ids": for each sample, stores the integer index (into unique_map_states)
        #   - "unit_states": for each sample, the 9-dimensional state vector
        #   - "actions": for each sample, the expert action
        unique_map_ds = None
        map_ids_ds = None
        unit_states_ds = None
        actions_ds = None

        global_map_counter = 0  # Counts unique map states across all replays.
        sample_count = 0

        # Buffers for batch-writing.
        unique_map_buffer = []       # Buffer for unique map states.
        sample_map_ids_buffer = []   # Buffer for map id for each sample.
        unit_state_buffer = []       # Buffer for unit state.
        action_buffer = []           # Buffer for actions.

        # We will use these shapes when creating the datasets.
        map_shape = None  # e.g., (10, H, W)
        unit_shape = (10,)
        player_0_num=0
        player_1_num=0
        # Loop over replays.
        for file in os.listdir(replay_folder):
            if file.endswith(".json"):
                replay_path = os.path.join(replay_folder, file)
                replay = get_data(replay_path)
                team_ids = []
                if replay["info"]["TeamNames"][0] == "Frog Parade":
                    team_ids.append(0)
                    player_0_num+=1
                if replay["info"]["TeamNames"][1] == "Frog Parade":
                    team_ids.append(1)
                    player_1_num+=1
                # For each replay, keep a mapping from its step to a global map id.
                replay_map_to_global = {}
                for team_id in team_ids:
                    for team_id in team_ids:
                        for step, map_state, unit_state, unit_action in process_replay(replay_path, team_id=team_id):
                            # Convert map_state to a NumPy array.
                            if isinstance(map_state, torch.Tensor):
                                map_state_np = map_state.cpu().numpy()
                            else:
                                map_state_np = np.array(map_state)
                            map_state_np = map_state_np.astype(np.float32)
                            
                            # If map_shape is not yet set, do it here.
                            if map_shape is None:
                                map_shape = map_state_np.shape

                            key = (team_id, step)   # Use both team_id and step as the key!
                            if key not in replay_map_to_global:
                                replay_map_to_global[key] = global_map_counter
                                global_map_counter += 1
                                unique_map_buffer.append(map_state_np)
                            # Retrieve the global map id.
                            map_id = replay_map_to_global[key]
                            sample_map_ids_buffer.append(map_id)
                            unit_state_buffer.append(unit_state)
                            action_buffer.append(unit_action)
                            sample_count += 1

                            # Flush buffers if we have reached the batch size.
                            if sample_count % batch_size == 0:
                                # Create the datasets on the first flush.
                                if unique_map_ds is None:
                                    unique_map_ds = hf.create_dataset("unique_map_states",
                                                                    shape=(0,) + map_shape,
                                                                    maxshape=(None,) + map_shape,
                                                                    dtype='float32', chunks=True)
                                    unit_states_ds = hf.create_dataset("unit_states",
                                                                    shape=(0,) + unit_shape,
                                                                    maxshape=(None,) + unit_shape,
                                                                    dtype='float32', chunks=True)
                                    map_ids_ds = hf.create_dataset("map_ids",
                                                                shape=(0,),
                                                                maxshape=(None,),
                                                                dtype='int32', chunks=True)
                                    actions_ds = hf.create_dataset("actions",
                                                                shape=(0,),
                                                                maxshape=(None,),
                                                                dtype='int32', chunks=True)

                                # Flush the unique map states buffer (if nonempty).
                                if unique_map_buffer:
                                    cur_unique = unique_map_ds.shape[0]
                                    new_unique = cur_unique + len(unique_map_buffer)
                                    unique_map_ds.resize(new_unique, axis=0)
                                    unique_map_ds[cur_unique:new_unique, ...] = np.stack(unique_map_buffer, axis=0)
                                    unique_map_buffer = []

                                # Flush the samples buffers.
                                sample_map_ids_array = np.array(sample_map_ids_buffer, dtype=np.int32)
                                unit_state_array = np.stack(unit_state_buffer, axis=0)
                                action_array = np.array(action_buffer, dtype=np.int32)
                                cur_samples = map_ids_ds.shape[0]
                                new_samples = cur_samples + sample_map_ids_array.shape[0]
                                map_ids_ds.resize(new_samples, axis=0)
                                unit_states_ds.resize(new_samples, axis=0)
                                actions_ds.resize(new_samples, axis=0)
                                map_ids_ds[cur_samples:new_samples] = sample_map_ids_array
                                unit_states_ds[cur_samples:new_samples, ...] = unit_state_array
                                actions_ds[cur_samples:new_samples] = action_array

                                # Clear the sample buffers.
                                sample_map_ids_buffer = []
                                unit_state_buffer = []
                                action_buffer = []

                                print(f"Flushed {new_samples} samples so far.")

        # Final flush for any remaining buffers.
        if sample_map_ids_buffer or unique_map_buffer:
            if unique_map_ds is None:
                unique_map_ds = hf.create_dataset("unique_map_states",
                                                  shape=(0,) + map_shape,
                                                  maxshape=(None,) + map_shape,
                                                  dtype='float32', chunks=True)
                unit_states_ds = hf.create_dataset("unit_states",
                                                   shape=(0,) + unit_shape,
                                                   maxshape=(None,) + unit_shape,
                                                   dtype='float32', chunks=True)
                map_ids_ds = hf.create_dataset("map_ids",
                                               shape=(0,),
                                               maxshape=(None,),
                                               dtype='int32', chunks=True)
                actions_ds = hf.create_dataset("actions",
                                               shape=(0,),
                                               maxshape=(None,),
                                               dtype='int32', chunks=True)
            if unique_map_buffer:
                cur_unique = unique_map_ds.shape[0]
                new_unique = cur_unique + len(unique_map_buffer)
                unique_map_ds.resize(new_unique, axis=0)
                unique_map_ds[cur_unique:new_unique, ...] = np.stack(unique_map_buffer, axis=0)
            sample_map_ids_array = np.array(sample_map_ids_buffer, dtype=np.int32)
            unit_state_array = np.stack(unit_state_buffer, axis=0)
            action_array = np.array(action_buffer, dtype=np.int32)
            cur_samples = map_ids_ds.shape[0]
            new_samples = cur_samples + sample_map_ids_array.shape[0]
            map_ids_ds.resize(new_samples, axis=0)
            unit_states_ds.resize(new_samples, axis=0)
            actions_ds.resize(new_samples, axis=0)
            map_ids_ds[cur_samples:new_samples] = sample_map_ids_array
            unit_states_ds[cur_samples:new_samples, ...] = unit_state_array
            actions_ds[cur_samples:new_samples] = action_array
            print(f"Final flush: total {new_samples} samples stored.")

    print("All replays processed and training samples saved.")
    return player_0_num,player_1_num

# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------
if __name__ == '__main__':
    # Folder containing replay JSON files
    
    replay_folder = "./agents/behaviour_cloning_agent_v2/replays_debug"  
    # Name of the output HDF5 file
    hdf5_filename = "training_samples_debug.hdf5"
    start_time = time.time()
    player_0_num, player_1_num= process_all_replays(replay_folder, hdf5_filename, batch_size=1000)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    with h5py.File(hdf5_filename, "r") as hf:
        # Access the correct dataset name
        print("Number of samples in unique_map_states:", hf["unique_map_states"].shape[0])
        print("Number of samples in unit_states:", hf["unit_states"].shape[0])
        print("Number of samples in actions:", hf["actions"].shape[0])
    print(f"Games with player 0: {player_0_num}")
    print(f"Games with player 1: {player_1_num}")
