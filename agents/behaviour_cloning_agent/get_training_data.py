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

def get_state(unit_pos, nearest_relic_node_position, unit_energy, env_cfg):
    """
    Returns a normalized 9-dimensional state vector for a unit.
    
    The state vector consists of:
      - Normalized unit position (x, y)
      - Normalized nearest relic node position (x, y)
      - Normalized unit energy
      - Additional normalized environmental parameters:
          unit_sensor_range, unit_move_cost, unit_sap_cost, unit_sap_range
    """
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
def process_replay(replay_path,team_id):
    """
    Processes a single replay file and yields training samples.
    
    For each timestep in the replay and for each active unit, a sample is produced:
      - map_state: full map representation (as a numpy array of float32)
      - unit_state: 9-dimensional state vector (as a numpy array of float32)
      - action: integer action (0 to 5)
    
    Args:
        replay_path (str): Path to the replay JSON file.
        
    Yields:
        tuple: (map_state, unit_state, action)
    """
    replay = get_data(replay_path)
    if replay is None:
        return

    env_cfg = replay["configuration"]["env_cfg"]
    team_id = team_id  # Process team 0's data

    # Initialize the playing map.
    # (Here the map is built with 2 channels for units, 5 for general map info, and 3 for relics)
    curr_map = map_processing.Playing_Map(
        player_id=0, 
        map_size=env_cfg["map_width"],
        unit_channels=2,
        map_channels=5,
        relic_channels=3
    )
    last_obs = None

    # For tracking relic nodes
    relic_node_positions = []         # List of discovered relic positions
    discovered_relic_nodes_ids = set()  # Set of relic IDs already added

    num_steps = len(replay["steps"])
    for step in range(num_steps):
        # Parse the observation for the current step for team 0
        next_obs = json.loads(replay["steps"][step][team_id]["observation"]["obs"])

        # Update the map state if not the first timestep
        if step > 0:
            curr_map.update_map(next_obs, last_obs)

        # Process relic node data: add new relic nodes to the map
        observed_relic_node_positions = np.array(next_obs["relic_nodes"])  # shape: (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(next_obs["relic_nodes_mask"])   # shape: (max_relic_nodes,)
        visible_relic_node_ids = np.where(observed_relic_nodes_mask)[0]
        for rid in visible_relic_node_ids:
            if rid not in discovered_relic_nodes_ids:
                discovered_relic_nodes_ids.add(rid)
                relic_node_positions.append(observed_relic_node_positions[rid])
                curr_map.add_relic(observed_relic_node_positions[rid])

        # Process unit data for team 0
        unit_mask = np.array(next_obs["units_mask"][team_id])
        available_unit_ids = np.where(unit_mask)[0]
        unit_positions = np.array(next_obs["units"]["position"][team_id])  # shape: (max_units, 2)
        unit_energys = np.array(next_obs["units"]["energy"][team_id])        # shape: (max_units, 1)

        # Get the current full map state once for the timestep.
        map_state = curr_map.map_stack()
        # If the map state is a torch tensor, convert it to a numpy array.
        if isinstance(map_state, torch.Tensor):
            map_state_np = map_state.cpu().numpy()
        else:
            map_state_np = np.array(map_state)
        map_state_np = map_state_np.astype(np.float32)

        # For each active unit, build the unit state and extract the expert action.
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            # Determine nearest relic node position (or use default if none found)
            if len(relic_node_positions) > 0:
                nrp = min(relic_node_positions, key=lambda pos: manhattan_distance(unit_pos, pos))
            else:
                nrp = np.array([13, 13])
            unit_state = get_state(unit_pos, nrp, unit_energy, env_cfg)
            unit_state = np.array(unit_state, dtype=np.float32)

            # Extract the expert action for the unit at this timestep.
            # [step][0] accesses team 0's action; then index by unit_id and take the first element.
            unit_action = replay["steps"][step][0]["action"][unit_id][0]
            unit_action = int(unit_action)

            # Yield the sample tuple
            yield map_state_np, unit_state, unit_action

        last_obs = next_obs

# ---------------------------------------------------------------
# Main Loop: Process All Replays and Save to HDF5
# ---------------------------------------------------------------
def process_all_replays(replay_folder, hdf5_filename, batch_size=1000):
    """
    Processes all replay files in the given folder and saves training samples to an HDF5 file.
    
    Each sample is a tuple (map_state, unit_state, action):
      - map_state: a float32 array (e.g. shape (10, H, W))
      - unit_state: a float32 array of shape (9,)
      - action: an integer in [0, 5]
      
    The samples are appended in batches to avoid holding all data in memory.
    
    Args:
        replay_folder (str): Folder containing replay JSON files.
        hdf5_filename (str): Output HDF5 file name.
        batch_size (int): Number of samples to accumulate before writing to disk.
    """
    # Open the HDF5 file for writing.
    with h5py.File(hdf5_filename, 'w') as hf:
        map_state_ds = None
        unit_state_ds = None
        action_ds = None
        sample_count = 0

        # Buffers to accumulate samples
        map_state_buffer = []
        unit_state_buffer = []
        action_buffer = []

        # Loop over all JSON files in the replay folder.
        for file in os.listdir(replay_folder):
            if file.endswith(".json"):
                replay_path = os.path.join(replay_folder, file)
                print(f"Processing replay: {replay_path}")
                for team_id in [0,1]:
                    for map_state, unit_state, action in process_replay(replay_path,team_id=team_id):
                        map_state_buffer.append(map_state)
                        unit_state_buffer.append(unit_state)
                        action_buffer.append(action)
                        sample_count += 1

                        # If we have reached the batch size, flush the buffers to the HDF5 file.
                        if sample_count % batch_size == 0:
                            # On the first flush, create the datasets using the shape of the first sample.
                            if map_state_ds is None:
                                map_shape = map_state.shape          # e.g., (10, H, W)
                                unit_shape = unit_state.shape          # should be (9,)
                                map_state_ds = hf.create_dataset("map_states",
                                                                shape=(0,) + map_shape,
                                                                maxshape=(None,) + map_shape,
                                                                dtype='float32', chunks=True)
                                unit_state_ds = hf.create_dataset("unit_states",
                                                                shape=(0,) + unit_shape,
                                                                maxshape=(None,) + unit_shape,
                                                                dtype='float32', chunks=True)
                                action_ds = hf.create_dataset("actions",
                                                            shape=(0,),
                                                            maxshape=(None,),
                                                            dtype='int32', chunks=True)

                            # Convert buffers to numpy arrays.
                            map_state_array = np.stack(map_state_buffer, axis=0)
                            unit_state_array = np.stack(unit_state_buffer, axis=0)
                            action_array = np.array(action_buffer, dtype=np.int32)

                            # Get the current dataset size and resize to accommodate new samples.
                            current_size = map_state_ds.shape[0]
                            new_size = current_size + map_state_array.shape[0]
                            map_state_ds.resize(new_size, axis=0)
                            unit_state_ds.resize(new_size, axis=0)
                            action_ds.resize(new_size, axis=0)

                            # Write the new data.
                            map_state_ds[current_size:new_size, ...] = map_state_array
                            unit_state_ds[current_size:new_size, ...] = unit_state_array
                            action_ds[current_size:new_size, ...] = action_array

                            # Clear the buffers.
                            map_state_buffer = []
                            unit_state_buffer = []
                            action_buffer = []
                            print(f"Flushed {new_size} samples so far.")

        # Flush any remaining samples in the buffers.
        if len(map_state_buffer) > 0:
            if map_state_ds is None:
                sample = map_state_buffer[0]
                map_shape = sample.shape
                unit_shape = unit_state_buffer[0].shape
                map_state_ds = hf.create_dataset("map_states",
                                                 shape=(0,) + map_shape,
                                                 maxshape=(None,) + map_shape,
                                                 dtype='float32', chunks=True)
                unit_state_ds = hf.create_dataset("unit_states",
                                                  shape=(0,) + unit_shape,
                                                  maxshape=(None,) + unit_shape,
                                                  dtype='float32', chunks=True)
                action_ds = hf.create_dataset("actions",
                                              shape=(0,),
                                              maxshape=(None,),
                                              dtype='int32', chunks=True)
            map_state_array = np.stack(map_state_buffer, axis=0)
            unit_state_array = np.stack(unit_state_buffer, axis=0)
            action_array = np.array(action_buffer, dtype=np.int32)
            current_size = map_state_ds.shape[0]
            new_size = current_size + map_state_array.shape[0]
            map_state_ds.resize(new_size, axis=0)
            unit_state_ds.resize(new_size, axis=0)
            action_ds.resize(new_size, axis=0)
            map_state_ds[current_size:new_size, ...] = map_state_array
            unit_state_ds[current_size:new_size, ...] = unit_state_array
            action_ds[current_size:new_size, ...] = action_array
            print(f"Final flush: total {new_size} samples stored.")

    print("All replays processed and training samples saved.")

# ---------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------
if __name__ == '__main__':
    # Folder containing replay JSON files
    replay_folder = "./agents/behaviour_cloning_agent/replays"  
    # Name of the output HDF5 file
    hdf5_filename = "training_samples.hdf5"
    start_time = time.time()
    process_all_replays(replay_folder, hdf5_filename, batch_size=1000)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    with h5py.File("training_samples.hdf5", "r") as hf:
        # Print the number of samples in each dataset
        print("Number of samples in map_states:", hf["map_states"].shape[0])
        print("Number of samples in unit_states:", hf["unit_states"].shape[0])
        print("Number of samples in actions:", hf["actions"].shape[0])
