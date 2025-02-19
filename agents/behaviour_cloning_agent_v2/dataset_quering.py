import os
import h5py
import numpy as np
import torch
import json  # Save logs as JSON
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(hdf5_file, 'r')
        # Load the unique map states and the sample arrays.
        self.unique_map_states = self.hf['unique_map_states']
        self.map_ids = self.hf['map_ids']
        self.unit_states = self.hf['unit_states']
        self.actions = self.hf['actions']

    def __len__(self):
        return self.map_ids.shape[0]

    def __getitem__(self, idx):
        # Get the map id for this sample.
        map_id = self.map_ids[idx]
        # Look up the corresponding unique map state.
        map_state = self.unique_map_states[map_id]  # e.g., shape (10, H, W)
        unit_state = self.unit_states[idx]           # shape (9,)
        action = self.actions[idx]                   # scalar

        map_state = torch.tensor(map_state, dtype=torch.float32)
        unit_state = torch.tensor(unit_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        return unit_state, map_state, action

if __name__ == '__main__':
    hdf5_filename = "training_samples_debug.hdf5"
    dataset = HDF5Dataset(hdf5_filename)
    actions=[]
    state, play_map, action = dataset.__getitem__(6000)
    actions.append(action)
    # Print the unit state (optional)
    print("Unit State:", state)
    print("Actions",actions)
    # Create a 2x5 grid for heatmaps.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()  # Make it easier to iterate

    for i in range(10):
        # Convert the i-th map (which is a torch tensor) to a NumPy array.
        heatmap = play_map[i, :, :].numpy()
        # Display the heatmap. You can change the colormap ('cmap') as desired.
        im = axes[i].imshow(heatmap, cmap='viridis')
        axes[i].set_title(f"Row {i}")
        axes[i].axis('off')  # Optional: Hide axis ticks
        
        # Optionally add a colorbar to each subplot
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
