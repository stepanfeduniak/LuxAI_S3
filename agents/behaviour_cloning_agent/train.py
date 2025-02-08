# training.py
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from agent import BC_Model  # Import the BC_Model from agent.py

class HDF5Dataset(Dataset):
    """
    PyTorch Dataset to load training samples from an HDF5 file.
    The HDF5 file is assumed to have three datasets:
      - "map_states": float32 array of shape (N, channels, H, W)
      - "unit_states": float32 array of shape (N, 9)
      - "actions": int32 array of shape (N,)
    """
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.hf = h5py.File(hdf5_file, 'r')
        self.map_states = self.hf['map_states']
        self.unit_states = self.hf['unit_states']
        self.actions = self.hf['actions']
        
    def __len__(self):
        return self.map_states.shape[0]
    
    def __getitem__(self, idx):
        # Get a single sample
        map_state = self.map_states[idx]           # e.g., shape (10, H, W)
        unit_state = self.unit_states[idx]           # shape (9,)
        action = self.actions[idx]                   # integer target
        # Convert to torch tensors
        map_state = torch.tensor(map_state, dtype=torch.float32)
        unit_state = torch.tensor(unit_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        return unit_state, map_state, action

    def close(self):
        self.hf.close()

def train_behavior_cloning(hdf5_filename, num_epochs=10, batch_size=64, learning_rate=1e-4, device=None):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HDF5Dataset(hdf5_filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create the behavior cloning model.
    # (The map has 10 channels, the unit state is 9-dim, and there are 6 actions.)
    model = BC_Model(map_channels_input=10, unit_feature_dim=9, action_dim=6).to(device)
    
    # Use CrossEntropyLoss (which expects raw logits)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (unit_states, map_states, actions) in enumerate(dataloader):
            unit_states = unit_states.to(device)   # [batch, 9]
            map_states = map_states.to(device)       # [batch, 10, H, W]
            actions = actions.to(device)             # [batch]
            
            optimizer.zero_grad()
            logits = model(unit_states, map_states)  # [batch, 6]
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Save the trained model weights.
    torch.save(model.state_dict(), "bc_model.pth")
    dataset.close()

if __name__ == '__main__':
    hdf5_filename = "training_samples.hdf5"  # Path to your HDF5 file with samples
    train_behavior_cloning(hdf5_filename, num_epochs=10, batch_size=64)
