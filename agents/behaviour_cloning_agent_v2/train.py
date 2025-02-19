import os
import h5py
import numpy as np
import torch
import json  # Save logs as JSON
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from agent import BC_Model  # Import the BC_Model from agent.py
import time
beta1 = 0.95
beta2= 0.99
CHECKPOINT_DIR = "./training_logs"

def save_checkpoint(file_path, data):
    """Saves training logs to JSON file."""
    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_checkpoint(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def plot_loss(losses, ema_history1,ema_history2):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(losses)), losses, label="Losses")
    plt.plot(range(len(ema_history1)), ema_history1, label="EMA Reward 1")
    plt.plot(range(len(ema_history2)), ema_history2, label="EMA Reward 2")
    plt.xlabel("Game #")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards Over Games")
    plt.legend()
    plt.grid()
    plt.show()

from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

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

    def close(self):
        self.hf.close()


def train_behavior_cloning(
    hdf5_filename, 
    num_epochs=10, 
    batch_size=64, 
    learning_rate=1e-4, 
    device=None, 
    path=None,
    checkpoint_interval=30
):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HDF5Dataset(hdf5_filename)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Make sure the directory exists before using it
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_folder = CHECKPOINT_DIR

    reward_log_path = os.path.join(run_folder, "reward_log.json")
    loaded_reward_logs = load_checkpoint(reward_log_path)

    model = BC_Model(map_channels_input=10, unit_feature_dim=10, action_dim=6).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except:
        print("No model found, starting from scratch.")

    if loaded_reward_logs is None:
        loss_history = []
        ema_history1 = []
        ema_history2 = []
    else:
        loss_history = loaded_reward_logs.get("loss_history", [])
        ema_history1 = loaded_reward_logs.get("ema_history1", [])
        ema_history2 = loaded_reward_logs.get("ema_history2", [])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if len(ema_history1)==0: 
        ema_loss1 = 1.4
        ema_loss2 = 1.4
    else:
        ema_loss1= ema_history1[-1]
        ema_loss2= ema_history1[-1]
    # Optionally plot if you want to see the existing loss
    plot_loss(loss_history, ema_history1,ema_history2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time=time.time()
        for i, (unit_states, map_states, actions) in enumerate(dataloader):
            unit_states, map_states, actions = unit_states.to(device), map_states.to(device), actions.to(device)
            optimizer.zero_grad()

            logits = model(unit_states, map_states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update EMA
            ema_loss1 = beta1 * ema_loss1 + (1 - beta1) * loss.item()
            ema_loss2 = beta2 * ema_loss2 + (1 - beta2) * loss.item()
            if i % 10 == 0:
                ema_history1.append(ema_loss1)
                ema_history2.append(ema_loss2)
                loss_history.append(loss.item())

            if (i + 1) % 50 == 0:
                finish_time=time.time()
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss(beta=.99): {ema_loss2:.4f}, Time per 50 epochs:{finish_time-start_time}")
                start_time=time.time()

            if i % 20 == 0:
                torch.save(model.state_dict(), "bc_model.pth")

            if i % checkpoint_interval == 0 or i == (len(dataloader) - 1):
                # Save the updated reward logs
                reward_log_data = {
                    "loss_history": loss_history,
                    "ema_history1": ema_history1,
                    "ema_history2": ema_history2
                }
                save_checkpoint(reward_log_path, reward_log_data)

        avg_loss = running_loss / len(dataloader)
        torch.save(model.state_dict(), "bc_model.pth")
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    dataset.close()

if __name__ == '__main__':
    hdf5_filename = "training_samples.hdf5"
    train_behavior_cloning(
        hdf5_filename, 
        num_epochs=10, 
        batch_size=64, 
        path="bc_model.pth"
    )
