import numpy as np
import torch
import os
import json
import time
import matplotlib.pyplot as plt
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from agent_test_1 import Agent_test_1  # Baseline agent
from agent_0 import Agent_0  # Baseline agent

def save_checkpoint(file_path, data):
    """Saves training logs to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_checkpoint(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def evaluate_agents(agent_1_cls, agent_2_cls, replay=True, games_to_play=1000, replay_save_dir="./trained_replays"):
    os.makedirs(replay_save_dir, exist_ok=True)
    
    # Initialize replay recording environment
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir
    )
    
    for i in range(games_to_play):
        print(f"Starting game {i + 1}/{games_to_play}")
        time_start = time.time()
        
        next_obs, info = env.reset()
        env_cfg = info["params"]
        
        # Create agents
        player_0 = agent_1_cls("player_0", env_cfg)
        player_1 = agent_2_cls("player_1", env_cfg)
        
        step = 0
        game_done = False
        
        while not game_done:
            actions = {
                agent.player: agent.act(step=step, obs=next_obs[agent.player])
                for agent in [player_0, player_1]
            }
            
            next_obs, _, terminated, truncated, _ = env.step(actions)
            game_done = terminated["player_0"].item() or terminated["player_1"].item() or \
                        truncated["player_0"].item() or truncated["player_1"].item()
            
            step += 1
        play_map=player_0.play_map.map_stack()
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
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
        time_total = time.time() - time_start
        print(f"Game {i + 1} finished in {time_total:.2f} seconds")
    
    env.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    evaluate_agents(Agent_test_1, Agent_0, replay=True, games_to_play=2)