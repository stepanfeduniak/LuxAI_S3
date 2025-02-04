import numpy as np
import torch
import time
import os
import json  # Save logs as JSON
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from luxai_s3.params import EnvParams
from agent import Agent  # PPO-based agent
from agent_0 import Agent_0  # Baseline agent
def graph_data(gamerewards, ema_rewards):

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(gamerewards)), gamerewards, label="Game Reward")
    plt.plot(range(len(ema_rewards)), ema_rewards, label="EMA Reward")
    plt.xlabel("Game #")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards Over Games")
    plt.legend()
    plt.grid()
    plt.show()
# Directory where logs are stored
CHECKPOINT_DIR = "./training_logs"
beta=0.95
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

def evaluate_agents(agent_1_cls,
                    agent_2_cls,
                    replay=False,
                    games_to_play=1000,
                    replay_save_dir="./replays",
                    checkpoint_interval=10):
    run_folder = os.path.join(CHECKPOINT_DIR)
    os.makedirs(run_folder, exist_ok=True)

    # File paths for logs
    reward_log_path = os.path.join(run_folder, "reward_log.json")
    ppo_log_path = os.path.join(run_folder, "ppo_log.json")

    # Default log structure if no file exists
    default_reward_logs = {
        "gamerewards": [],
        "ema_rewards": []
    }

    # Try to load existing logs; if they exist, continue from last
    loaded_reward_logs = load_checkpoint(reward_log_path)
    if loaded_reward_logs is None:
        gamerewards = []
        ema_rewards = []
    else:
        gamerewards = loaded_reward_logs.get("gamerewards", [])
        ema_rewards = loaded_reward_logs.get("ema_rewards", [])

    # Choose the env depending on whether we want replays
    if not replay:
        env = LuxAIS3GymEnv(numpy_output=True)
    else:
        env = RecordEpisode(
            LuxAIS3GymEnv(numpy_output=True),
            save_on_close=True,
            save_on_reset=True,
            save_dir=replay_save_dir
        )
    ema_reward = ema_rewards[-1] if len(ema_rewards) > 0 else 0
    # Resume from where we left off if logs exist
    start_game_idx = len(gamerewards)
    graph_data(gamerewards,ema_rewards)
    all_fleet_memories=[]
    for i in range(start_game_idx, games_to_play):
        if i%2==0:
            player_PPO="player_0"
            player_baseline="player_1"
        else:
            player_PPO="player_1"
            player_baseline="player_0"
        time_start = time.time()
        
        #game setup
        next_obs, info = env.reset()
        zero_obs= next_obs
        env_cfg = info["params"]

        last_obs=next_obs
        # Create the agents
        player_0 = agent_1_cls(player_PPO, env_cfg)  # PPO agent
        player_1 = agent_2_cls(player_baseline, env_cfg)  # baseline agent
        
        old_points_0 = next_obs[player_PPO]["team_points"][player_0.team_id]
        old_points_1 = next_obs[player_baseline]["team_points"][player_1.team_id]
        game_rew = 0
        step = 0
        game_done = False
        game_breakdown= {
            "movement": 0.0,
            "energy": 0.0,
            "proximity": 0.0,
            "exploration": 0.0,
            "combat": 0.0,
            "relic": 0.0
        }
        while not game_done:
            actions = {
                agent.player: agent.act(step=step, obs=next_obs[agent.player])
                for agent in [player_0, player_1]
            }
            player_0_prev_available_units=np.where(last_obs[player_PPO]["units_mask"][player_0.team_id])[0]
            player_1_prev_available_units=np.where(last_obs[player_baseline]["units_mask"][player_1.team_id])[0]
            next_obs, wins, terminated, truncated, info = env.step(actions)
            player_0.play_map.update_map(next_obs[player_PPO],last_obs[player_PPO])
            player_1.play_map.update_map(next_obs[player_baseline],last_obs[player_baseline])
            done_0 = bool(terminated[player_PPO].item()) or bool(truncated[player_PPO].item())
            done_1 = bool(terminated[player_baseline].item()) or bool(truncated[player_baseline].item())

            new_points_0 = next_obs[player_PPO]["team_points"][player_0.team_id]
            new_points_1 = next_obs[player_baseline]["team_points"][player_1.team_id]

            r0 = new_points_0 - old_points_0
            r1 = new_points_1 - old_points_1

            # Agent calculates its own internal shaping reward
            new_reward, total_breakdown= player_0.calculate_rewards_and_dones(next_obs[player_PPO],last_obs[player_PPO],env_cfg, r0, done_0,player_0_prev_available_units)
            _,_ = player_1.calculate_rewards_and_dones(next_obs[player_baseline],last_obs[player_baseline],env_cfg, r1, done_1,player_1_prev_available_units)
            game_rew += new_reward
            for key in total_breakdown:
                game_breakdown[key] += total_breakdown[key]
                
            old_points_0, old_points_1 = new_points_0, new_points_1
            last_obs = next_obs
            if next_obs[player_PPO]["match_steps"]==100:
                all_fleet_memories.append(player_0.return_memories())
                player_0.clear_memories()
                
            if next_obs[player_baseline]["match_steps"]==100:
                all_fleet_memories.append(player_1.return_memories())
                player_1.clear_memories()
            step += 1
            game_done = done_0 or done_1
        #player_0.update_ppo()
        ema_reward = beta * ema_reward + (1 - beta) * game_rew
        for key in game_breakdown:
            print(f"{key}:{game_breakdown[key]}")
        print(f"Relic nodes discovered:{player_0.relic_node_positions}")
        print(f"{env_cfg['unit_sensor_range']}")
        if i%2 ==1 and len(all_fleet_memories)>0:
            player_0.update_ppo(all_fleet_memories)
            all_fleet_memories=[]
            player_0.save_model()
            

        

        # Log the final reward for this game
        gamerewards.append(game_rew)
        ema_rewards.append(ema_reward)
        # Save logs every 'checkpoint_interval' games
        if i % checkpoint_interval == 0 or i == games_to_play - 1:
            # Save the updated reward logs
            reward_log_data = {
                "gamerewards": gamerewards,
                "ema_rewards": ema_rewards
            }
            save_checkpoint(reward_log_path, reward_log_data)

        # Visualization at key points (optional)
        if i in [1, 10, games_to_play - 1]:

            plt.figure(figsize=(10, 4))

            # 1) Rewards over games
            plt.subplot(1, 1, 1)
            plt.plot(range(len(gamerewards)), gamerewards,ema_rewards, label="Game Reward")
            plt.xlabel("Game #")
            plt.ylabel("Cumulative Reward")
            plt.title("Rewards Over Games")
            plt.legend()
            plt.grid()
            plt.show()

        time_finish = time.time()
        time_total=time_finish - time_start
        print(f"Game {i} took {time_total:.2f} seconds")
        part_times=player_0.get_track_times()
        #print(f"part_times: {part_times}")
        for key in part_times.keys():
            print(f"{key} took {part_times[key]/time_total*100:.2f}% of total time")
        #print(f"Game {i} reward: {game_rew}")
        #print(f"Unit visibility range:{ env_cfg["unit_sensor_range"]}")
        

    env.close()

# Run the evaluation
if __name__ == "__main__":
    evaluate_agents(Agent, Agent, replay=True, games_to_play=10000, checkpoint_interval=1)
