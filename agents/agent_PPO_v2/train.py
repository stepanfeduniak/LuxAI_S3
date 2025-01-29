import numpy as np
import torch
import time
import os
import glob
import re
import pandas as pd
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from luxai_s3.params import EnvParams
from agent import Agent  # <-- your PPO-based Agent
from agent_0 import Agent_0  # some baseline agent

def evaluate_agents(agent_1_cls, agent_2_cls, replay=False, games_to_play=3, replay_save_dir="./agents/agent_PPO_v2/replays"):
    if not replay:
        env = LuxAIS3GymEnv(numpy_output=True)
    else:
        env = RecordEpisode(
            LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
        )

    # We'll track points to compute step-based reward for each agent
    old_points_0 = 0
    old_points_1 = 0

    for i in range(games_to_play):
        print(f"Game {i}")

        obs, info = env.reset()
        env_cfg = info["params"]
        
        # Create the agents
        player_0 = agent_1_cls("player_0", env_cfg)  # PPO agent
        player_1 = agent_2_cls("player_1", env_cfg)  # baseline or some other agent

        # Initialize old points for reward calculation
        # (We assume obs["player_0"]["team_points"][player_0.team_id] is valid)
        old_points_0 = obs["player_0"]["team_points"][player_0.team_id]
        old_points_1 = obs["player_1"]["team_points"][player_1.team_id]

        step = 0
        game_done = False

        while not game_done:
            actions = {}
            for agent in [player_0, player_1]:
                # Each agent decides its actions
                actions[agent.player] = agent.act(step=step, obs=obs[agent.player])
            
            # Take one environment step
            next_obs, wins, terminated, truncated, info = env.step(actions)
            
            # Build 'done' flags for each agent
            done_0 = bool(terminated["player_0"].item()) or bool(truncated["player_0"].item())
            done_1 = bool(terminated["player_1"].item()) or bool(truncated["player_1"].item())
            
            # Compute "new" points for reward
            new_points_0 = next_obs["player_0"]["team_points"][player_0.team_id]
            new_points_1 = next_obs["player_1"]["team_points"][player_1.team_id]
            
            # Reward is the difference in points
            r0 = new_points_0 - old_points_0
            r1 = new_points_1 - old_points_1
            
            # Store them in the agent's memory
            # For the PPO agent:
            player_0.calculate_rewards_and_dones(next_obs["player_0"],r0, done_0)
            
            old_points_0 = new_points_0
            old_points_1 = new_points_1
            
            # Next step
            obs = next_obs
            step += 1

            # If either agent is done => game is done
            if done_0 or done_1:
                game_done = True
        player_0.save_model()
    env.close()

# Actually run it
if __name__ == "__main__":
    evaluate_agents(Agent, Agent_0, replay=True, games_to_play=1000)
