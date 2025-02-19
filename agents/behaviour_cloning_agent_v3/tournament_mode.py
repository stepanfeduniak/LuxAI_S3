import numpy as np
import torch
import os
import json
import time
import matplotlib.pyplot as plt
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from agent import Agent  # BC-based agent
from agent_0 import Agent_0  # Baseline agent
from agent_test_1 import Agent_test_1  # Baseline agent
from agent_x.agent import Agent_agd4b
from agent_resnet import Agent_resnet  # Baseline agent
import math
import random
import itertools
import warnings
warnings.filterwarnings("ignore")
# Function to compute expected score for agent A given ratings of A and B.
def expected_score(rating_a, rating_b):
    return 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400.0))

# Function to update a player's rating.
def update_rating(rating, expected, score, k=64):
    return rating + k * (score - expected)

# Dictionary to keep track of agent ratings.
# For self-play, these could be versions of your agent or different agents.
agent_ratings = {
    "agent_v1": 1500,
    "agent_v2": 1500,
    "agent_v3": 1500,
    "agent_v4": 1500,
    # Add more agents if needed.
}

# Function to process the outcome of a match between two agents.
def process_match(agent_a, agent_b,score_a,score_b, outcome, k=16):
    """
    Updates the Elo ratings after a match.
    
    Parameters:
        agent_a (str): Identifier for Agent A.
        agent_b (str): Identifier for Agent B.
        outcome (float): Outcome from Agent A's perspective: 
                         1 for win, 0.5 for draw, 0 for loss.
        k (int): The K-factor determining the sensitivity of rating updates.
    """
    diff=abs(score_a-score_b)
    rating_a = agent_ratings[agent_a]
    rating_b = agent_ratings[agent_b]
    
    exp_a = expected_score(rating_a, rating_b)
    exp_b = expected_score(rating_b, rating_a)  # This equals 1 - exp_a
    k=diff*k
    # Update ratings based on the outcome.
    new_rating_a = update_rating(rating_a, exp_a, outcome, k)
    new_rating_b = update_rating(rating_b, exp_b, 1 - outcome, k)
    
    agent_ratings[agent_a] = new_rating_a
    agent_ratings[agent_b] = new_rating_b

    print(f"After match: {agent_a} vs {agent_b}")
    print(f"    {agent_a}: {rating_a:.1f} -> {new_rating_a:.1f} (expected {exp_a:.3f})")
    print(f"    {agent_b}: {rating_b:.1f} -> {new_rating_b:.1f} (expected {exp_b:.3f})")


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

def play_tournament(agent_1_cls, agent_2_cls,agent_3_cls,agent_4_cls, replay=True, tournament_rounds=1000, replay_save_dir="./trained_replays"):
    os.makedirs(replay_save_dir, exist_ok=True)
    agents={"agent_v1":agent_1_cls,"agent_v2":agent_2_cls,"agent_v3":agent_3_cls,"agent_v4":agent_4_cls}
    # Initialize replay recording environment
    env = RecordEpisode(
        LuxAIS3GymEnv(numpy_output=True),
        save_on_close=True,
        save_on_reset=True,
        save_dir=replay_save_dir
    )
    
    for i in range(tournament_rounds):
        games_players = list(agents.keys())
        pairs = list(itertools.combinations(games_players, 2))
        print(f"Starting Round {i + 1}/{tournament_rounds}")
        random.shuffle(pairs)
        for pair in pairs:
            time_start = time.time()
        
            next_obs, info = env.reset()
            env_cfg = info["params"]
            # Create agents
            if pair:
                player_0 = agents[pair[0]]("player_0", env_cfg)
                player_1 = agents[pair[1]]("player_1", env_cfg)
            
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
            outcome=int(next_obs["player_0"]['team_wins'][0]>next_obs["player_0"]['team_wins'][1])
            print(f"Player 0:{next_obs["player_0"]['team_wins'][0]}, Player 1:{next_obs["player_0"]['team_wins'][1]}")
            time_total = time.time() - time_start
            # Suppose agent_v1 wins against agent_v2.
            process_match(pair[0],pair[1],next_obs["player_0"]['team_wins'][0],next_obs["player_0"]['team_wins'][1], outcome=outcome)  # agent_v1 win, agent_v2 loss

        # You can run multiple matches and track rating changes over time.
        print(f"\nUpdated Ratings:")
        for agent, rating in agent_ratings.items():
            print(f"{agent}: {rating:.1f}")
        print(f"Game {i + 1} finished in {time_total:.2f} seconds")
    
    env.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    play_tournament(Agent_agd4b,Agent_0,Agent_test_1 ,Agent_test_1, replay=True, tournament_rounds=20)