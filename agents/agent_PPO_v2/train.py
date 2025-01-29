import numpy as np
import torch
import time
import os
import glob
import re
import pandas as pd
from luxai_s3.wrappers import LuxAIS3GymEnv
from luxai_s3.params import EnvParams
from agent import Agent
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import traceback
from agent_0 import Agent_0

def evaluate_agents(agent_1_cls, agent_2_cls, replay=False, games_to_play=3,replay_save_dir="./agents/agentA2C/replays"):
    training=True

    if not replay:
        env = LuxAIS3GymEnv(numpy_output=True)
    else:
        env = RecordEpisode(
            LuxAIS3GymEnv(numpy_output=True), save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
        )
    game_rewards_history = {"player_0":[],"player_1":[]}

    obs, info = env.reset()
    env_cfg = info["params"]  

    player_0 = agent_1_cls("player_0", info["params"], training=training)
    player_1 = agent_2_cls("player_1", info["params"], training=training)
    n_agents=16

evaluate_agents(Agent, Agent_0, replay=True, games_to_play=10) 

