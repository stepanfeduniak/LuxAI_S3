from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from agent import Agent, Playing_Map
import torch
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

def calculate_reward(player_0,player_1,actions,obs,prev_score_player_0,prev_score_player_1,last_obs,vis_priority=False,training=True):
    if training==True:
        current_score_p0 = obs["player_0"]["team_points"][player_0.team_id]
        current_score_p1 = obs["player_1"]["team_points"][player_1.team_id]
        # Reward Parameters
        RELIC_FACTOR = 2.0          # Relic points are the primary objective
        MOVEMENT_SPAM_PENALTY = 0.08 
        MOVE_CENTER_PENALTY=0.02     # Penalize center spam
        if vis_priority==False:
            VISIBILITY_FACTOR = 0.2  # Encourage exploring new tiles
            
        else:
            VISIBILITY_FACTOR = 2
            MOVEMENT_SPAM_PENALTY = 0.03 
            
        ENERGY_FACTOR = 0#.0005       # Reward energy preservation
        SUCCESSFUL_SAP_BONUS = 0.5    # Bonus for successful sapping
        WALL_MOVE_PENALTY = 0.4
        CLOSE_TO_RELIC=0.5
        # Reward calculation for player 0
        score_diff_p0 = current_score_p0 - prev_score_player_0  # Difference in relic points
        score_diff_p1 = current_score_p1 - prev_score_player_1

        # Movement spam penalty
        same_movement_penalty_0 = np.max(np.bincount(actions["player_0"][:, 0], minlength=6))
        same_movement_penalty_1 = np.max(np.bincount(actions["player_1"][:, 0], minlength=6))
        center_mov_0=np.bincount(actions["player_0"][:, 0])[0]
        center_mov_1=np.bincount(actions["player_1"][:, 0])[0]
        # Visibility reward: Encourage discovering new tiles
        # Optionally reward for new visible tiles instead of total visible tiles
        new_visible_tiles_0 = np.sum(np.bitwise_xor(obs["player_0"]["sensor_mask"], last_obs["player_0"]["sensor_mask"]))
        new_visible_tiles_1 = np.sum(np.bitwise_xor(obs["player_1"]["sensor_mask"], last_obs["player_1"]["sensor_mask"]))

        # Energy change reward: Reward for energy preservation
        energy_diff_p0 = (np.sum(obs["player_0"]["units"]["energy"]) 
                        - np.sum(last_obs["player_0"]["units"]["energy"]))
        energy_diff_p1 = (np.sum(obs["player_1"]["units"]["energy"]) 
                        - np.sum(last_obs["player_1"]["units"]["energy"]))

        # Successful sap bonus: Reward if any enemy unit loses energy due to sap
        sap_success_0 = np.sum(last_obs["player_1"]["units"]["energy"] 
                                    - obs["player_1"]["units"]["energy"])
        sap_success_1 = np.sum(last_obs["player_0"]["units"]["energy"] 
                                    - obs["player_0"]["units"]["energy"])

        # Final rewards
        reward_p0 = (RELIC_FACTOR * score_diff_p0) \
                            - (MOVEMENT_SPAM_PENALTY * same_movement_penalty_0) \
                            + (VISIBILITY_FACTOR * new_visible_tiles_0) \
                            + (ENERGY_FACTOR * energy_diff_p0) \
                            -(MOVE_CENTER_PENALTY*center_mov_0)
                            #+ (SUCCESSFUL_SAP_BONUS * sap_success_0)

        reward_p1 = (RELIC_FACTOR * score_diff_p1) \
                            - (MOVEMENT_SPAM_PENALTY * same_movement_penalty_1) \
                            + (VISIBILITY_FACTOR * new_visible_tiles_1) \
                            + (ENERGY_FACTOR * energy_diff_p1) \
                            -(MOVE_CENTER_PENALTY*center_mov_1)
                            #+ (SUCCESSFUL_SAP_BONUS * sap_success_1)
          # Tune this as you like

        # Track how many wall-moves each player does this step
        wall_moves_p0 = 0
        wall_moves_p1 = 0

                # Because each player's code uses the same move definitions:
                # 0=center, 1=up, 2=right, 3=down, 4=left, 5=sap
                # We'll check if a "move" is out of bounds for each unit.
                # (Map is 24x24, so valid x,y in [0..23]).

        # For each agent, identify which units are present and check their positions + actions
        for agent in [player_0, player_1]:
            # Which player's ID is this?
            if agent == player_0:
                team_id = agent.team_id     # 0 or 1
                wall_moves = 0             # count how many times we try to move off the map
            else:
                team_id = agent.team_id
                wall_moves = 0
                    
            # Extract the set of units that exist for this team
            unit_mask = obs[agent.player]["units_mask"][team_id]
            unit_positions = obs[agent.player]["units"]["position"][team_id]

            # Loop over units that are alive
            alive_unit_ids = np.where(unit_mask)[0]
            for uid in alive_unit_ids:
                # The chosen action is in actions[agent.player][uid][0]
                action_type = actions[agent.player][uid][0]
                
                # If action_type is 1..4, we check if it would go off the map
                x, y = unit_positions[uid]  # current position
                if action_type == 1 and y == 0:         # up but at top edge
                    wall_moves += 1
                elif action_type == 2 and x == 23:      # right but at right edge
                            wall_moves += 1
                elif action_type == 3 and y == 23:      # down but at bottom edge
                            wall_moves += 1
                elif action_type == 4 and x == 0:       # left but at left edge
                            wall_moves += 1
                    
            # Now apply the penalty for that agent
            if agent == player_0:
                reward_p0 -= wall_moves * WALL_MOVE_PENALTY
            else:
                reward_p1 -= wall_moves * WALL_MOVE_PENALTY 
    return reward_p0 , reward_p1
def cumulative_reward(obs,prev_obs,player):
    NEW_POINTS_FACTOR=1
    NEW_VISIBILITY_FACTOR=0.0
    VISIBILITY_FACTOR=0.003
    #if step%100==0:
    
    new_points=obs["team_points"][player.team_id]-prev_obs["team_points"][player.team_id]
    #else:
    #   return 0
    new_visibility = np.sum(obs["sensor_mask"]) - np.sum(prev_obs["sensor_mask"])
    visibility=np.sum(obs["sensor_mask"])
    
    if new_points<-1:
         return np.array([0])
    reward=new_points*NEW_POINTS_FACTOR+new_visibility*NEW_VISIBILITY_FACTOR+visibility*VISIBILITY_FACTOR
    return np.array([reward],dtype=np.float32)
    
    

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



    for game in range(games_to_play):
        obs, info = env.reset()  
        game_done = False
        step = 0
        gamerewards={}
        for player in [player_0.player,player_1.player]:
            gamerewards[player] = np.zeros(n_agents, dtype=np.float32)
        print(f"{game}")
        start_time = time.time()
        player_0.map=Playing_Map(0,24)
        player_1.map=Playing_Map(1,24)
        while not game_done:
            
            previous_obs=obs
            actions = {}
            log_probs = {}
            values ={}
            rewards={}
            # Convert list of actions to e.g. np array for the environment
            for agent in [player_0, player_1]:
                rewards[agent.player]=np.zeros(agent.max_units, dtype=np.float32)
                actions[agent.player],log_probs[agent.player],values[agent.player] = agent.act(step=step, obs=obs[agent.player])
            # Environment step
            obs, wins ,terminated, truncated, info = env.step(actions)
            

            dones = {
                k: bool(terminated[k].item()) or bool(truncated[k].item())
                    for k in terminated
                }
            for player in [player_0,player_1]:
                new_cumulative_reward=cumulative_reward(obs[player.player],previous_obs[player.player],player)
                gamerewards[player.player]+= new_cumulative_reward
                rewards[player.player]+=new_cumulative_reward
            
            ###implement_awards
            
            if training :
                # Store experience for each unit
                for agent in [player_0, player_1]:
                    # Store transition in memory
                    agent.memory.add(
                        log_probs[agent.player],  # list of log_probs for each agent
                        values[agent.player],      # single value from the critic
                        rewards[agent.player],    # list of rewards
                        dones[agent.player]
                    )
            if dones["player_0"] or dones["player_1"]:
                game_done = True
            step += 1
                
        
        # At episode end, we do an A2C update
        # final next_value can be assumed 0 if episode ended
        # in more advanced usage, you'd handle partial episodes, but let's keep it simple.
        for player in [player_0,player_1]:
            player.train_on_episode_memory()
            game_rewards_history[player.player].append(np.sum(gamerewards[player.player]))
            if (game+1) % 2 == 0:
                avg_reward = np.mean(game_rewards_history[player.player][-5:])
                print(f"Player:{player.player}Episode {game+1}, avg reward over last 2 episodes: {avg_reward:.2f}")
            player.memory.clear()
        if training:
                    player_0.save_model()
                    player_1.save_model()
        

        
        end_time = time.time()
        try:
            print(f"Game time: {end_time-start_time}") 
        except: pass
    env.close()
    if training:
      player_0.save_model()
      player_1.save_model()

    print("Training complete.")

# Training
evaluate_agents(Agent, Agent, replay=True, games_to_play=100) 
#evaluate_agents(Agent, Agent, replay=False, games_to_play=10000)
#evaluate_agents(Agent, Agent, replay=True, games_to_play=10) # 250*
