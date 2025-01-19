from training_loop import evaluate_agents
from agent import Agent

evaluate_agents(Agent, Agent, training=False, games_to_play=1)