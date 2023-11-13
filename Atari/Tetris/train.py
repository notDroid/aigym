'''
Contains algorithim for training reinforcement learning agent.

Important parameters:
    1. gamma, lrs
    2. weight decays
    3. neural network architecture
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/rl')

import utils
import config
import engine
from models import FeedForwardNetwork
import agents

# Create agent
policy_network = FeedForwardNetwork(**config.policy_model_params)
value_network = FeedForwardNetwork(**config.value_model_params)
agent = agents.ActorCriticAgent(policy_network, config.state_dim, config.action_dim, value_network = value_network)

# Train policy agent
history = engine.ppo_learning(agent = agent, **config.policy_training_params)

# Save the policy and value neural networks
utils.save_agent(agent, config.policy_model_attriutes, config.path, config.policy_model_names)

# Visualize performance
utils.plot_ppo_history(history)