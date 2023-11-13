'''
Testing trained models.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/rl')

import utils
from tetris_utils import make_env
import torch
import config
from agents import ActorCriticAgent
from models import FeedForwardNetwork
import gymnasium as gym

# Create environment
_, env = make_env(render_mode = 'human')

# Load agent from save path
policy_network = FeedForwardNetwork(**config.policy_model_params)
value_network = FeedForwardNetwork(**config.value_model_params)
policy_network.load_state_dict(torch.load(config.full_policy_path))
value_network.load_state_dict(torch.load(config.full_value_path))
agent = ActorCriticAgent(policy_network, config.state_dim, config.action_dim, value_network = value_network)

# Watch the agent play
utils.play_game(env, agent, iterations = None)

# Save video
env = utils.TorchEnv(gym.make('MountainCar-v0', max_episode_steps = 200, render_mode = 'rgb_array'))
recorder = gym.wrappers.RecordVideo(env, config.video_path)

#utils.play_game(recorder, agent)