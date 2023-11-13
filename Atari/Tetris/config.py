'''
Contains parameter choices.
'''
import os

# Policy and value network parameter dictionary
state_dim = 128
action_dim = 5

policy_model_params = {'input_dim': state_dim, 'output_dim': action_dim, 'blocks': 2, 'layers': 3, 'hidden_dim': 128, 'softmax': True}
value_model_params = {'input_dim': state_dim, 'output_dim': 1, 'blocks': 2, 'layers': 3, 'hidden_dim': 128}

# Policy learning params
policy_training_params = {
    'generations': 1,
    'n_env': 32,
    'timesteps': 1_000,
    'epochs': 16,
    'steps_per_epoch': 128,
    'gamma': 0.99,
    'gae_lambda': 0.98,
    'epsilon': 0.1,
    'policy_lr': 1e-3,
    'value_lr': 1e-3,
    'update_period': 1,
    'entropy_weight': 1e-5
}

path = os.path.join('aigym', 'Atari', 'Tetris', 'Tetris Models')

# Policy and value model save paths
policy_model_name = 'policy_network.pt'
value_model_name = 'value_network.pt'
policy_model_attriutes = ['policy_network', 'value_network']
policy_model_names = [policy_model_name, value_model_name]
full_policy_path = os.path.join(path, policy_model_name)
full_value_path = os.path.join(path, value_model_name)

# Video save path
video_path = os.path.join(path, 'videos')

