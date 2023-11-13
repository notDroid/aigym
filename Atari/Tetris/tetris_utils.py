import gymnasium as gym
from utils import TorchEnv

def make_env(n_env: int = 16, truncation: int = 10_000, render_mode = None):
    vec_env = gym.vector.make('ALE/Tetris-v5', max_episode_steps = truncation, num_envs = n_env, obs_type = 'ram')
    vec_env = NormalizePixels(vec_env)
    vec_env = TorchEnv(vec_env)

    test_env = gym.make('ALE/Tetris-v5', max_episode_steps = truncation, obs_type = 'ram', render_mode = render_mode)
    test_env = NormalizePixels(test_env)
    test_env = TorchEnv(test_env)

    return vec_env, test_env

class NormalizePixels:
    def __init__(self, env) -> None:
        self.env = env

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        state = state/255.0
        return state, reward, terminated, truncated, info
    
    def reset(self):
        state, info = self.env.reset()
        return state, info
    
    def close(self):
        self.env.close()

