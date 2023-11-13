'''
Contains q learning training algorithim.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/utils')

import torch
import utils
from tetris_utils import make_env
import agents
from collections import defaultdict
from ppo import PPODataCollector, PPOOptimizer



def ppo_learning(agent: agents.ActorCriticAgent,
                generations: int = 501,
                n_env: int = 8,
                timesteps: int = 500,
                epochs: int = 3,
                steps_per_epoch: int = 8,
                gamma: float = 0.99,
                gae_lambda: float = 0.92,
                epsilon: float = 0.1,
                policy_lr: float = 3e-4,
                value_lr: float = 1e-3,
                weight_decay: float = 1e-4,
                entropy_weight: float = 1e-3,
                max_grad_norm: float = 1.0,
                update_period = 50):
    ''' Train agent using policy learning

    Based on Advantage, train the policy network to predict the best action to take probalistically.
    Train the value network to predict the return in order to provide a baseline to compute advantage.

    Args:
        agent: Policy learning agent
        episodes: Integer, number of training episodes
        N: Integer, memory capacity
        gamma: Float, discount factor   
        lr: Float, policy learning rate
        weight_decay: Float, policy weight decay
        val_lr: Float, value learning rate
        val_weight_decay: Float, value weight decay
        tuning: Bool, send plots too see progress (only useful when tuning hyperparams)

    Returns:
        history: Defaultdict of performance over iteratons/episodes
    '''
    # Initialize environment
    env, test_env = make_env(n_env = n_env, truncation = timesteps)

    # Define utility
    history = defaultdict(list)

    # Define optimization utility
    network_optimizers = torch.optim.AdamW([
        {'params': agent.policy_network.parameters(), 'lr': policy_lr, 'weight_decay': weight_decay},
        {'params': agent.value_network.parameters(), 'lr': value_lr, 'weight_decay': weight_decay}
    ])
    
    # Use cosine annealing lr
    network_schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(network_optimizers, generations)
    
    # PPO utility
    rollout = PPODataCollector(agent, env, n_env, gamma, gae_lambda)
    ppo_optimizer = PPOOptimizer(agent, network_optimizers, epsilon, entropy_weight, max_grad_norm = max_grad_norm)

    for generation in range(generations):
        ### Sampling
        buffer = rollout(timesteps)
        
        ### Training
        loss = ppo_optimizer.step(buffer, epochs, steps_per_epoch = steps_per_epoch)
        network_schedulers.step()

        ### Evaluate
        reward = evaluate_policy_step(agent, test_env)

        history['reward'].append(reward)
        history['loss'].append(loss)

        ### Update messages
        if generation % update_period == 0:
            last_lr = network_schedulers.get_last_lr()

            print(f'episode {generation + 1} / {generations} | avg loss: {loss} reward: {reward}| lr: {last_lr}')

    return history  

def evaluate_policy(agent, env, num: int = 5) -> float:
    ''' Evaluate policy

    Args:
        agent
        env
        num: Integer, number of test runs

    Returns:
        avg_episode_length: Float
    '''
    avg_reward = 0

    for _ in range(num):
        # Play episode and compute episode length
        avg_reward += evaluate_policy_step(agent, env)

    return avg_reward/num
def evaluate_policy_step(agent, env):
    ''' Play through episode and return episode length

    Args:
        agent
        env
    
    Returns:
        episode_length
    '''
    state, info = env.reset()
    cumulitve_reward = 0

    # Play through episode
    while True:

        action, _ = agent.sample(state)
        state, reward, terminated, truncated, info = env.step(action)  
        cumulitve_reward += reward.item()

        if terminated or truncated: return cumulitve_reward