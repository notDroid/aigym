'''
Contains important utility functions.
'''

import torch
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        
        self.tree = [0 for i in range(2 * self.capacity)]
    
    def update(self, idx, value):

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.tree[idx] = value

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def sum(self):
        return self.tree[1]

class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        
        self.tree = [float('inf') for i in range(2 * self.capacity)]
    
    def update(self, idx, value):

        # Leaf of the binary tree
        idx += self.capacity
        # Set the priority at the leaf
        self.tree[idx] = value

        # Update tree, by traversing along ancestors.
        # Continue until the root of the tree.
        while idx >= 2:
            # Get the index of the parent node
            idx //= 2
            # Value of the parent node is the sum of it's two children
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])

    def min(self):
        return self.tree[1]

class TorchEnv:
    def __init__(self, env, continuous: bool = False) -> None:
        self.env = env
        self.continuous = continuous

    def step(self, action):
        if self.continuous: 
            action = action.squeeze(1).numpy()
        else: 
            action = action.squeeze().numpy()
        state, reward, terminated, truncated, info = self.env.step(action)

        state = torch.from_numpy(state).type(torch.float).view(-1, state.shape[-1])
        reward = torch.tensor(reward, dtype = torch.float).view(-1,1)
        terminated = torch.tensor(terminated, dtype = torch.float).view(-1,1)

        return state, reward, terminated, truncated, info
    
    def reset(self):
        state, info = self.env.reset()

        state = torch.from_numpy(state).type(torch.float).view(-1, state.shape[-1])

        return state, info
    
    def close(self):
        self.env.close()

class ReplayBuffer:
    """A simple torch replay buffer."""

    def __init__(self, state_dim: int, capacity: int):
        self.state_buf = torch.zeros((capacity, state_dim))
        self.next_state_buf = torch.zeros((capacity, state_dim))
        self.action_buf = torch.zeros((capacity, 1), dtype = torch.int64)
        self.reward_buf = torch.zeros((capacity, 1))
        self.done_buf = torch.zeros((capacity, 1), dtype = torch.float)
        self.capacity = capacity
        self.ptr, self.size, = 0, 0

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor, 
        reward: torch.Tensor, 
        next_state: torch.Tensor, 
        done: torch.Tensor,
    ):
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        sample_size = min(self.size, batch_size)
        idx = torch.randperm(self.size)[:sample_size]

        return dict(
            state = self.state_buf[idx],
            next_state = self.next_state_buf[idx],
            action = self.action_buf[idx],
            reward = self.reward_buf[idx],
            done = self.done_buf[idx]
        )

    def __len__(self) -> int:
        return self.size

class PriorityReplayBuffer:
    def __init__(self, state_dim, capacity: int = 2**10, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        # Maintain segment binary trees to take sum and find minimum over a range
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        self.priorities = torch.zeros((self.capacity,1), dtype = torch.float)

        self.max_priority = 1.

        # Arrays for buffer
        self.data = {
            'state': torch.zeros((self.capacity, state_dim)),
            'action': torch.zeros((self.capacity, 1), dtype = torch.int64),
            'reward': torch.zeros((self.capacity, 1)),
            'next_state': torch.zeros((self.capacity, state_dim)),
            'done': torch.zeros((self.capacity, 1), dtype = torch.float)
        }
        # We use cyclic buffers to store data, and `next_idx` keeps the index of the next empty
        # slot
        self.next_idx = 0

        # Size of the buffer
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """ Add sample to queue
        """

        # Get next available slot
        idx = self.next_idx

        # store in the queue
        self.data['state'][idx] = state
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_state'][idx] = next_state
        self.data['done'][idx] = done

        # Increment next available slot
        self.next_idx = (idx + 1) % self.capacity
        # Calculate the size
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha

        # Update the two segment trees for sum and minimum
        self.priorities[idx] = priority_alpha
        self.min_tree.update(idx, priority_alpha)
        self.sum_tree.update(idx, priority_alpha)

    def sample(self, batch_size, beta = 0.4):
        """ Sample from buffer
        """
        size = min(batch_size, self.size)

        priority_sum = self.sum_tree.sum()
        prob_min = self.min_tree.min() / priority_sum
        probs = self.priorities / priority_sum

        idx = torch.from_numpy(np.random.choice(self.capacity, size = size, p = probs.squeeze().numpy(), replace = False))
        max_weight = (prob_min * self.size + 1e-6) ** (-beta)

        prob = probs[idx]
        weight = (prob * self.size) ** (-beta)
        weight /= max_weight

        return dict(
            state = self.data['state'][idx],
            next_state = self.data['next_state'][idx],
            action = self.data['action'][idx],
            reward = self.data['reward'][idx],
            done = self.data['done'][idx],
            weight = weight,
            indices = idx
        )

    def update_priorities(self, indices, priorities):
        """ Update priorities
        """

        for i in range(indices.shape[0]):
            idx = indices[i]
            priority = priorities[i].item()

            # Calculate priority
            priority_alpha = priority ** self.alpha
            self.priorities[idx] = priority_alpha

            # Set current max priority
            self.max_priority = max(self.max_priority, priority_alpha)

            # Update the trees
            self.sum_tree.update(idx, priority_alpha)
            self.min_tree.update(idx, priority_alpha)

    def is_full(self):
        """ Whether the buffer is full
        """

        return self.capacity == self.size
    
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

        action = agent.policy(state)
        state, reward, terminated, truncated, info = env.step(action)  
        cumulitve_reward += reward.item()

        if terminated or truncated: return cumulitve_reward

def plot_history(history, name = 'reward', beta = None) -> None:
    ''' Visualize performance including an ewma
    '''
    # Create ewma
    ewma = 0
    episode_ewma = []

    if not beta:
        beta = max(1 - 100/(len(history[name])), 0.5)
    for t, metric in enumerate(history[name]):
        t += 1
        ewma = beta * ewma + (1 - beta) * metric
        corrected = ewma/(1 - beta**t)
        episode_ewma.append(corrected)
    
    # Plot results
    plt.figure(figsize = (10,5))

    plt.plot(history['timestep'], history[name], label = name, alpha = 0.5, color = 'green')
    plt.plot(history['timestep'], episode_ewma, label = f'smooth {name}', color = 'purple', linewidth = 2.5)
    plt.plot(history['eval_timestep'], history[f'eval_{name}'], label = f'eval {name}', color = 'red', linewidth = 2)

    plt.legend(loc = 'best')
    plt.title(f'{name} over training timesteps', weight = 'bold')
    
    plt.ylabel(name)
    plt.xlabel('timestep')

    plt.grid(True)

    plt.show()   

    plt.figure(figsize = (10,5))

    plt.plot(history['timestep'], history['loss'], color = 'red', linewidth = 2)

    plt.title('loss over training timesteps', weight = 'bold')
    
    plt.ylabel('loss')
    plt.xlabel('timestep')

    plt.grid(True)

    plt.show()   


def save_agent(agent, attribute_names: list, path: str, model_names: list):
    ''' Saves all networks in agent
    
    For agents with more than one network.
    '''
    for i in range(len(attribute_names)):
        save_model(getattr(agent,attribute_names[i]), path, model_names[i])

def save_model(model, path: str, model_name: str) -> None:
    ''' Save model
    '''
    # Check if path exists already
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get full path
    model_path = os.path.join(path, model_name)
    
    # Save model to path
    print(f"[INFO] Saving model to: {model_path}")
    torch.save(obj = model.state_dict(), f = model_path)

def play_game(env, agent, iterations = None) -> None:
    ''' Watch the game be played by an agent
    
    Args:
        agent
        iterations: None or Integer, number of steps taken
    '''

    # Initialize environment
    state, info = env.reset()
    done = False
    i = 0
    
    while not done:

        i += 1

        # Get best action from agent
        action = agent.policy(state)
        
        # Take step
        state, reward, terminated, truncated, info = env.step(action)

        # Check for termination       
        if terminated or truncated:
            if not iterations: break
            state, info = env.reset()            
        
        if iterations and iterations <= i: break
        

    env.close()

from matplotlib.animation import FuncAnimation
def play_histogram(env, agent):

    # Initialize environment
    state, info = env.reset()
    bins = np.linspace(agent.q_network.v_min, agent.q_network.v_max, agent.q_network.n_atom)
    T = 0
    values = []
    
    while True:

        T += 1

        # Get best action from agent
        dist = agent.q_network.dist(state).detach()
        action = agent.sample(state, epsilon = 0.5)
        values.append(dist[:,action.item(),:].squeeze().numpy())
        
        # Take step
        state, reward, terminated, truncated, info = env.step(action)

        # Check for termination       
        if terminated or truncated:
            break
            state, info = env.reset()  

    env.close()

    def animate(t):
        plt.cla()
        print(t)
        plt.bar(bins, values[t], 2, color = 'g')

    ani = FuncAnimation(plt.gcf(), animate)

    plt.tight_layout()
    plt.show()

class RolloutBuffer:
    def __init__(self, gamma, gae_lambda = 0.98, n_env = 8):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_env = n_env

        self.buffer = defaultdict(list)
        self.attributes = ('state', 'action', 'log_prob', 'reward', 'value', 'done')

    def push(self, *args):
        for arg, attibute in zip(args, self.attributes):
            self.buffer[attibute].append(arg)

    def stack(self):
        for key, value in self.buffer.items():
            self.buffer[key] = torch.stack(value, dim = 0).view(-1, value[0].shape[-1])

    def compute_gae(self, final_value):
        timesteps = len(self.buffer['reward'])

        for i in range(self.n_env):
            next_value = final_value[i]
            g = 0
            step = 0

            for j in reversed(range(timesteps)):
                value = self.buffer['value'][j][i]
                done = self.buffer['done'][j][i]
                step *= (1 - done)
                step += 1

                delta = self.buffer['reward'][j][i] + self.gamma * (1 - done) * next_value - value
                g = delta + self.gamma * self.gae_lambda * g * (1 - done)
                g_corrected = g#/(1 - (self.gae_lambda**(step)))

                self.buffer['reward'][j][i] = g_corrected
                next_value = value

def plot_ppo_history(history, beta = None) -> None:
    ''' Visualize performance including an ewma
    '''
    # Create ewma
    ewma = 0
    episode_ewma = []

    if not beta:
        beta = max(1 - 100/(len(history['reward'])), 0.5)
    for t, metric in enumerate(history['reward']):
        t += 1
        ewma = beta * ewma + (1 - beta) * metric
        corrected = ewma/(1 - beta**t)
        episode_ewma.append(corrected)
    
    # Plot results
    plt.figure(figsize = (10,5))

    plt.plot(history['reward'], label = 'reward', alpha = 0.5, color = 'green')
    plt.plot(episode_ewma, label = 'smooth reward', color = 'purple', linewidth = 2.5)

    plt.legend(loc = 'best')
    plt.title('reward over training generations', weight = 'bold')
    
    plt.ylabel('reward')
    plt.xlabel('generation')

    plt.grid(True)

    plt.show()   

    plt.figure(figsize = (10,5))

    plt.plot(history['loss'], color = 'red', linewidth = 2)

    plt.title('loss over training generations', weight = 'bold')
    
    plt.ylabel('loss')
    plt.xlabel('generation')

    plt.grid(True)

    plt.show()   