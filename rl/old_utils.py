'''
Contains important utility functions.
'''

# Add models path
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/rlearn/models')

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict

    
class EWMA:
    ''' Class to keep track and update an EWMA

    Args:
        beta: Float, momentum parameter
    '''
    def __init__(self, beta: float) -> None:
        # Initialize variables
        self.beta = beta
        self.t = 0
        self.ewma = 0

    def step(self, x) -> float:
        ''' Update the EWMA

        Args:
            x: Float or integer, value to update ewma with

        Returns:
            Float, bias corrected next value in ewma
        '''
        self.ewma = self.ewma * self.beta + x * (1 - self.beta)
        self.t += 1

        # Correct bias
        bias_correction = self.ewma/(1 - (self.beta)**self.t)

        return bias_correction

class ewma_normalization:
    ''' Update reward ewma

    Use better reward function and normalize reward according to a mean and variance ewma.

    Args:
        state: state vector from env

    Returns:
        reward: Float, better reward
    '''
    def __init__(self, beta: float = 0.995) -> None:
        self.beta = beta
        self.mean_ewma = EWMA(beta)
        self.var_ewma = EWMA(beta)

    def step(self, reward) -> float:
        ''' Update reward ewmas and get normalized reward

        Args:
            reward
        
        Returns:
            normalized reward: Float
        '''
        reward_mean = self.mean_ewma.step(reward.mean(dim = 0))
        reward_var = self.var_ewma.step(((reward - reward_mean)**2).mean(dim = 0))
        reward = (reward - reward_mean)/(np.sqrt(reward_var) + 1e-7)

        return reward

def vpg_loss(prob, advantage) -> float:
    ''' Vanilla Policy Gradient Loss

    Args:
        prob: Tensor, policy network output
        advantage: Tensor, detached advantage

    Returns:
        loss: Float
    '''
    return - (advantage * torch.log(prob)).mean()

def compute_return(rewards, gamma: float):
    ''' Compute return

    Args:
        rewards: Tensor, full reward history
        gamma: Float: discount factor

    Returns:
        returns: Tensor
    '''
    returns = [0 for _ in range(len(rewards))]
    g = 0.0

    # Iteratively calculate return in reverse
    for i, reward in enumerate(reversed(rewards)):
        g = reward + gamma * g
        returns[-i] = g

    return returns

def PPOLoss(epsilon = 0.1) -> float:
    ''' Proximal Policy Optimization Loss

    Args:
        prob: Tensor, policy network output
        advantage: Tensor, detached advantage

    Returns:
        loss: Float
    '''
    def loss_fn(log_prob, old_log_prob, advantage):
        ratio = torch.exp(log_prob - old_log_prob)
        clipped = 1 + torch.sign(advantage) * epsilon
        return - torch.min(ratio * advantage, clipped * advantage).mean()
    
    return loss_fn

class RolloutBuffer:
    def __init__(self, gamma, gae_lambda = 0.98):
        self.buffer = defaultdict(list)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.attributes = ('state', 'action', 'log_prob', 'reward', 'value')

    def push(self, *args):
        for arg, attibute in zip(args, self.attributes):
            self.buffer[attibute].append(arg)

    def extend(self, buffer):
        for key, value in buffer.buffer.items():
            self.buffer[key].extend(value)

    def tensorize(self):
        for key, value in self.buffer.items():
            self.buffer[key] = torch.concat(value, 0)

    def compute_returns(self):
        g = 0
        returns = []
        for reward in reversed(self.buffer['reward']):
            g = reward + self.gamma * g
            returns.append(g)
        self.buffer['reward'] = reversed(returns)

    def compute_gae(self):
        values = self.buffer['value']
        
        gae = []
        g = 0
        for step, i in enumerate(reversed(range(len(values) - 1))):
            delta = self.buffer['reward'][i] + self.gamma * values[i + 1] - values[i]
            g = delta + self.gamma * self.gae_lambda * g
            g_corrected = g/(1 + (self.gae_lambda**(step + 1)))
            gae.append(g_corrected)
        self.buffer['reward'] = reversed(gae)