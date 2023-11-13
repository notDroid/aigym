'''
Contains q learning training algorithim.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/utils')

import torch
from torch import nn
import copy

class DDPGOptimizer:
    def __init__(self, agent, q_optimizer, policy_optimizer, gamma = 0.99, polyak = 0.95) -> None:
        self.agent = agent
        self.policy_target_network = copy.deepcopy(agent.policy_network)
        self.q_target_network = copy.deepcopy(agent.q_network)
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        self.gamma = gamma
        self.polyak = polyak
        self.q_loss_fn = nn.MSELoss()

    def step(self, batch):
        ''' Preform gradient descent step

        Args:
            batch: Memory_unit, random batch from memory
            network: neural Network for predictions
            target_network: Neural network for targetting
            gamma: Float, measures foresight approx 1/(1 - gamma) steps ahead
            optimizer: Pytorch optimizer
            loss_fn: Loss function

        Returns:
            loss: Float
        '''
        self.agent.policy_network.train()
        self.agent.q_network.train()
        self.policy_target_network.eval()
        self.q_target_network.eval()

        # Get predictions
        q_pred = self.agent.q_network(batch['state'], batch['action'])

        # Get target
        with torch.inference_mode():
            target_action = self.policy_target_network(batch['next_state'])
            q_target_pred = self.q_target_network(batch['next_state'], target_action)
        q_target = batch['reward'] + (1 - batch['done']) * self.gamma * q_target_pred

        self.q_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        
        # Get loss and do backprop
        q_loss = self.q_loss_fn(q_pred, q_target)

        q_loss.backward()
        self.q_optimizer.step()

        self.agent.q_network.eval()
        policy_pred = self.agent.policy_network(batch['state'])
        policy_loss = - self.agent.q_network(batch['state'], policy_pred).mean()

        policy_loss.backward()
        self.policy_optimizer.step()

        return q_loss.detach().numpy(), policy_loss.detach().numpy()
    
    def polyak_update(self):
        for param, target_param in zip(self.agent.policy_network.parameters(), self.policy_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

        for param, target_param in zip(self.agent.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

