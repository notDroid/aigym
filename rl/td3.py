'''
Contains q learning training algorithim.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/utils')

import torch
from torch import nn
import old_utils
import copy

class TD3Optimizer:
    def __init__(self, agent, q_optimizer, policy_optimizer, gamma = 0.99, polyak = 0.95, policy_update = 2) -> None:
        self.agent = agent
        self.policy_target_network = copy.deepcopy(agent.policy_network)
        self.q_target_network = copy.deepcopy(agent.q_network)
        self.double_q_target_network = copy.deepcopy(agent.double_q_network)
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        self.gamma = gamma
        self.polyak = polyak
        self.q_loss_fn = nn.MSELoss()
        self.update_number = 0
        self.policy_update = policy_update

    def step(self, batch, sigma, noise_clip = 1.0):
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
        self.agent.double_q_network.train()
        self.policy_target_network.eval()
        self.double_q_target_network.eval()
        self.q_target_network.eval()

        # Get predictions
        q_pred = self.agent.q_network(batch['state'], batch['action'])
        double_q_pred = self.agent.double_q_network(batch['state'], batch['action'])
        # Get target
        
        with torch.inference_mode():
            target_action = self.policy_target_network(batch['next_state'])

            epsilon = torch.clamp(sigma * torch.randn_like(target_action), - noise_clip, noise_clip)
            target_action = torch.clamp(target_action + epsilon, self.agent.low, self.agent.high)

            q_target_pred = self.q_target_network(batch['next_state'], target_action)
            double_q_target_pred = self.double_q_target_network(batch['next_state'], target_action)
        q_target = batch['reward'] + (1 - batch['done']) * self.gamma * torch.min(q_target_pred, double_q_target_pred)

        self.q_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        
        # Get loss and do backprop
        q_loss = 0.5 * self.q_loss_fn(q_pred, q_target) + 0.5 * self.q_loss_fn(double_q_pred, q_target)

        q_loss.backward()
        self.q_optimizer.step()
        self.q_polyak_update()

        loss = q_loss.detach().numpy()

        if self.update_number % self.policy_update == 0:
            self.agent.q_network.eval()
            policy_pred = self.agent.policy_network(batch['state'])
            policy_loss = - self.agent.q_network(batch['state'], policy_pred).mean()

            policy_loss.backward()
            self.policy_optimizer.step()
            self.policy_polyak_update()

            #loss += policy_loss.detach().numpy()

        self.update_number += 1

        return loss
    
    def policy_polyak_update(self):
        for param, target_param in zip(self.agent.policy_network.parameters(), self.policy_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    def q_polyak_update(self):
        for param, target_param in zip(self.agent.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
        
        for param, target_param in zip(self.agent.double_q_network.parameters(), self.double_q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    