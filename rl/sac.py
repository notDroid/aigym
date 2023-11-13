'''
Contains q learning training algorithim.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/utils')

import torch
from torch import nn
import copy

class SACOptimizer:
    def __init__(self, agent, q_optimizer, policy_optimizer, log_alpha_optimizer = None, gamma = 0.99, polyak = 0.95, log_alpha = 1e-1, max_grad_norm = 1.0) -> None:
        self.agent = agent
        self.q_target_network = copy.deepcopy(agent.q_network)
        self.double_q_target_network = copy.deepcopy(agent.double_q_network)
        self.q_optimizer = q_optimizer
        self.policy_optimizer = policy_optimizer
        self.log_alpha_optimizer = log_alpha_optimizer
        self.gamma = gamma
        self.polyak = polyak
        self.q_loss_fn = nn.MSELoss()
        self.log_alpha = log_alpha
        self.target_entropy = agent.action_dim
        self.max_grad_norm = max_grad_norm

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
        self.agent.policy_network.eval()
        self.agent.q_network.train()
        self.agent.double_q_network.train()
        self.double_q_target_network.eval()
        self.q_target_network.eval()

        # Get predictions
        q_pred = self.agent.q_network(batch['state'], batch['action'])
        double_q_pred = self.agent.double_q_network(batch['state'], batch['action'])
        # Get target
        
        with torch.inference_mode():
            target_action, log_prob = self.agent.sample_log_prob(batch['next_state'])

            q_target_pred = self.q_target_network(batch['next_state'], target_action)
            double_q_target_pred = self.double_q_target_network(batch['next_state'], target_action)
                
        min_q = torch.min(q_target_pred, double_q_target_pred)
        q_target = batch['reward'] + (1 - batch['done']) * self.gamma * (min_q - self.log_alpha.exp().detach() * log_prob)


        self.q_optimizer.zero_grad()

        # Get loss and do backprop
        q_loss = 0.5 * self.q_loss_fn(q_pred, q_target) + 0.5 * self.q_loss_fn(double_q_pred, q_target)

        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.agent.double_q_network.parameters(), self.max_grad_norm)
        self.q_optimizer.step()
        self.polyak_update()

        loss = q_loss.detach().numpy()

        self.agent.q_network.eval()
        self.agent.double_q_network.eval()
        self.agent.policy_network.train()
        self.policy_optimizer.zero_grad()

        policy_pred, log_prob = self.agent.sample_log_prob(batch['state'])
        q_pred = self.agent.q_network(batch['state'], policy_pred)
        double_q_pred = self.agent.double_q_network(batch['state'], policy_pred)
        min_q = torch.min(q_pred, double_q_pred)
        policy_loss = - (min_q - self.log_alpha.exp().detach() * log_prob).mean()

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_network.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        #loss += policy_loss.detach().numpy()

        if self.log_alpha_optimizer:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.log_alpha.exp() * (- log_prob.detach() - self.target_entropy)).mean()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.log_alpha_optimizer.step()

        return loss

    def polyak_update(self):
        for param, target_param in zip(self.agent.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
        
        for param, target_param in zip(self.agent.double_q_network.parameters(), self.double_q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    