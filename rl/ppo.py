'''
Contains q learning training algorithim.
'''
# Access rl modules
import sys
sys.path.append('/home/wali/Downloads/vsprojects/aigym/utils')

import torch
import utils
from torch.distributions import Categorical
import math
from tqdm import tqdm

class PPOOptimizer:
    def __init__(self, agent, optimizer, epsilon = 0.1, entropy_weight = 1e-2, value_weight = 0.5, max_grad_norm = 1.0):
        self.entropy_weight = entropy_weight 
        self.value_weight = value_weight
        self.value_loss_fn = torch.nn.MSELoss()
        self.ppo_loss_fn = self.PPOLoss(epsilon)
        self.optimizer = optimizer
        self.agent = agent
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def PPOLoss(epsilon):
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

    def step(self, buffer, epochs, steps_per_epoch = 32):
        self.agent.policy_network.train()
        self.agent.value_network.train()

        buffer.stack()
        buffer = buffer.buffer

        state = buffer['state']
        action = buffer['action']
        log_prob = buffer['log_prob']
        value_target = buffer['reward']
        value = buffer['value']

        advantage = value_target - value
        advantage = (advantage - advantage.mean())/(advantage.std() + 1e-6)

        data_len = state.shape[0]
        batch_size = math.ceil(data_len / steps_per_epoch)

        avg_loss = 0
        for epoch in range(epochs):
            loss = self.train_step(state, action, log_prob, advantage, value_target, data_len, batch_size, steps_per_epoch)

            avg_loss += loss

        avg_loss /= epochs

        return avg_loss


    def train_step(self, state, action, log_prob, advantage, value_target, data_len, batch_size, steps_per_epoch) -> float:
        ''' Preform gradient descent step

        Args:
            states: Tensor
            actions: Tensor
            returns: Tensor
            agent
            loss_fn: Loss function
            pol_opt: Policy optimizer
            val_opt: Value optimizer
            pol_scheduler: Policy scheduler
            val_scheduler: Value scheduler

        Returns:
            loss: Float
        '''
        random_indices = torch.randperm(data_len)
        avg_loss = 0
        for start in range(0, data_len, batch_size):
            end = min(start + batch_size, data_len)
            idx = random_indices[start:end]
            
            batch_state = state[idx]
            batch_action = action[idx]
            batch_log_prob = log_prob[idx]
            batch_advantage = advantage[idx]
            batch_value_target = value_target[idx]

            # Get predictions
            value = self.agent.value_network(batch_state)
            prob = Categorical(self.agent.policy_network(batch_state))
            pred_log_prob = prob.log_prob(batch_action.squeeze()).unsqueeze(1)

            self.optimizer.zero_grad()
            
            # Get loss and do backprop
            policy_loss = self.ppo_loss_fn(pred_log_prob, batch_log_prob, batch_advantage)
            value_loss = self.value_loss_fn(value, batch_value_target)

            loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * prob.entropy().mean()
            loss.backward()

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.agent.policy_network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.agent.value_network.parameters(), self.max_grad_norm)

            # Update parameters
            self.optimizer.step()

            # Keep track of loss
            avg_loss += loss.detach().numpy()

        # Average over loss
        avg_loss /= steps_per_epoch

        return avg_loss


class PPODataCollector:
    def __init__(self, agent, env, n_env, gamma, gae_lambda = 0.98):
        self.agent = agent
        self.env = env
        self.n_env = n_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def __call__(self, timesteps):
        ''' Play through episode

        Args:
            agent
            env
            rewards_norm: reward_normalization

        Returns:
            episode_length: Integer
            states: List
            rewards: List
            actions: List
        '''
        self.agent.policy_network.eval()
        self.agent.value_network.eval()

        # Reset environment
        initial_state, info = self.env.reset()

        buffer = utils.RolloutBuffer(self.gamma, self.gae_lambda, self.n_env)

        ### Sampling
        for t in tqdm(range(timesteps)):
            # Sample
            with torch.inference_mode():
                action, prob = self.agent.sample(initial_state)
                value = self.agent.value_network(initial_state)

            final_state, reward, terminated, truncated, info = self.env.step(action)   
            
            # calculate log_prob
            log_prob = prob.log_prob(action.squeeze()).unsqueeze(1)
            
            buffer.push(initial_state, action, log_prob, reward, value, terminated)
                
            initial_state = final_state

        with torch.inference_mode():
            final_value = self.agent.value_network(initial_state)

        buffer.compute_gae(final_value)

        return buffer