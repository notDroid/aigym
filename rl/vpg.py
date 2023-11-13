import torch
import math

class VPGOptimizer:
    def __init__(self, agent, value_optimizer, policy_optimizer, gamma = 0.99, entropy_weight = 1e-4) -> None:
        self.agent = agent
        self.value_optimizer = value_optimizer
        self.policy_optimizer = policy_optimizer
        self.gamma = gamma
        self.value_loss_fn = torch.nn.MSELoss()
        self.value_weight = 0.5
        self.entropy_weight = entropy_weight
        self.max_grad_norm = 1.0

    def vpg_loss_fn(self, advantage, log_prob):
        ''' Vanilla Policy Gradient Loss

        Args:
            prob: Tensor, policy network output
            advantage: Tensor, detached advantage

        Returns:
            loss: Float
        '''
        loss = - (advantage * log_prob)
        return loss.mean()

    def compute_return(self, rewards):
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
            g = reward + self.gamma * g
            returns[-i] = g

        return returns


    def step(self, batch):
        self.agent.value_network.train()
        self.agent.policy_network.train()

        rstg = self.compute_return(batch['reward'])

        reward = torch.concat(rstg, 0)
        action = torch.concat(batch['action'], 0)
        state = torch.concat(batch['state'], 0)
        value = self.agent.value_network(state)

        advantage = reward - value.detach()
        advantage = (advantage - advantage.mean())/advantage.std()

        data_len = advantage.shape[0]

        loss = 0

        steps_per_epoch = 10
        batch_size = math.ceil(data_len/steps_per_epoch)

        for i in range(1):
            loss += self.train_step(state, action, advantage, reward, data_len, batch_size, steps_per_epoch)

        return loss

    def train_step(self, state, action, advantage, value_target, data_len, batch_size, steps_per_epoch) -> float:
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
            batch_advantage = advantage[idx]
            batch_value_target = value_target[idx]
            batch_value = self.agent.value_network(batch_state)

            prob = torch.distributions.Categorical(self.agent.policy_network(batch_state))

            self.value_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()
                
            # Get loss and do backprop
            policy_loss = self.vpg_loss_fn(batch_advantage, prob.log_prob(batch_action.T).T)
            value_loss = self.value_loss_fn(batch_value, batch_value_target)

            loss = policy_loss + self.value_weight * value_loss - self.entropy_weight * prob.entropy().mean()
            loss.backward()  

            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(self.agent.policy_network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.agent.value_network.parameters(), self.max_grad_norm)

            # Update parameters
            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Keep track of loss
            avg_loss += loss.detach().numpy()

        # Average over loss
        avg_loss /= steps_per_epoch

        return avg_loss

