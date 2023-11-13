import torch
import numpy as np

class Agent:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
    
class QAgent(Agent):
    def __init__(self, q_network, state_dim: int, action_dim: int) -> None:
        super().__init__(state_dim, action_dim)
        self.q_network = q_network

    def sample(self, state, epsilon):
        ''' Use epsilon greedy sampling

        Args:
            state: state vector from env
        
        Returns:
            action
        '''
        # Randomly decide with probabilty epsilon to choose randomly instead of greedily
        greedy = np.random.rand() > epsilon

        if greedy:
            return self.policy(state)
        
        action = torch.randint(low = 0, high = self.action_dim, size = (1,1))
        
        return action
    
    def policy(self, state):
        '''Get best action according to Q network

        Args:
            state: List state from env

        Returns:
            policy: Integer, best action
        '''
        self.q_network.eval()

        with torch.inference_mode():
            action = self.q_network(state).argmax(keepdims = True).type(torch.int64)
        
        return action
    
class ActorCriticAgent(Agent):
    def __init__(self, policy_network, state_dim: int, action_dim: int, value_network = None) -> None:
        super().__init__(state_dim, action_dim)
        self.policy_network = policy_network
        self.value_network = value_network

    def sample(self, state):
        # Get polciy predictions
        prob = self.policy_network(state)

        prob = torch.distributions.Categorical(prob)

        action = prob.sample().unsqueeze(1)

        return action, prob
    
    def policy(self, state):
        '''Get best action according to Q network

        Args:
            state: List state from env

        Returns:
            policy: Integer, best action
        '''
        self.policy_network.eval()
        with torch.inference_mode():
            prob = self.policy_network(state)
        
        action = prob.argmax(keepdims = True)

        return action
    
class DPGAgent(Agent):
    def __init__(self, policy_network, state_dim: int, action_dim: int, low: float = None, high: float = None, q_network = None, double_q_network = None) -> None:
        super().__init__(state_dim, action_dim)
        self.policy_network = policy_network
        self.q_network = q_network
        self.double_q_network = double_q_network
        self.low = low
        self.high = high

    def sample(self, state, sigma):
        action = self.policy(state)
        action_noise_clip = torch.clamp(torch.normal(action, sigma), self.low, self.high)

        return action_noise_clip

    def policy(self, state):
        self.policy_network.eval()
        with torch.inference_mode():
            action = self.policy_network(state)
        
        return action
    
class SACAgent(Agent):
    def __init__(self, policy_network, state_dim: int, action_dim: int, low: float, high: float, q_network = None, double_q_network = None) -> None:
        super().__init__(state_dim, action_dim)
        self.policy_network = policy_network
        self.q_network = q_network
        self.double_q_network = double_q_network
        self.center = (high + low)/2
        self.scale = (high - low)/2
        self.tanh = torch.nn.Tanh()

    def sample(self, state):
        dist = self.policy_network(state)
        action = self.center + self.scale * self.tanh(dist.rsample())

        return action
    
    def policy(self, state):
        self.policy_network.eval()
        with torch.inference_mode():
            dist = self.policy_network(state)
            action = self.center + self.scale * self.tanh(dist.loc)

        return action
    
    def sample_log_prob(self, state):
        dist = self.policy_network(state)
        x = dist.rsample()
        action = self.center + self.scale * self.tanh(x)

        log_prob = dist.log_prob(x)

        return action, log_prob
