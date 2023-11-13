import torch
from torch import nn
from model_utils import DenseBlock, NoisyLinear

class FeedForwardNetwork(nn.Module):
    '''Neural Network

    Args:
        state_dim: Integer, input dimension
        action_dim: Integer, output dimension
        hidden_dim: Integer, hidden dimension
        blocks: Integer, number of dense blocks
        layers: Integer, layers in each block
    '''
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int = 32, 
                 blocks: int = 1, 
                 layers: int = 3, 
                 dropout = 0, 
                 batchnorm: bool = False, 
                 softmax: bool = False, 
                 tanh: bool = False, 
                 center: float = 0.0, 
                 scale: float = 1.0) -> None:
        super().__init__()

        ffn = nn.ModuleList()
        ffn.extend([
            nn.Linear(in_features = input_dim, out_features = hidden_dim),
            nn.LeakyReLU()
        ])

        for _ in range(blocks):
            ffn.append(
                DenseBlock(dim = hidden_dim, layers = layers)
            )
            if batchnorm:
                ffn.append(
                    nn.BatchNorm1d(num_features = hidden_dim)
                )
            if dropout != 0:
                ffn.append(
                    nn.Dropout1d(p = dropout)
                )

        ffn.append(
            nn.Linear(in_features = hidden_dim, out_features = output_dim)
        )

        if softmax:
            ffn.append(
                nn.Softmax(dim = -1)
            )
        if tanh:
            ffn.append(
                nn.Tanh()
            )
        
        self.center = center
        self.scale = scale
        self.ffn = nn.Sequential(*ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.center + self.ffn(x) * self.scale

class SplitFFN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ffn = FeedForwardNetwork(**kwargs)

    def forward(self, x, y) -> torch.Tensor:
        return self.ffn(torch.concat([x,y], axis = 1))
    
class CategoricalDQN(nn.Module):
    def __init__(self, state_dim, action_dim, v_min, v_max, n_atom = 200, **kwargs) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min)/(n_atom - 1)
        self.n_atom = n_atom
        self.ffn = FeedForwardNetwork(input_dim = state_dim, output_dim = n_atom * action_dim, **kwargs)
        self.values = torch.linspace(v_min, v_max, n_atom).unsqueeze(0).unsqueeze(0)
        self.leaky_relu = nn.LeakyReLU()
        self.noisy_linear = NoisyLinear(n_atom * action_dim, n_atom * action_dim)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x)
        q = (dist * self.values).sum(dim = -1)

        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        x = self.leaky_relu(x)
        x = self.noisy_linear(x)
        logits = x.view(-1, self.action_dim, self.n_atom)
        dist = self.softmax(logits)
        dist = dist.clamp(min = 1e-6)

        return dist
    
    def reset_noise(self):
        self.noisy_linear.reset_noise()

class SoftActor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, log_std_min: float = -5.0, log_std_max: float = 2.0, **kwargs) -> None:
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.output_dim = output_dim
        self.ffn = FeedForwardNetwork(input_dim, 2 * output_dim, **kwargs)

    def forward(self, x):
        mean, log_std = self.ffn(x).chunk(2, dim = -1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        return dist