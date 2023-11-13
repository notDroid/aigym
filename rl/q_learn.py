import torch
import copy

class QOptimizer:
    def __init__(self, agent, optimizer, gamma = 0.99, polyak = 0.995, n = 1) -> None:
        self.agent = agent
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss(reduction = 'none')
        self.gamma = gamma
        self.q_target_network = copy.deepcopy(agent.q_network)
        self.q_target_network.eval()
        self.polyak = polyak
        self.n = n

    def step(self, batch: dict):
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
        self.agent.q_network.train()

        # Get predictions
        pred = self.agent.q_network(batch['state']).gather(1, batch['action'])

        # Get target
        with torch.inference_mode():
            target_pred = self.q_target_network(batch['next_state']).max(dim = -1, keepdims = True)[0]

        q_target = batch['reward'] + (self.gamma**self.n) * (1 - batch['done']) * target_pred

        self.optimizer.zero_grad()
        
        # Get loss and do backprop
        elementwise_loss = self.loss_fn(pred, q_target)
        loss = (elementwise_loss * batch['weight']).mean()
        #loss = elementwise_loss.mean()

        loss.backward()
        # Update parameters
        self.optimizer.step()        

        return loss.detach().numpy(), elementwise_loss.detach()
    
    def polyak_update(self):
        for param, target_param in zip(self.agent.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    def hard_update(self):
        self.q_target_network.load_state_dict(
            self.agent.q_network.state_dict()
        )

class CategoricalQOptimizer:
    def __init__(self, agent, optimizer, gamma = 0.99, polyak = 0.995, n = 1) -> None:
        self.agent = agent
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss(reduction = 'none')
        self.gamma = gamma
        self.q_target_network = copy.deepcopy(agent.q_network)
        self.q_target_network.eval()
        self.polyak = polyak
        self.support = self.q_target_network.values
        self.n_atom = self.q_target_network.n_atom
        self.v_min = self.q_target_network.v_min
        self.v_max = self.q_target_network.v_max
        self.delta_z = self.q_target_network.delta_z
        self.n = n

    def step(self, batch: dict):
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
        self.agent.q_network.train()

        # Get predictions
        batch_size = batch['action'].shape[0]
        q_pred = self.agent.q_network.dist(batch['state'])[range(batch_size), batch['action'].squeeze(1)]

        # Get target
        with torch.no_grad():
            q_target = self.target_pred(batch['next_state'], batch['reward'], batch['done'])

        self.optimizer.zero_grad()
        
        # Get loss and do backprop
        elementwise_loss = - (q_target * torch.log(q_pred)).sum(-1)
        loss = (elementwise_loss * batch['weight']).mean()
        #loss = elementwise_loss.mean()

        loss.backward()
        # Update parameters
        self.optimizer.step()        

        # reset noise
        self.agent.q_network.reset_noise()
        self.q_target_network.reset_noise()

        return loss.detach().numpy(), elementwise_loss.detach()
    
    def target_pred(self, next_state, reward, done):
        batch_size = next_state.shape[0]

        with torch.no_grad():
            next_action = self.q_target_network(next_state).argmax(1)
            next_dist = self.q_target_network.dist(next_state)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = reward + (1 - done) * (self.gamma**self.n) * self.support
            t_z = t_z.clamp(min = self.v_min, max = self.v_max)
            b = (t_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.n_atom, batch_size
                ).long()
                .unsqueeze(1)
                .expand(batch_size, self.n_atom)
            )

            proj_dist = torch.zeros(next_dist.size())
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )
        
        return proj_dist
    
    def polyak_update(self):
        for param, target_param in zip(self.agent.q_network.parameters(), self.q_target_network.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)

    def hard_update(self):
        self.q_target_network.load_state_dict(
            self.agent.q_network.state_dict()
        )