import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        # input
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        # advantage
        self.ad1 = nn.Linear(64, 32)
        self.ad2 = nn.Linear(32, action_size)
        # value
        self.va1 = nn.Linear(64, 32)
        self.va2 = nn.Linear(32, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # input
        linear_1 = F.relu(self.fc1(state))
        # advantage
        advantage_1 = F.relu(self.ad1(linear_1))
        action_advantage = self.ad2(advantage_1)
        # value
        value_1 = F.relu(self.va1(linear_1))
        state_value = self.va2(value_1)
        # combining
        max_action_advantage = torch.max(action_advantage, dim=1)[0].unsqueeze(1)
        value_state_action = state_value + action_advantage - max_action_advantage 
        
        return value_state_action