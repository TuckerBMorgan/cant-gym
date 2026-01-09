"""Deep Q-Network model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HIDDEN_LAYER_SIZE


class DQN(nn.Module):
    """Deep Q-Network for CartPole."""

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, HIDDEN_LAYER_SIZE)
        self.layer2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.layer3 = nn.Linear(HIDDEN_LAYER_SIZE, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Called with either one element to determine next action, or a batch
        during optimization. Returns tensor([[left0exp,right0exp]...]).
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
