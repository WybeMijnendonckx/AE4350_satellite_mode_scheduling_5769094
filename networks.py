"""networks.py

Q-network architecture for the DQN agent
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Simple feedforward network mapping a 14 dimensional observation to
    4 Q-values, one per action (Idle, Sun-point, Nadir-point and Ground-station-pointing)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_layer_sizes: tuple = (64, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)