import torch
from torch import nn
from typing import List


class SiluMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, device: str = 'cuda'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0], device=device))
        for h1, h2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(h1, h2, device=device))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim, device=device))
        self.act_fn = nn.SiLU()
        self.init_weights()

    def forward(self, x: torch.Tensor):
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))
        return self.layers[-1](x)

    def init_weights(self):
        for layer in self.layers:
            for n, p in layer.named_parameters():
                if 'weight' in n:
                    nn.init.kaiming_uniform_(p, nonlinearity='relu')
                elif 'bias' in n:
                    nn.init.zeros_(p)


class ConstantRescale(nn.Module):
    def __init__(self, scale: float, bias: float):
        """
        Elementwise operator: (x + bias) * scale
        """
        super().__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x: torch.Tensor):
        return (x + self.bias) * self.scale


def get_distance(neighbor_link: List[List[int]]):
    """Compute the shortest distance between each node pair using Floyd algorithm. The edge weights are set to 1.
    Args:
        neighbor_link (`List[List[int]]`):
            `neighbor_link[i]` represents the neighbors (index) of node i.
    Returns:
        distance (`List[List[int]]`):
            A square. `distance[i][j]` represents the distance between node i and node j.
    """
    num_nodes = len(neighbor_link)
    distance = [[num_nodes for j in range(num_nodes)] for i in range(num_nodes)]  # initiate with INF
    for i in range(num_nodes):
        distance[i][i] = 0
        for j in neighbor_link[i]:
            distance[i][j] = 1
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance
