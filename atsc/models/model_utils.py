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

    def forward(self, x: torch.Tensor):
        for l in self.layers[:-1]:
            x = self.act_fn(l(x))
        return self.layers[-1](x)
