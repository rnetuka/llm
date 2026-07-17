import torch.nn as nn

from activation import GELU
from torch import Tensor


class DeepNeuralNetwork(nn.Module):

    def __init__(self, layer_sizes: list[int], use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[1]),
                GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[1], layer_sizes[2]),
                GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[2], layer_sizes[3]),
                GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[3], layer_sizes[4]),
                GELU()
            ),
            nn.Sequential(
                nn.Linear(layer_sizes[4], layer_sizes[5]),
                GELU()
            )
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            output = layer(x)
            if self.use_shortcut and x.shape == output.shape:
                x = x + output
            else:
                x = output
        return x
