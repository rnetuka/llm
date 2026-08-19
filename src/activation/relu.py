import torch
import torch.nn as nn

from torch import Tensor


class ReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.maximum(x, torch.tensor(0.0, device=x.device))
