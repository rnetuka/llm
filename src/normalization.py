import torch
import torch.nn as nn

from config import EMBEDDING_DIMENSIONS
from torch import Tensor


class LayerNormalization(nn.Module):

    eps: float # epsilon - a very small number added to variance in order to avoid divisions by zero
    scale: nn.Parameter # trainable parameter
    shift: nn.Parameter # trainable parameter

    # both trainable parameters are adjusted during training if it is determined that doing so would improve the
    # model's performance on its training task

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(EMBEDDING_DIMENSIONS))
        self.shift = nn.Parameter(torch.zeros(EMBEDDING_DIMENSIONS))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)     # mean of x1, x2, ... xN = (x1 + x2 + ... + xN) / N
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
