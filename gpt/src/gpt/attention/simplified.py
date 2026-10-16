import torch
import torch.nn as nn

from torch import Tensor


class SimplifiedSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        attention_scores = x @ x.T
        attention_weights = torch.softmax(attention_scores, dim=-1)

        context_vector = attention_weights @ x
        return context_vector
