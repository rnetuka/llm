import torch
import torch.nn as nn

from config import GptConfig
from torch import  Tensor


class SelfAttention(nn.Module):

    def __init__(self, config: GptConfig):
        super().__init__()

        d_in = config.embedding_dimensions
        d_out = config.embedding_dimensions
        qkv_bias = config.qkv_bias

        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)

    def forward(self, x: Tensor) -> Tensor:
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vector = attention_weights @ values
        return context_vector
