import torch
import torch.nn as nn

from config import EMBEDDING_DIMENSIONS, CONTEXT_LENGTH, DROP_RATE, QKV_BIAS
from torch import Tensor


class CasualAttention(nn.Module):

    def __init__(self,
                 d_in: int = EMBEDDING_DIMENSIONS,
                 d_out: int = EMBEDDING_DIMENSIONS,
                 context_length: int = CONTEXT_LENGTH,
                 dropout_rate: float = DROP_RATE,
                 qkv_bias: bool = QKV_BIAS):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input: Tensor) -> Tensor:
        b, num_tokens, d_in = input.shape

        queries = self.W_query(input)
        keys = self.W_key(input)
        values = self.W_value(input)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector
