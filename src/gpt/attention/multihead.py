import torch
import torch.nn as nn

from gpt.config import CONTEXT_LENGTH, GptConfig
from torch import Tensor


class MultiHeadAttention(nn.Module):

    d_out: int
    num_heads: int # number of heads in the multi-head attention
    head_dim: int
    W_query: nn.Linear # matrix of queries
    W_key: nn.Linear # matrix of keys
    W_value: nn.Linear # matrix of values
    out_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config: GptConfig):
        super().__init__()

        d_in = config.embedding_dimensions
        d_out = config.embedding_dimensions
        drop_rate = config.drop_rate
        num_heads = config.attention_heads
        qkv_bias = config.qkv_bias

        if d_out % num_heads != 0:
            raise ValueError('d_out must be divisible by num_heads')

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(drop_rate)
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector
