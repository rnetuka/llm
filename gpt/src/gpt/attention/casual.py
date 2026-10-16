import torch
import torch.nn as nn

from gpt.config import CONTEXT_LENGTH, GptConfig
from torch import Tensor


class CasualAttention(nn.Module):

    def __init__(self, config: GptConfig):
        super().__init__()
        dim = config.embedding_dimensions
        self.W_query = nn.Linear(dim, dim)
        self.W_key = nn.Linear(dim, dim)
        self.W_value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.drop_rate)
        self.register_buffer('mask', torch.triu(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH), diagonal=1))

    def forward(self, x: Tensor) -> Tensor:
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector
