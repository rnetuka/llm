from torch import inf, ones, triu, softmax, Tensor
from torch.nn import Dropout, Linear, Module


class CasualAttention(Module):

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout_rate: float = 0.0, bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = Linear(d_in, d_out, bias)
        self.W_key = Linear(d_in, d_out, bias)
        self.W_value = Linear(d_in, d_out, bias)
        self.dropout = Dropout(dropout_rate)
        self.register_buffer('mask', triu(ones(context_length, context_length), diagonal=1))

    def forward(self, input: Tensor) -> Tensor:
        b, num_tokens, d_in = input.shape

        queries = self.W_query(input)
        keys = self.W_key(input)
        values = self.W_value(input)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -inf)

        attention_weights = softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        return context_vector
