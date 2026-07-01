from torch import softmax, Tensor
from torch.nn import Linear, Module


class SelfAttention(Module):

    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.W_query = Linear(d_in, d_out, bias)
        self.W_key = Linear(d_in, d_out, bias)
        self.W_value = Linear(d_in, d_out, bias)

    def forward(self, input: Tensor) -> Tensor:
        queries = self.W_query(input)
        keys = self.W_key(input)
        values = self.W_value(input)

        attention_scores = queries @ keys.T
        attention_weights = softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vector = attention_weights @ values
        return context_vector
