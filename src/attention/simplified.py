from torch import softmax, Tensor
from torch.nn import Module


class SimplifiedSelfAttention(Module):

    def forward(self, input: Tensor) -> Tensor:
        attention_scores = input @ input.T
        attention_weights = softmax(attention_scores, dim=-1)

        context_vector = attention_weights @ input
        return context_vector
