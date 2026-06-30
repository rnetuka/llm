import torch
from torch import Tensor
from torch.nn import Linear, Parameter


def random_matrix(n: int, m: int) -> Parameter:
    return Parameter(torch.rand(n, m))


def linear_transformation(dimension_in: int, dimension_out: int, bias=False) -> Linear:
    return Linear(dimension_in, dimension_out, bias)


class SimplifiedSelfAttention:

    def apply(self, input: Tensor) -> Tensor:
        scores = input @ input.T
        weights = torch.softmax(scores, dim=-1)
        context = weights @ input
        return context


class SelfAttention:

    def __init__(self, dimension_in: int, dimension_out: int):
        self.Wq = random_matrix(dimension_in, dimension_out)
        self.Wk = random_matrix(dimension_in, dimension_out)
        self.Wv = random_matrix(dimension_in, dimension_out)

    def apply(self, input: Tensor) -> Tensor:
        queries = input @ self.Wq
        keys = input @ self.Wk
        values = input @ self.Wv
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context = attention_weights @ values
        return context


class SelfAttentionV2:

    def __init__(self, dimension_in: int, dimension_out: int, bias: bool = False):
        self.query_transformation = Linear(dimension_in, dimension_out, bias)
        self.key_transformation = Linear(dimension_in, dimension_out, bias)
        self.value_transformation = Linear(dimension_in, dimension_out, bias)

    def apply(self, input: Tensor) -> Tensor:
        queries = self.query_transformation(input)
        keys = self.query_transformation(input)
        values = self.value_transformation(input)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context = attention_weights @ values
        return context


class CasualAttention:

    def __init__(self, dimension_in: int, dimension_out: int, bias: bool = False):
        self.query_transformation = Linear(dimension_in, dimension_out, bias)
        self.key_transformation = Linear(dimension_in, dimension_out, bias)
        self.value_transformation = Linear(dimension_in, dimension_out, bias)

    def apply(self, input: Tensor) -> Tensor:
        queries = self.query_transformation(input)
        keys = self.query_transformation(input)
        values = self.value_transformation(input)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_length = attention_scores.shape[0]
        mask = torch.tril(torch.ones(context_length, context_length))
        mask = attention_weights * mask
        # normalize the mask
        mask = mask / mask.sum(dim=-1, keepdim=True)

        context = attention_weights @ values
        return context