import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.eps = 1e-6
        self.bias = False
        self.qwen3_complatible=True
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension)) if self.bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        if self.qwen3_complatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)
