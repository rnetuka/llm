import torch
import torch.nn as nn

from qwen.attention import GroupedQueryAttention
from qwen.config import QwenConfig
from qwen.feedforward import GLUFeedForward
from qwen.normalization import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(
            d_in=config.embedding_dimension,
            num_heads=config.number_of_heads,
            head_dim=config.head_dimension,
            num_kv_groups=config.number_of_kv_groups,
            qk_norm=config.qk_normalization,
            dtype=config.dtype
        )
        self.ff = GLUFeedForward(config)
        self.norm1 = RMSNorm(config.embedding_dimension)
        self.norm2 = RMSNorm(config.embedding_dimension)

    def forward(self, x: torch.Tensor, mask, cos, sin, start_position: int = 0, cache=None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.attention(x, mask, cos, sin, start_pos=start_position, cache=cache)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, next_cache
