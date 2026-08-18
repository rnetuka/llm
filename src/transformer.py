import torch.nn as nn

from activation import GELU
from attention import MultiHeadAttention
from config import GptConfig
from normalization import LayerNormalization
from torch import Tensor


class TransformerBlock(nn.Module):

    def __init__(self, config: GptConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = LayerNormalization(config)
        self.layer_norm_2 = LayerNormalization(config)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class FeedForward(nn.Module):

    def __init__(self, config: GptConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dimensions, 4 * config.embedding_dimensions),
            GELU(),
            nn.Linear(4 * config.embedding_dimensions, config.embedding_dimensions)
        )
        # For batch size 2, 4 tokens in each batch and embedding size 768
        #
        # 1. Linear layer
        #    input:  (2, 4, 768)
        #    output: (2, 4, 3072)
        #
        # 2. GELU activation
        #    input:  (2, 4, 3072)
        #    output: (2, 4, 3072)
        #
        # 3. Linear layer
        #    input:  (2, 4, 3072)
        #    output: (2, 4, 768)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
