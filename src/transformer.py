import torch.nn as nn

from activation import GELU
from attention.multihead import MultiHeadAttention
from config import DROP_RATE, EMBEDDING_DIMENSIONS
from normalization import LayerNormalization
from torch import Tensor


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.dropout = nn.Dropout(DROP_RATE)

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

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSIONS, 4 * EMBEDDING_DIMENSIONS),
            GELU(),
            nn.Linear(4 * EMBEDDING_DIMENSIONS, EMBEDDING_DIMENSIONS)
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
