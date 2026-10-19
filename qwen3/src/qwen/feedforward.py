import torch
import torch.nn as nn

from qwen.config import QwenConfig


class GLUFeedForward(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.layer1 = nn.Linear(
            config.embedding_dimension,
            config.hidden_dimension,
            dtype=config.dtype,
            bias=False
        )
        self.layer2 = nn.Linear(
            config.embedding_dimension,
            config.hidden_dimension,
            dtype=config.dtype,
            bias=False
        )
        self.layer3 = nn.Linear(
            config.hidden_dimension,
            config.embedding_dimension,
            dtype=config.dtype,
            bias=False
        )
        #self.activation = nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #y1 = self.layer1(x)
        #y1 = self.activation(y1)
        #y2 = self.layer2(x)
        #return self.layer3(y1 * y2)
        x_fc1 = self.layer1(x)
        x_fc2 = self.layer2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.layer3(x)
