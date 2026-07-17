import torch
import torch.nn as nn

from config import EMBEDDING_DIMENSIONS, DROP_RATE, VOCABULARY_SIZE, CONTEXT_LENGTH, N_LAYERS
from normalization import LayerNormalization
from torch import Tensor
from transformer import TransformerBlock


class GptModel(nn.Module):

    # token embedding layer transforming every word in the vocabulary into N dimensional vector
    tok_emb: nn.Embedding

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSIONS)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIMENSIONS)
        self.dropout = nn.Dropout(DROP_RATE)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(N_LAYERS)]
        )
        self.final_norm = LayerNormalization()
        self.output_layer = nn.Linear(EMBEDDING_DIMENSIONS, VOCABULARY_SIZE, bias=False)

    def forward(self, in_idx: Tensor) -> Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # device is CPU/GPU based on the input data
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits

    @property
    def number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self) -> str:
        total_size_bytes = self.number_of_parameters * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        return f'{total_size_mb:.2f} MB'


