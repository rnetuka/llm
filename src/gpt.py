import torch
import torch.nn as nn

from config import EMBEDDING_DIMENSIONS, DROP_RATE, VOCABULARY_SIZE, CONTEXT_LENGTH, N_LAYERS
from normalization import LayerNormalization
from torch import Tensor
from transformer import TransformerBlock


class GptModel(nn.Module):

    # token embedding layer transforming every word in the vocabulary into N dimensional vector
    tok_emb: nn.Embedding

    # temperature parameter for token selection process
    # value > 0 (default is 1)
    # - values between 0 and 1 results in sharper probability distribution, making the model to select the most likely
    #   token almost every time
    # - values greater than 1 results in more uniform probability distribution, making the model select tokens more
    #   variably, but with a risk of producing nonsense output
    temperature: float | int

    # top K parameter for token selection process
    # if provided, select top K probability values and choose only among them
    top_k: int | None

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
        self.temperature = 1
        self.top_k = None

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

    def apply_top_k(self, logits: Tensor) -> Tensor:
        if self.top_k is not None:
            logits, _ = torch.topk(logits, k=self.top_k)
            min_value = logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        return logits

    def apply_temperature(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def generate_text(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            cropped_context = context[:, -CONTEXT_LENGTH:]
            with torch.no_grad():
                logits = self(cropped_context)

            logits = logits[:, -1, :]          # last vector, corresponding to the next token
            logits = self.apply_top_k(logits)
            logits = self.apply_temperature(logits)  # scale with temperature
            probabilities = torch.softmax(logits, dim=-1)  # converts vector into probability distribution

            # select a token with large probability score (choose among largest values)
            next_token = torch.multinomial(probabilities, num_samples=1)

            context = torch.cat((context, next_token), dim=1) # appends sampled index to the running sequence

        return context
