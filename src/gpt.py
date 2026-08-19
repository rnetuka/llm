from __future__ import annotations

import torch
import torch.nn as nn

from config import device, CONTEXT_LENGTH, GptConfig, VOCABULARY_SIZE
from normalization import LayerNormalization
from pathlib import Path
from pretraining import openai
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

    def __init__(self, config: GptConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCABULARY_SIZE, config.embedding_dimensions)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, config.embedding_dimensions)
        self.dropout = nn.Dropout(config.drop_rate)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNormalization(config)
        self.output_layer = nn.Linear(config.embedding_dimensions, VOCABULARY_SIZE, bias=False)
        self.config = config
        self.temperature = 1
        self.top_k = None
        self.filename_suffix = ''

    @staticmethod
    def pretrained(config: GptConfig) -> GptModel:
        model = GptModel(config)
        model.temperature = 1.5
        model.top_k = 50
        model.to(device)
        model.eval()
        openai.pretrain(model)
        return model

    def forward(self, input: Tensor) -> Tensor:
        batch_size, sequence_length = input.shape
        tok_embeds = self.tok_emb(input)
        pos_embeds = self.pos_emb(torch.arange(sequence_length, device=device))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits

    @property
    def name(self) -> str:
        return self.config.model_name

    @property
    def filename(self) -> str:
        filename = self.name.lower().replace('-', '', count=1).replace(' ', '-')
        filename += self.filename_suffix
        return filename

    @property
    def number_of_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size(self) -> str:
        total_size_bytes = self.number_of_parameters * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        return f'{total_size_mb:.2f} MB'

    @property
    def state_file(self) -> Path:
        return Path('..') / 'resources' / f'{self.filename}.pth'

    def pick_top_k(self, logits: Tensor) -> Tensor:
        if self.top_k is not None:
            top_logits, _ = torch.topk(logits, self.top_k)
            min_val = top_logits[:, -1]
            return torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        return logits

    def generate_text(self, context: Tensor, max_new_tokens: int) -> Tensor:
        for _ in range(max_new_tokens):
            cropped_context = context[:, -CONTEXT_LENGTH:]
            with torch.no_grad():
                logits = self(cropped_context)

            logits = logits[:, -1, :]          # last vector, corresponding to the next token
            logits = self.pick_top_k(logits)
            logits = logits / self.temperature  # scale with temperature
            probabilities = torch.softmax(logits, dim=-1)  # converts vector into probability distribution

            # select a token with large probability score (choose among largest values)
            next_token = torch.multinomial(probabilities, num_samples=1)

            context = torch.cat((context, next_token), dim=1) # appends sampled index to the running sequence

        return context.squeeze(0)

    def save(self):
        torch.save(self.state_dict(), self.state_file)

    def load(self):
        self.load_state_dict(torch.load(self.state_file, map_location=device))
