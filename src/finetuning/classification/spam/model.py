import torch
import torch.nn as nn

from config import CONTEXT_LENGTH, GPT_2_SMALL, GptConfig
from encoding import Tokenizer
from gpt import GptModel
from pathlib import Path
from torch import Tensor


class SpamClassifier(nn.Module):

    def __init__(self, config: GptConfig = GPT_2_SMALL):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.gpt = GptModel.pretrained(config)

        for param in self.gpt.parameters():  # disable learning for the whole model
            param.requires_grad = False

        self.gpt.output_layer = nn.Linear(config.embedding_dimensions, 2)  # output layer now maps to two classes

        for param in self.gpt.trf_blocks[-1].parameters():  # re-enable learning for last transformer block
            param.requires_grad = True

        for param in self.gpt.final_norm.parameters():  # re-enable learning for normalization layer
            param.requires_grad = True

    #@classmethod
    #def pretrained(cls, config: GptConfig):


    @property
    def number_of_parameters(self) -> int:
        return self.gpt.number_of_parameters

    @property
    def model_size(self) -> str:
        return self.gpt.model_size

    @property
    def state_file(self) -> Path:
        return Path('..') / 'resources' / 'gpt2-small-spam.pth'

    def forward(self, x: Tensor) -> Tensor:
        return self.gpt(x)

    def save(self):
        torch.save(self.state_dict(), self.state_file)

    def load(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(self.state_file, map_location=device))

    def classify(self, text: str) -> bool:
        pad_token = self.tokenizer.encode('<|endoftext|>')[0]

        input = self.tokenizer.encode(text)
        input = input[:CONTEXT_LENGTH]
        input += [pad_token] * (CONTEXT_LENGTH - len(input))

        with torch.no_grad():
            output = self.gpt(torch.tensor(input).unsqueeze(0))

        last_output_token = output[:, -1, :]     # the last output token contains the most precise data, thanks to multi-head attention
        probabilities = torch.softmax(last_output_token, dim=-1)
        label = torch.argmax(probabilities).item()  # 0 = Not spam, 1 = Spam
        return bool(label)
