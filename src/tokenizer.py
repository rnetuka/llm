import tiktoken
import torch

from torch import Tensor
from vocabulary import Vocabulary


class Tokenizer:

    def __init__(self):
        self.encoding = tiktoken.get_encoding('gpt2')
        self.vocabulary = Vocabulary()

    def split(self, text: str) -> list[str]:
        token_ids = self.encode(text)
        return [self.vocabulary[token_id] for token_id in token_ids]

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special={'<|endoftext|>'})

    def tokenize(self, text: str) -> Tensor:
        token_ids = self.encode(text)
        return torch.tensor(token_ids).unsqueeze(0)
