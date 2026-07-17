import tiktoken
import torch

from torch import Tensor


class Tokenizer:

    def __init__(self):
        self.encoding = tiktoken.get_encoding('gpt2')

    @property
    def vocabulary_size(self) -> int:
        return 50_257

    def __decode(self, token_id: int) -> str:
        return self.encoding.decode_single_token_bytes(token_id).decode('utf-8', errors='replace')

    def tokenize(self, text: str) -> Tokens:
        token_ids = self.encoding.encode(text, allowed_special={'<|endoftext|>'})
        tokens = [self.__decode(token_id) for token_id in token_ids]
        return Tokens(tokens, token_ids)


class Tokens:

    tokens: list[str]
    token_ids: list[int]

    def __init__(self, tokens: list[str], token_ids: list[int]):
        self.tokens = tokens
        self.token_ids = token_ids

    @property
    def tensor(self) -> Tensor:
        return torch.tensor(self.token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)
