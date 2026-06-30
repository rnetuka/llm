import torch
from torch import Tensor


class Token:

    string: str
    id: int

    def __init__(self, string: str, id: int):
        self.string = string
        self.id = id

    def __str__(self):
        return self.string



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
