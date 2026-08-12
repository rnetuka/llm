import pandas
import torch

from config import CONTEXT_LENGTH
from os import PathLike
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from tokenizer import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
from typing import override


class SpamDataset(Dataset):

    data: DataFrame | TextFileReader

    def __init__(self, data: DataFrame | TextFileReader):
        self.data = data
        self.tokenizer = Tokenizer()
        self.item_length = self.longest_encoded_length()        # every item has to have the same length
        self.pad_token = self.tokenizer.encode('<|endoftext|>')[0]

    @staticmethod
    def from_csv(path: str | PathLike[str]) -> SpamDataset:
        data = pandas.read_csv(path)
        return SpamDataset(data)

    def longest_encoded_length(self) -> int:
        encoded_texts = [self.tokenizer.encode(text) for text in self.data['Text']]
        longest = max(encoded_texts, key=len)
        return len(longest)

    def encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        encoded = encoded + [self.pad_token] * (self.item_length - len(encoded))   # pad text up to item length
        encoded = encoded[:CONTEXT_LENGTH]     # truncate to context length
        return encoded

    def __len__(self) -> int:
        return len(self.data['Text'])

    @override
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        text = self.data['Text'][i]
        label = self.data.iloc[i]['Label']
        return (
            torch.tensor(self.encode_text(text), dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
