from .data import Entry
from .format import AlpacaFormatter
from tokenizer import Tokenizer
from torch.utils.data import Dataset
from typing import override


class InstructionDataset(Dataset):

    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self.formatter = AlpacaFormatter()
        self.tokenizer = Tokenizer()

    def __len__(self) -> int:
        return len(self.entries)

    @override
    def __getitem__(self, i) -> list[int]:
        entry = self.entries[i]
        formatted = self.formatter.format(entry.instruction, entry.input, entry.output)
        return self.tokenizer.encode(formatted)
