import torch

from .data import Entry
from .dataset import InstructionDataset
from gpt.config import CONTEXT_LENGTH, PAD_TOKEN
from gpt.encoding import Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Iterable


def collate(batch: Iterable[list[int]]) -> tuple[Tensor, Tensor]:
    tokenizer = Tokenizer()
    eof_token = tokenizer.encode('<|endoftext|>')[0]
    pad_token = PAD_TOKEN

    max_batch_length = max(len(entry) for entry in batch)  # finds the longest sequence in the batch
    inputs = []
    targets = []

    for tokens in batch:  # pads and prepares inputs
        tokens = tokens + [eof_token]
        padded_inputs = tokens + [eof_token] * (max_batch_length + 1 - len(tokens))
        padded_targets = tokens + [pad_token] * (max_batch_length + 1 - len(tokens))

        inputs.append(torch.tensor(padded_inputs[:-1])[:CONTEXT_LENGTH])  # removes extra padded token added earlier
        targets.append(torch.tensor(padded_targets[1:])[:CONTEXT_LENGTH])  # shifts +1 to the right for targets

    return torch.stack(inputs), torch.stack(targets)


def create_dataloader(data: list[Entry], batch_size: int = 8, shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset=InstructionDataset(data),
        collate_fn=collate,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
