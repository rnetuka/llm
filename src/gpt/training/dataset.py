import torch

from gpt.config import CONTEXT_LENGTH
from gpt.encoding import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset


class TrainingDataset(Dataset):

    input_samples: list[Tensor]
    target_samples: list[Tensor]

    def __init__(self, text: str, context_length: int = CONTEXT_LENGTH, stride: int = CONTEXT_LENGTH):
        self.tokenizer = Tokenizer()
        self.input_samples = []
        self.target_samples = []

        token_ids = self.tokenizer.encode(text)

        # For text 'Every effort moves you forward. Every journey begins with first step'
        # length=4, stride=4, the dataset contains:
        # input  0: [Every effort moves you]
        # target 0: [effort moves you forward.]
        # input  1: [Every journey begins with]
        # target 1: [journey begins with first]

        for i in range(0, len(token_ids) - context_length, stride):
            j = i + context_length
            input_chunk = token_ids[i : j]
            target_chunk = token_ids[i + 1 : j + 1]
            self.input_samples.append(torch.tensor(input_chunk))
            self.target_samples.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_samples)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.input_samples[i], self.target_samples[i]
