import torch
import torch.nn as nn

from config import device, PAD_TOKEN
from gpt import GptModel
from torch import Tensor
from torch.utils.data import DataLoader


class LossCalculator:

    def __init__(self, model: GptModel):
        self.model = model

    def batch_loss(self, input_batch: Tensor, target_batch: Tensor) -> Tensor:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        logits = self.model(input_batch)

        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten(), ignore_index=PAD_TOKEN)
        return loss

    def data_loss(self, data: DataLoader, n_batches: int | None = None) -> float:
        total_loss = 0

        if len(data) == 0:
            return float('NaN')

        if n_batches is None:
            n_batches = len(data)

        n_batches = min(n_batches, len(data))

        for i, (input_batch, target_batch) in enumerate(data):
            if i >= n_batches:
                break

            loss = self.batch_loss(input_batch, target_batch)
            total_loss += loss.item()

        return total_loss / n_batches

    def model_loss(self, training_data: DataLoader, validation_data: DataLoader, eval_iter: int) -> tuple[float, float]:
        with torch.no_grad():
            return (
                self.data_loss(training_data, n_batches=eval_iter),
                self.data_loss(validation_data, n_batches=eval_iter)
            )
