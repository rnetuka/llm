import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader


class SpamClassifierEvaluator:

    def __init__(self, model: nn.Module):
        self.model = model

    def batch_loss(self, input_batch: Tensor, target_batch: Tensor) -> Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        logits = self.model(input_batch)
        logits = logits[:, -1, :]   # consider only the last output token

        loss = nn.functional.cross_entropy(logits, target_batch)
        return loss

    def data_loss(self, data: DataLoader, n_batches: int | None = None) -> float:
        total_loss = 0

        if len(data) == 0:
            return float('nan')

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
        self.model.eval()
        with torch.no_grad():
            training_loss = self.data_loss(training_data, n_batches=eval_iter)
            validation_loss = self.data_loss(validation_data, n_batches=eval_iter)
        return training_loss, validation_loss

    def model_accuracy(self, test_data: DataLoader, n_batches: int | None = None) -> float:
        self.model.eval()
        correct_predictions = 0
        total_examples = 0

        if n_batches is None:
            n_batches = len(test_data)

        n_batches = min(n_batches, len(test_data))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, (input_batch, target_batch) in enumerate(test_data):
            if i >= n_batches:
                break

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = self.model(input_batch)
                logits = logits[:, -1, :]

            predicted_labels = torch.argmax(logits, dim=-1)
            total_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        return correct_predictions / total_examples
