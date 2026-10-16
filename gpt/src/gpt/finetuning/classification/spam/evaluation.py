import torch

from gpt.config import device
from gpt.finetuning.classification.spam.model import SpamClassifier
from torch.utils.data import DataLoader


class SpamClassifierEvaluator:

    def __init__(self, model: SpamClassifier):
        self.model = model

    def model_accuracy(self, test_data: DataLoader, n_batches: int | None = None) -> float:
        self.model.eval()
        correct_predictions = 0
        total_examples = 0

        if n_batches is None:
            n_batches = len(test_data)

        n_batches = min(n_batches, len(test_data))

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
