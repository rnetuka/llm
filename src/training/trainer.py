import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader


class ModelTrainer:

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    def batch_loss(self, input_batch: Tensor, target_batch: Tensor) -> Tensor:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def train(self, training_data: DataLoader, n_epochs: int = 1):
        self.model.train()
        try:
            for epoch in range(n_epochs):
                for input_batch, target_batch in training_data:
                    self.optimizer.zero_grad()
                    loss = self.batch_loss(input_batch, target_batch)
                    loss.backward()
                    self.optimizer.step()
        finally:
            self.model.to('cpu')
            self.model.eval()

    def evaluate_model(self, validation_data: DataLoader) -> float:
        with torch.no_grad():
            total_loss = 0

            if len(validation_data) == 0:
                return float('NaN')

            for input_batch, target_batch in validation_data:
                total_loss += self.batch_loss(input_batch, target_batch).item()

            return total_loss / len(validation_data)
