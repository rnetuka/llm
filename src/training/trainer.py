import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from training.evaluator import ModelEvaluator


class ModelTrainer:

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        self.evaluator = ModelEvaluator(model)

    def train(self, training_data: DataLoader, n_epochs: int = 1):
        self.model.train()
        try:
            for epoch in range(n_epochs):
                for input_batch, target_batch in training_data:
                    self.optimizer.zero_grad()
                    loss = self.evaluator.batch_loss(input_batch, target_batch)
                    loss.backward()
                    self.optimizer.step()
        finally:
            self.model.to('cpu')
            self.model.eval()
