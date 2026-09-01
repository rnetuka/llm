import torch.nn as nn

from .model import SpamClassifier
from gpt.config import device
from torch import Tensor
from gpt.training.loss import LossCalculator
from typing import override


class SpamLossCalculator(LossCalculator):

    def __init__(self, model: SpamClassifier):
        super().__init__(model.gpt)

    @override
    def batch_loss(self, input_batch: Tensor, target_batch: Tensor) -> Tensor:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        logits = self.model(input_batch)
        logits = logits[:, -1, :]  # consider only the last output token

        loss = nn.functional.cross_entropy(logits, target_batch)
        return loss
