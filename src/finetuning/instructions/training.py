import math
import time
import torch.nn as nn

from .data import Entry, read_data
from .dataloader import create_dataloader
from gpt import GptModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from training.loss import LossCalculator


class InstructionTrainer:

    data: list[Entry]               # data for training
    training_ratio: float           # percentage of data to be used for training
    test_ratio: float               # percentage of data to be used for testing

    def __init__(self):
        self.data = read_data()
        self.training_ratio = 0.85
        self.test_ratio = 0.1
        self.warmup = True
        self.gradient_clipping = True

    @property
    def training_portion(self) -> int:
        return int(len(self.data) * self.training_ratio)

    @property
    def test_portion(self) -> int:
        return int(len(self.data) * self.test_ratio)

    @property
    def validation_portion(self) -> int:
        return len(self.data) - self.training_portion - self.test_portion

    @property
    def training_data(self) -> DataLoader:
        i = self.training_portion
        return create_dataloader(self.data[:i], batch_size=1, shuffle=True, drop_last=True)

    @property
    def test_data(self) -> DataLoader:
        i = self.training_portion
        j = self.training_portion + self.test_portion
        return create_dataloader(self.data[i:j], batch_size=1, shuffle=False, drop_last=False)

    @property
    def validation_data(self) -> DataLoader:
        j = self.training_portion + self.test_portion
        return create_dataloader(self.data[j:], batch_size=1, shuffle=False, drop_last=False)

    def train(self, model: GptModel, n_epochs: int = 2):
        start_time = time.time()
        optimizer = AdamW(model.parameters(), weight_decay=0.1)
        loss_calculator = LossCalculator(model)

        training_losses = []
        validation_losses = []
        track_lrs = []
        track_examples_seen = []
        examples_seen = 0
        global_step = -1
        eval_frequency = 5
        eval_iter = 5

        initial_lr = 0.00003
        min_lr = 1e-6
        peak_lr = optimizer.param_groups[0]['lr']
        total_steps = n_epochs * len(self.training_data)
        warmup_steps = int(0.2 * total_steps)
        lr_increment = (peak_lr - initial_lr) / warmup_steps

        for epoch in range(n_epochs):
            model.train()

            for input_batch, target_batch in self.training_data:
                optimizer.zero_grad()
                global_step += 1

                if self.warmup:
                    if global_step < warmup_steps:
                        lr = initial_lr + (global_step * lr_increment)
                    else:
                        progress = ((global_step - warmup_steps) / (total_steps - warmup_steps))
                        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    track_lrs.append(lr)

                loss = loss_calculator.batch_loss(input_batch, target_batch)
                loss.backward()

                if self.warmup and self.gradient_clipping:
                    if global_step >= warmup_steps:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                examples_seen += input_batch.shape[0]

                if global_step % eval_frequency == 0:
                    model.eval()
                    training_loss, validation_loss = loss_calculator.model_loss(
                        self.training_data,
                        self.validation_data,
                        eval_iter
                    )
                    model.train()

                    training_losses.append(training_loss)
                    validation_losses.append(validation_loss)
                    track_examples_seen.append(examples_seen)

                    print(f'Epoch {epoch + 1} (Step {global_step:06d}): ' +
                          f'Training loss: {training_loss:.3f} ' +
                          f'Validation loss: {validation_loss:.3f}')

        model.eval()
        end_time = time.time()

        print(f'Training completed in {((end_time - start_time) / 60):2f} minutes\n')
