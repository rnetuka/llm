import time

from .data import read_data
from .dataloader import create_dataloader
from gpt import GptModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from training.loss import batch_loss, model_loss


class InstructionTrainer:

    def __init__(self):
        self.data = read_data()
        self.training_portion = int(len(self.data) * 0.85)
        self.test_portion = int(len(self.data) * 0.1)
        self.validation_portion = len(self.data) - self.training_portion - self.test_portion

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
        optimizer = AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

        training_losses = []
        validation_losses = []
        examples_seen = 0
        global_step = -1
        eval_frequency = 5
        eval_iter = 5

        for epoch in range(n_epochs):
            model.train()

            for input_batch, target_batch in self.training_data:
                optimizer.zero_grad()
                loss = batch_loss(model, input_batch, target_batch)
                loss.backward()

                optimizer.step()
                examples_seen += input_batch.shape[0]
                global_step += 1

                if global_step % eval_frequency == 0:
                    model.eval()
                    training_loss, validation_loss = model_loss(model,
                        self.training_data,
                        self.validation_data,
                        eval_iter
                    )
                    model.train()

                    training_losses.append(training_loss)
                    validation_losses.append(validation_loss)

                    print(f'Epoch {epoch + 1} (Step {global_step:06d}): ' +
                          f'Training loss: {training_loss:.3f} ' +
                          f'Validation loss: {validation_loss:.3f}')

        model.eval()
        end_time = time.time()
        print(f'Training completed in {((end_time - start_time) / 60):2f} minutes')
