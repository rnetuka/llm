import pandas

from .data import download_sms_spam_data, resource_directory
from .dataloader import create_dataloader
from .evaluation import SpamClassifierEvaluator
from .loss import SpamLossCalculator
from .model import SpamClassifier
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from torch.optim import AdamW
from torch.utils.data import DataLoader


class SmsSpamTrainer:

    training_data: DataLoader
    validation_data: DataLoader
    test_data: DataLoader

    def __init__(self):
        self.train_fraction = 0.7       # 70% data is for training
        self.validation_fraction = 0.1  # 10% data is for validation
        self.test_fraction = 0.2        # 20% data is for testing
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.batch_size = 8
        self.num_workers = 0

    def balance_dataset(self, data_frame: DataFrame | TextFileReader) -> DataFrame:
        n_spam = data_frame[data_frame['Label'] == 'spam'].shape[0]
        n_ham = n_spam  # balanced, data is split 50 / 50
        spam_subset = data_frame[data_frame['Label'] == 'spam']
        ham_subset = data_frame[data_frame['Label'] == 'ham'].sample(n_ham)

        result = pandas.concat([ham_subset, spam_subset])
        result['Label'] = result['Label'].map({'ham': 0, 'spam': 1})
        return result

    def prepare_data(self):
        tsv_path = download_sms_spam_data()
        data_frame = pandas.read_csv(tsv_path, sep='\t', header=None, names=['Label', 'Text'])
        data_frame = self.balance_dataset(data_frame)
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)       # shuffle the entire DataFrame

        i = int(len(data_frame) * self.train_fraction)
        j = i + int(len(data_frame) * self.validation_fraction)

        training_data = data_frame[:i]
        validation_data = data_frame[i:j]
        test_data = data_frame[j:]

        training_data.to_csv(resource_directory / 'sms-spam-train.csv', index=False)
        validation_data.to_csv(resource_directory / 'sms-spam-validation.csv', index=False)
        test_data.to_csv(resource_directory / 'sms-spam-test.csv', index=False)

        self.training_data = create_dataloader(resource_directory / 'sms-spam-train.csv', drop_last=True)
        self.validation_data = create_dataloader(resource_directory / 'sms-spam-validation.csv')
        self.test_data = create_dataloader(resource_directory / 'sms-spam-test.csv')

    def train_model(self, model: SpamClassifier, n_epochs: int = 5):
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        loss_calculator = SpamLossCalculator(model)
        evaluator = SpamClassifierEvaluator(model)

        training_losses = []
        validation_losses = []
        training_accuracies = []
        validation_accuracies = []
        examples_seen = 0
        global_step = -1
        eval_frequency = 50
        eval_iter = 5

        for epoch in range(n_epochs):
            model.train()

            for input_batch, target_batch in self.training_data:
                optimizer.zero_grad()
                loss = loss_calculator.batch_loss(input_batch, target_batch)
                loss.backward()

                optimizer.step()
                examples_seen += input_batch.shape[0]
                global_step += 1

                if global_step % eval_frequency == 0:
                    model.eval()
                    train_loss, validation_loss = loss_calculator.model_loss(self.training_data, self.validation_data, eval_iter)
                    model.train()

                    training_losses.append(train_loss)
                    validation_losses.append(validation_loss)

                    print(f'Epoch {epoch + 1} (Step {global_step:06d}): ' +
                          f'Training loss: {train_loss:.3f}' +
                          f'Validation loss: {validation_loss:.3f}')

            model.eval()
            training_accuracy = evaluator.model_accuracy(self.training_data, n_batches=eval_iter)
            training_accuracies.append(training_accuracy)
            print(f'Training accuracy: {training_accuracy * 100:.2f}% | ', end='')

            validation_accuracy = evaluator.model_accuracy(self.validation_data, n_batches=eval_iter)
            validation_accuracies.append(validation_accuracy)
            print(f'Validation accuracy: {validation_accuracy * 100:.2f}%')
            model.train()

        model.eval()
