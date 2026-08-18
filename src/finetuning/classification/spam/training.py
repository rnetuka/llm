import pandas

from .dataloader import create_dataloader
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from pathlib import Path
from urllib import request
from zipfile import ZipFile

from .evaluation import SpamClassifierEvaluator
from .model import SpamClassifier
from torch.optim import AdamW


resource_directory = Path('..') / 'resources' / 'finetuning'


class SmsSpamTrainer:

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
        self.download_data()
        tsv_path = resource_directory / 'SMSSpamCollection.tsv'
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

    def download_data(self):
        data_file_path = Path(resource_directory) / 'SMSSpamCollection.tsv'

        if data_file_path.exists():
            return

        url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'

        if not resource_directory.exists():
            resource_directory.mkdir()

        zip_path = Path(resource_directory) / 'sms_spam_collection.zip'

        with request.urlopen(url) as response:
            with open(zip_path, 'wb') as file:
                file.write(response.read())

        with ZipFile(zip_path, 'r') as zip:
            zip.extractall(resource_directory)

        # rename the unziped content
        Path(resource_directory).joinpath('SMSSpamCollection').rename(data_file_path)

        # clen up the unused files
        Path(resource_directory).joinpath('sms_spam_collection.zip').unlink()
        Path(resource_directory).joinpath('readme').unlink()

    def train_model(self, model: SpamClassifier, n_epochs: int = 5):
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
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
                loss = evaluator.batch_loss(input_batch, target_batch)
                loss.backward()

                optimizer.step()
                examples_seen += input_batch.shape[0]
                global_step += 1

                if global_step % eval_frequency == 0:
                    train_loss, validation_loss = evaluator.model_loss(self.training_data, self.validation_data, eval_iter)
                    model.train()

                    training_losses.append(train_loss)
                    validation_losses.append(validation_loss)

                    print(f'Epoch {epoch + 1} (Step {global_step:06d}): ' +
                          f'Training loss: {train_loss:.3f}' +
                          f'Validation loss: {validation_loss:.3f}')

            training_accuracy = evaluator.model_accuracy(self.training_data, n_batches=eval_iter)
            training_accuracies.append(training_accuracy)
            print(f'Training accuracy: {training_accuracy * 100:.2f}% | ', end='')

            validation_accuracy = evaluator.model_accuracy(self.validation_data, n_batches=eval_iter)
            validation_accuracies.append(validation_accuracy)
            print(f'Validation accuracy: {validation_accuracy * 100:.2f}%')

        model.eval()
