import torch.nn

from training.trainer import ModelTrainer
from vocabulary import Vocabulary
from config import CONTEXT_LENGTH
from gpt import GptModel
from tokenizer import Tokenizer
from training.dataloader import dataloaders


if __name__ == '__main__':
    text1 = 'Every effort moves you'
    text2 = 'Every day holds a'

    tokenizer = Tokenizer()
    vocabulary = Vocabulary()

    model = GptModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print('GPT-2 small+')
    print(f'Total number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Total size of the model: {model.model_size}')

    start_context = 'Hello, I am'
    tokens = tokenizer.tokenize(start_context)

    output = model.generate_text(tokens, max_new_tokens=6, context_size=CONTEXT_LENGTH)
    print(f'Output: {output}')
    print(f'Output length: {len(output[0])}')

    decoded_text = vocabulary.decode(output.squeeze(0).tolist())
    print(f'Decoded text: {decoded_text}')

    with open('../resources/texts/the-verdict.txt') as file:
        text_data = file.read()

    print(f'Training text characters: {len(text_data)}')
    print(f'Training text tokens: {tokenizer.tokenize(text_data).shape[1]}')

    trainer = ModelTrainer(model)
    train_loader, validation_loader = dataloaders()

    print('Before training')
    print(f'Train Loss: {trainer.evaluate_model(train_loader):.3f}')
    print(f'Validation Loss: {trainer.evaluate_model(validation_loader):.3f}')

    print('Training the model (takes around 5 minutes)')
    trainer.train(train_loader, 10)

    print('After training')
    print(f'Train Loss: {trainer.evaluate_model(train_loader):.3f}')
    print(f'Validation Loss: {trainer.evaluate_model(validation_loader):.3f}')

    output = model.generate_text(tokens, max_new_tokens=6, context_size=CONTEXT_LENGTH)
    decoded_text = vocabulary.decode(output.squeeze(0).tolist())
    print(f'Output: {decoded_text}')
