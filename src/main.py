import torch.nn

from gpt import GptModel
from tokenizer import Tokenizer


if __name__ == '__main__':
    text1 = 'Every effort moves you'
    text2 = 'Every day holds a'

    tokenizer = Tokenizer()

    batch = []
    batch.append(tokenizer.tokenize(text1).tensor)
    batch.append(tokenizer.tokenize(text2).tensor)
    batch = torch.stack(batch, dim=0)

    model = GptModel()

    print('GPT-2 small+')
    print(f'Total number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Total size of the model: {model.model_size}')

    logits = model(batch)

    print(f'Output shape: {logits.shape}')
    print(logits)
