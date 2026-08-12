import os
import torch.nn

from gpt import GptModel
from pretraining import openai
from pretraining.openai import GPT_2_SMALL
from tokenizer import Tokenizer
from training.dataloader import create_dataloader
from training.evaluator import ModelEvaluator
from vocabulary import Vocabulary


if __name__ == '__main__':
    tokenizer = Tokenizer()
    vocabulary = Vocabulary()

    model = GptModel()
    model.temperature = 1.5
    model.top_k = 50
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    print('GPT-2 small (custom)')
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')

    evaluator = ModelEvaluator(model)
    validation_data = create_dataloader('../resources/texts/the-verdict.txt')

    print(f'Vanilla model validation Loss: {evaluator.evaluate_model(validation_data):.3f}')

    # Load OpenAI weights instead of training
    openai.pretrain(model)

    print(f'Trained model validation Loss: {evaluator.evaluate_model(validation_data):.3f}')

    tokens = tokenizer.tokenize('Every effort moves you')
    output = model.generate_text(tokens, max_new_tokens=30)
    decoded_text = vocabulary.decode(output.tolist())

    print()
    print('Output:')
    print(decoded_text)
