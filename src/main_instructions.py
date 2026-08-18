# Fine-tuned GPT-2 model for processing instructions
import types

from config import GPT_2_MEDIUM, GPT_2_SMALL
from copy import copy
from finetuning.instructions.format import AlpacaFormatter
from finetuning.instructions.training import InstructionTrainer
from gpt import GptModel
from tokenizer import Tokenizer
from training.loss import data_loss
from vocabulary import Vocabulary


tokenizer = Tokenizer()
vocabulary = Vocabulary()
formatter = AlpacaFormatter()


def process(model: GptModel, instruction: str, input: str | None = None) -> str:
    input_text = formatter.format_instructions(instruction,input)
    output = model.generate_text(tokenizer.tokenize(input_text), max_new_tokens=35)
    output = vocabulary.decode(output.tolist())
    response = output[len(input_text):].replace('### Response', '').strip()
    if '<|endoftext|>' in response:
        response = response[:response.index('<|endoftext|>')]
    return response


if __name__ == '__main__':
    config = copy(GPT_2_SMALL)

    model = GptModel.pretrained(config)
    model.process = types.MethodType(process, model)
    config.model_size += '-instructions'

    print(f'{model.name}, fine-tuned for processing instructions')
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')
    print()

    trainer = InstructionTrainer()

    print(f'Validation loss before fine-tuning: {data_loss(model, trainer.test_data)}')
    print()

    print('Test (before fine-tuning):')
    instruction = 'Convert the active sentence to passive'
    input = '"The chef cooks meal every day"'
    print('### Instruction:')
    print(instruction)
    print()
    print('### Input:')
    print(input)
    print()
    print('### Correct response:')
    print('The meal is cooked by the chef every day')
    print()
    response = model.process(instruction, input)
    print('### Model response:')
    print(response)
    print()

    if model.state_file.exists():
        print('Model loaded from state file')
        model.load()
    else:
        print('Training model, please wait...')
        trainer.train(model)
        model.save()

    print()
    print(f'Validation loss after fine-tuning: {data_loss(model, trainer.test_data)}')

    print('Test 1')

    input_text = formatter.format_instructions(
        'Rewrite the sentence using a simile',
        'The car is very fast'
    )
    print(input_text)
    print('Correct response:')
    print('  The car is fast as lightning')
    output = model.generate_text(tokenizer.tokenize(input_text), max_new_tokens=35)
    output = vocabulary.decode(output.tolist())
    response = output[len(input_text):].replace('### Response', '').strip()
    print('Model response:')
    print(f'  {response}')

    print('Test 2')

    input_text = formatter.format_instructions('What type of cloud is typically associated with thunderstorms?')
    print(input_text)
    print('Correct response:')
    print('  The type of cloud typically associated with thunderstorms is cumulonimbus')
    output = model.generate_text(tokenizer.tokenize(input_text), max_new_tokens=35)
    output = vocabulary.decode(output.tolist())
    response = output[len(input_text):].replace('### Response', '').strip()
    print('Model response:')
    print(f'  {response}')

    print('Test 3')

    input_text = formatter.format_instructions('Name the author of "Pride and Prejudice".')
    print(input_text)
    print('Correct response:')
    print('  Jane Austen')
    output = model.generate_text(tokenizer.tokenize(input_text), max_new_tokens=35)
    output = vocabulary.decode(output.tolist())
    response = output[len(input_text):].replace('### Response', '').strip()
    print('Model response:')
    print(f'  {response}')
