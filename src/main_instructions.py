# Fine-tuned GPT-2 model for processing instructions
import types

from config import GPT_2_MEDIUM
from copy import copy
from encoding import Tokenizer, Vocabulary
from finetuning.instructions.format import AlpacaFormatter
from finetuning.instructions.training import InstructionTrainer
from gpt import GptModel
from training.loss import data_loss


tokenizer = Tokenizer()
vocabulary = Vocabulary()
formatter = AlpacaFormatter()


# method attached to the GptModel for convenient processing of instructions
def process(model: GptModel, instruction: str, input: str | None = None) -> str:
    input_text = formatter.format_instructions(instruction,input)
    output = model.generate_text(tokenizer.tokenize(input_text), max_new_tokens=35)
    output = vocabulary.decode(output.tolist())
    response = output[len(input_text):].replace('### Response', '').strip()
    if '<|endoftext|>' in response:
        response = response[:response.index('<|endoftext|>')]
    return response


def test_model(model: GptModel, instruction: str, correct_response: str, input: str | None = None, test_suffix: str | int = ''):
    test_name = f'Test {test_suffix}'.strip()
    print(f'{test_name}:')
    print('### Instruction:')
    print(instruction)
    print()
    if input:
        print('### Input:')
        print(input)
        print()
    print('### Correct response:')
    print(correct_response)
    print()
    response: str = model.process(instruction, input)
    print('### Model response:')
    if '\n' in response:
        response = '\n'.join('> ' + line for line in response.splitlines())
    print(response)
    print()


if __name__ == '__main__':
    config = copy(GPT_2_MEDIUM)

    model = GptModel.pretrained(config)
    model.process = types.MethodType(process, model)
    config.model_size += '-instructions'

    print(f'{model.name}, fine-tuned for processing instructions')
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')
    print()

    trainer = InstructionTrainer()

    print(f'Loss before fine-tuning: {data_loss(model, trainer.test_data):.3f}')
    print()

    test_model(model,
        test_suffix='(before fine-tuning)',
        instruction='Convert the active sentence to passive',
        input='"The chef cooks meal every day".',
        correct_response='The meal is cooked by the chef every day.'
    )

    if model.state_file.exists():
        print('Model loaded from state file')
        model.load()
        print()
    else:
        print('Training model, please wait...')
        trainer.train(model)
        model.save()
        print()

    print(f'Validation loss after fine-tuning: {data_loss(model, trainer.test_data)}')
    print()

    test_model(model,
        test_suffix=1,
        instruction='Rewrite the sentence using a simile',
        input='The car is very fast',
        correct_response='The car is fast as lightning'
    )
    print()

    test_model(model,
        test_suffix=2,
        instruction='Convert 125 kilometers to meters',
        correct_response='125 kilometers is 125 000 meters'
    )
    print()

    test_model(model,
        test_suffix=3,
        instruction='Generate a short sentence using the word "miraculous"',
        correct_response='Against all odds, her recovery was nothing short of miraculous.'
    )
