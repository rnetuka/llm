from config import device, GPT_2_SMALL
from gpt.encoding import Tokenizer, Vocabulary
from gpt import GptModel
from pretraining import openai
from gpt.training.dataloader import create_dataloader
from gpt.training.loss import LossCalculator


if __name__ == '__main__':
    tokenizer = Tokenizer()
    vocabulary = Vocabulary()

    model = GptModel(config=GPT_2_SMALL)
    model.temperature = 1.5
    model.top_k = 50
    model.to(device)
    model.eval()

    print(model.name)
    print(f'Number of parameters: {model.number_of_parameters:_}'.replace('_', ' '))
    print(f'Model size: {model.model_size}')
    print()

    test_data = create_dataloader('../../resources/texts/the-verdict.txt')
    loss_calculator = LossCalculator(model)

    print(f'Vanilla model loss: {loss_calculator.data_loss(test_data):.3f}')

    # Load OpenAI weights instead of training
    openai.pretrain(model)

    # Model loss, calculated with test data. The closer to zero the better
    print(f'Pretrained model loss: {loss_calculator.data_loss(test_data):.3f}')
    print()

    text = 'Every effort moves you'     # starting context for the model
    tokens = tokenizer.tokenize(text)
    output = model.generate_text(tokens, max_new_tokens=30)
    decoded_text = vocabulary.decode(output.tolist())

    print('Output:')
    print(decoded_text)
