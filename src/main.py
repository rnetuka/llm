import torch.nn

from config import CONTEXT_LENGTH
from gpt import GptModel
from llm import Llm
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

    #logits = model(batch)

    #print(f'Output shape: {logits.shape}')
    #print(logits)

    llm = Llm()
    start_context = 'Hello, I am'
    encoded = tokenizer.tokenize(start_context)
    print(f'Encoded: {encoded}')
    encoded_tensor = encoded.tensor.unsqueeze(0)
    print(f'Encoded tensor shape: {encoded_tensor.shape}')

    llm.model.eval()
    out = llm.generate_text(encoded_tensor, max_new_tokens=6, context_size=CONTEXT_LENGTH)
    print(f'Output: {out}')
    print(f'Output length: {len(out[0])}')

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f'Decoded text: {decoded_text}')
