import time
import torch

from qwen.config import get_device, QwenConfig
from qwen.model import Qwen3
from qwen.pretraining import download_qwen3_small
from qwen.tokenizer import QwenTokenizer


def generate_stats(generated_tokens: torch.Tensor, start_time: float, end_time: float):
    total_time = end_time - start_time
    print(f'\n\nTime: {total_time:.2f} sec')
    print(f'{int(generated_tokens.numel() / total_time)} tokens/sec')


if __name__ == '__main__':
    device = get_device()

    #download_qwen3_small(kind='base', tokenizer_only=False, out_dir='qwen3')

    tokenizer = QwenTokenizer.base()
    model = Qwen3.pretrained()
    model.to(device)

    prompt = 'Explain large language models in a single sentence.'

    print('User prompt:')
    print(prompt)
    print()
    print('LLM response:')

    tokens = tokenizer.tokenize(prompt)

    start_time = time.time()
    token_count = 0

    for i, token in enumerate(model.generate_text_basic_stream(tokens, max_new_tokens=100, eos_token_id=151643)):
        token_id = token.squeeze(0).tolist()
        token_str = tokenizer.decode(token_id)
        if i == 0:
            token_str = token_str.lstrip()  # the first output token has leading whitespace
        print(token_str, end='', flush=True)
        next_token_id = token.squeeze(0)
        token_count += 1

    end_time = time.time()
    total_time = end_time - start_time

    print('\n\n')
    print(f'Time: {total_time:.2f} sec')
    print(f'{int(token_count // total_time)} tokens/sec')
