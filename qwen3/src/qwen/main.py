from pathlib import Path

import torch

from qwen.config import QwenConfig
from qwen.model import Qwen3
from qwen.pretraining import download_qwen3_small
from qwen.resources import resources
from qwen.tokenizer import QwenTokenizer


if __name__ == '__main__':
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    #download_qwen3_small(kind='base', tokenizer_only=False, out_dir='qwen3')

    model_file = resources / 'qwen3-0.6B-base.pth'

    tokenizer = QwenTokenizer.base()
    model = Qwen3.pretrained()
    model.to(device)
