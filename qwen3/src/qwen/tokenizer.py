from __future__ import annotations

import re
import torch

from pathlib import Path
from qwen import resources
from qwen.config import get_device
from qwen.resources import resources
from tokenizers import Tokenizer


class QwenTokenizer:

    SPECIALS = [
        '<|endoftext|>',
        '<|im_start|>',
        '<|im_end|>',
        '<|object_ref_start|>',
        '<|object_ref_end|>',
        '<|box_start|>',
        '<|box_end|>',
        '<|quad_start|>',
        '<|quad_end|>',
        '<|vision_start|>',
        '<|vision_end|>',
        '<|vision_pad|>',
        '<|image_pad|>',
        '<|video_pad|>'
    ]

    SPLIT_REGEX = re.compile(r'(<\|[^>]+?\|>)')


    def __init__(self, path: Path, apply_chat_template: bool = False, add_generation_prompt: bool = False, add_thinking: bool = False):
        if not path.exists():
            raise FileNotFoundError(f'Tokenizer file ${path} does not exist')

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        self.impl = Tokenizer.from_file(str(path))
        self.special_tokens = {token: self.impl.token_to_id(token) for token in self.SPECIALS}

        self.pad_token = '<|endoftext|>'
        self.eos_token = '<|im_end|>' if 'reasoning' in path.name.lower() else '<|endoftext|>'

    @staticmethod
    def base() -> QwenTokenizer:
        return QwenTokenizer(resources / 'tokenizer-base.json')

    @staticmethod
    def reasoning() -> QwenTokenizer:
        return QwenTokenizer(
            resources / 'tokenizer-reasoning.json',
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True
        )

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens[self.eos_token]

    def split(self, prompt: str) -> list[str]:
        return [part for part in self.SPLIT_REGEX.split(prompt) if part]

    def encode(self, prompt: str) -> list[int]:
        if (stripped := prompt.strip()) in self.special_tokens and '\n' not in stripped:
            return [self.special_tokens[stripped]]

        if self.apply_chat_template:
            prompt = self._wrap_chat(prompt)

        ids = []
        for sequence in [part for part in self.SPLIT_REGEX.split(prompt) if part]:
            if sequence in self.special_tokens:
                ids.append(self.special_tokens[sequence])
            else:
                ids += self.impl.encode(sequence).ids

        return ids

    def tokenize(self, prompt: str) -> torch.Tensor:
        token_ids = self.encode(prompt)
        return torch.tensor(token_ids, device=get_device()).unsqueeze(0)

    def decode(self, token_ids: list[int]) -> str:
        return self.impl.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_message: str):
        s = f'<|im_start|>user\n{user_message}\n<|im_end|>\n'
        if self.add_generation_prompt:
            s += '<|im_start|>assistant'
            if self.add_thinking:
                s += '\n'
            else:
                s += '\n<think>\n\n</think>\n\n'
        return s
