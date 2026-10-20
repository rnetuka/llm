import torch

from pathlib import Path
from qwen.tokenizer import QwenTokenizer
from unittest import main as run_tests
from unittest import TestCase


resources = Path('resources')


class TokenizerTest(TestCase):

    def setUp(self):
        self.tokenizer = QwenTokenizer(resources / 'tokenizer-base.json')

    def test_encode(self):
        text = 'Explain large language models.'
        expected = [840, 20772, 3460, 4128, 4119, 13]
        # 840 = Ex
        # 20772 = plain
        # 3460 = large
        # 4128 = language
        # 4119 = models
        # 13 = .
        result = self.tokenizer.encode(text)
        self.assertEqual(expected, result)

    def test_tokenize(self):
        text = 'Explain large language models.'
        expected = torch.tensor([[840, 20772, 3460, 4128, 4119, 13]])
        result = self.tokenizer.tokenize(text)
        self.assertTrue(torch.equal(result, expected))

    def test_decode(self):
        tokens = [840, 20772, 3460, 4128, 4119, 13]
        expected = 'Explain large language models.'
        result = self.tokenizer.decode(tokens)
        self.assertEqual(expected, result)

    def test_base_tokenizer_eos_token(self):
        tokenizer = QwenTokenizer(resources / 'tokenizer-base.json')
        self.assertEqual('<|endoftext|>', tokenizer.eos_token)

    def test_reasoning_tokenizer_eos_token(self):
        tokenizer = QwenTokenizer(resources / 'tokenizer-reasoning.json')
        self.assertEqual('<|im_end|>', tokenizer.eos_token)


if __name__ == '__main__':
    run_tests()
