from unittest import main as run_tests
from unittest import TestCase
from tokenizer import Tokenizer


class Gpt2EncodingTest(TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenize(self):
        text = f'Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.'
        result = self.tokenizer.tokenize(text)
        expected = [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
        self.assertEqual(result, expected)

    def test_decode(self):
        tokens = [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
        result = self.tokenizer.encoding.decode(tokens)
        expected = f'Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.'
        self.assertEqual(result, expected)


if __name__ == '__main__':
    run_tests()