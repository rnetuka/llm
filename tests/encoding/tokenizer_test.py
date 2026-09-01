from gpt.encoding import Tokenizer
from typing import override
from unittest import main as run_tests
from unittest import TestCase


class TokenizerTest(TestCase):

    @override
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.text = 'In the realm of visual compositions, where content reigns supreme, there exists a tranquil harbor ' \
               + 'known as Placeholder Bay. Here, amidst the gentle sway of design elements, words meander without true purpose, ' \
               + 'yet with a semblance of meaning. This text, neither too captivating nor too bland, serves as a beacon for eyes ' \
               + 'that scan, seeking the eventual substance that will fill the void. As the sun casts'

    def test_split(self):
        tokens = self.tokenizer.split(self.text)
        expected = [
            'In', ' the', ' realm', ' of', ' visual', ' compositions', ',',
            ' where', ' content', ' reign', 's', ' supreme', ',',
            ' there', ' exists', ' a', ' tranquil', ' harbor', ' known', ' as', ' Place', 'holder', ' Bay', '.',
            ' Here', ',', ' amidst', ' the', ' gentle', ' sway', ' of', ' design', ' elements', ',',
            ' words', ' me', 'ander', ' without', ' true', ' purpose', ',',
            ' yet', ' with', ' a', ' semblance', ' of', ' meaning', '.',
            ' This', ' text', ',', ' neither', ' too', ' capt', 'ivating', ' nor', ' too', ' bland', ',',
            ' serves', ' as', ' a', ' beacon', ' for', ' eyes', ' that', ' scan', ',',
            ' seeking', ' the', ' eventual', ' substance', ' that', ' will', ' fill', ' the', ' void', '.',
            ' As', ' the', ' sun', ' casts'
        ]
        self.assertEqual(tokens, expected)

    def test_encode(self):
        text = 'In the realm of visual compositions, where content reigns supreme, there exists a tranquil harbor ' \
               + 'known as Placeholder Bay. Here, amidst the gentle sway of design elements, words meander without true purpose, ' \
               + 'yet with a semblance of meaning. This text, neither too captivating nor too bland, serves as a beacon for eyes ' \
               + 'that scan, seeking the eventual substance that will fill the void. As the sun casts'
        tokens = self.tokenizer.encode(text)
        expected = [
            818, 262, 13360, 286, 5874, 33543, 11,
            810, 2695, 13580, 82, 17700, 11,
            612, 7160, 257, 46944, 25451, 1900, 355, 8474, 13829, 4696, 13,
            3423, 11, 31095, 262, 10296, 20009, 286, 1486, 4847, 11,
            2456, 502, 4066, 1231, 2081, 4007, 11,
            1865, 351, 257, 45960, 286, 3616, 13,
            770, 2420, 11, 6159, 1165, 3144, 39438, 4249, 1165, 34377, 11,
            9179, 355, 257, 34538, 329, 2951, 326, 9367, 11,
            6095, 262, 19657, 9136, 326, 481, 6070, 262, 7951, 13,
            1081, 262, 4252, 26217 # Note: in the book the value 3350 is wrong, 3350 = ' cast', 26217 = ' casts'
        ]
        self.assertEqual(tokens, expected)


if __name__ == '__main__':
    run_tests()
