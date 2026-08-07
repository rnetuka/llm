import torch

from src.transformer import TransformerBlock
from unittest import main as run_tests
from unittest import TestCase


class TransformerBlockTest(TestCase):

    def test_output_shape(self):
        x = torch.rand(2, 4, 768)
        block = TransformerBlock()
        output = block(x)
        self.assertEqual(torch.Size([2, 4, 768]), output.shape)


if __name__ == '__main__':
    run_tests()
