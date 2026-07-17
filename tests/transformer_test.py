import torch
import unittest

from transformer import TransformerBlock
from unittest import TestCase


class MyTestCase(TestCase):

    def test_transformer_block_dimension(self):
        x = torch.rand(2, 4, 768)
        block = TransformerBlock()
        output = block(x)
        self.assertEqual(torch.Size([2, 4, 768]), output.shape)  # add assertion here


if __name__ == '__main__':
    unittest.main()
