import torch
import unittest

from gpt import FeedForward
from unittest import TestCase


class MyTestCase(TestCase):

    def test_feed_forward_dimensions(self):
        ffn = FeedForward(emb_dim=768)
        x = torch.randn(2, 3, 768)
        out = ffn(x)
        self.assertEqual(torch.Size([2, 3, 768]), out.shape)


if __name__ == '__main__':
    unittest.main()
