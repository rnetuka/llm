import torch

from torch.testing import assert_close
from unittest import main as run_tests
from unittest import TestCase
from qwen.normalization import RMSNorm


class RMSNormTest(TestCase):

    def test_normalization(self):
        norm = RMSNorm(6)
        x = torch.tensor([5.61, 14.32, 0., 34.88, 38.7, 11.29])
        y = norm(x)
        expected = torch.tensor([0.25, 0.63, 0., 1.54, 1.71, 0.5])
        assert_close(y, expected, rtol=0, atol=0.01)


if __name__ == '__main__':
    run_tests()
