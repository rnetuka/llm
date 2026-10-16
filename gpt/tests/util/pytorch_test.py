import torch

from torch import Tensor
from unittest import TestCase


class PyTorchTest(TestCase):

    def assertTensorsEqual(self, a: Tensor, b: Tensor):
        self.assertTrue(torch.equal(a, b))
