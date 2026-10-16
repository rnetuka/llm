import torch

from gpt.config import CONTEXT_LENGTH, PAD_TOKEN
from gpt.encoding import Tokenizer
from gpt.finetuning.instructions.dataloader import collate
from unittest import main as run_tests
from util.pytorch_test import PyTorchTest


class InstructionDataloaderTest(PyTorchTest):

    def test_collate_inputs(self):
        input_1 = [0, 1, 2, 3, 4]
        input_2 = [5, 6]
        input_3 = [7, 8, 9]
        batch = (input_1, input_2, input_3)
        tokenizer = Tokenizer()
        eof = tokenizer.encode('<|endoftext|>')[0]
        pad = PAD_TOKEN
        inputs, _ = collate(batch)
        expected = torch.tensor(
            [[0, 1, 2, 3, 4],
             [5, 6, eof, eof, eof],
             [7, 8, 9, eof, eof]]
        )
        self.assertTensorsEqual(inputs, expected)

    def test_collate_targets(self):
        input_1 = [0, 1, 2, 3, 4]
        input_2 = [5, 6]
        input_3 = [7, 8, 9]
        batch = (input_1, input_2, input_3)
        tokenizer = Tokenizer()
        eof = tokenizer.encode('<|endoftext|>')[0]
        pad = PAD_TOKEN
        _, targets = collate(batch)
        expected = torch.tensor(
            [[1, 2, 3, 4, eof],
             [6, eof, pad, pad, pad],
             [8, 9, eof, pad, pad]]
        )
        self.assertTensorsEqual(targets, expected)

    def test_collate_truncate_inputs(self):
        input = [i for i in range(3000)]
        batch = [input]
        inputs, _ = collate(batch)
        self.assertEqual(CONTEXT_LENGTH, inputs.shape[-1])

    def test_collate_truncate_targets(self):
        input = [i for i in range(3000)]
        batch = [input]
        _, targets = collate(batch)
        self.assertEqual(CONTEXT_LENGTH, targets.shape[-1])


if __name__ == "__main__":
    run_tests()
