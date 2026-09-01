from gpt.config import GPT_2_SMALL
from gpt import GptModel
from pathlib import Path
from typing import override
from unittest import TestCase
from unittest import main as run_tests


class Gpt2SmallTest(TestCase):

    @override
    def setUp(self):
        self.model = GptModel(GPT_2_SMALL)

    def test_name(self):
        name = self.model.name
        expected = 'GPT-2 Small'
        self.assertEqual(expected, name)

    def test_filename(self):
        filename = self.model.filename
        expected = 'gpt2-small'
        self.assertEqual(expected, filename)

    def test_state_file(self):
        path = self.model.state_file.resolve()
        expected = (Path('..') / 'resources' / 'gpt2-small.pth').resolve()
        self.assertEqual(expected, path)


if __name__ == '__main__':
    run_tests()
