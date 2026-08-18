import textwrap
from typing import override

from src.finetuning.instructions.format import AlpacaFormatter
from unittest import main as run_tests
from unittest import TestCase


class AlpacaFormatTest(TestCase):

    @override
    def setUp(self):
        self.formatter = AlpacaFormatter()

    def test_full_entry(self):
        text = self.formatter.format(
            'Identify the correct spelling of the following word.',
            'Ocassion',
            'The correct spelling is "Occasion".'
        )
        expected = textwrap.dedent(f'''\
            Below is an instruction that describes a task. Write a response that appropriately completes the request. 

            ### Instruction:
            Identify the correct spelling of the following word.

            ### Input:
            Ocassion

            ### Response:
            The correct spelling is "Occasion".
        ''')
        self.assertEqual(expected, text)

    def test_format_instructions(self):
        text = self.formatter.format_instructions(
            'Identify the correct spelling of the following word.',
            'Ocassion'
        )
        expected = textwrap.dedent(f'''\
            Below is an instruction that describes a task. Write a response that appropriately completes the request. 
            
            ### Instruction:
            Identify the correct spelling of the following word.
            
            ### Input:
            Ocassion
        ''')
        self.assertEqual(expected, text)

    def test_format_instructions_without_input(self):
        text = self.formatter.format_instructions('What is an antonym of "complicated"?')
        expected = textwrap.dedent(f'''\
            Below is an instruction that describes a task. Write a response that appropriately completes the request. 
            
            ### Instruction:
            What is an antonym of "complicated"?
        ''')
        self.assertEqual(expected, text)

    def test_format_output(self):
        text = self.formatter.format_output('An antonym of "complicated" is "simple".')
        expected = textwrap.dedent(f'''\
            ### Response:
            An antonym of "complicated" is "simple".
        ''')
        self.assertEqual(expected, text)


if __name__ == '__main__':
    run_tests()
