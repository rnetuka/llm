import textwrap

from .data import Entry
from typing import override, Protocol


class DataFormatter(Protocol):

    def format(self, instruction: str, input: str | None = None, output: str | None = None) -> str:
        ...

    def format_instructions(self, instruction: str, input: str | None = None) -> str:
        ...

    def format_output(self, output: str) -> str:
        ...


class AlpacaFormatter(DataFormatter):

    @override
    def format(self, instruction: str, input: str | None = None, output: str | None = None) -> str:
        text = self.format_instructions(instruction, input)
        if output:
            text += '\n'
            text += self.format_output(output)
        return text

    @override
    def format_instructions(self, instruction: str, input: str | None = None) -> str:
        text = textwrap.dedent(f'''\
            Below is an instruction that describes a task. Write a response that appropriately completes the request. 

            ### Instruction:
            {instruction}
        ''')
        if input:
            text += textwrap.dedent(f'''\

                ### Input:
                {input}
            ''')
        return text

    @override
    def format_output(self, output: str) -> str:
        return textwrap.dedent(f'''\
            ### Response:
            {output}
        ''')


def alpaca(entry: Entry) -> str:
    text = textwrap.dedent(f'''\
        Below is an instruction that describes a task. Write a response that appropriately completes the request. 

        ### Instruction:
        {entry.instruction}
    ''')
    if entry.input:
        text += textwrap.dedent(f'''\
        
            ### Input:
            {entry.input}
        ''')
    if entry.output:
        text += textwrap.dedent(f'''\
            
            ### Response:
            {entry.output}
        ''')
    return text


def phi3(entry: Entry) -> str:
    text = textwrap.dedent(f'''\
        <|user|>
        {entry.instruction}
    ''')
    if entry.instruction:
        text += f"'{entry.input}'"
    if entry.output:
        text += textwrap.dedent(f'''\
            
            <|assistant|>
            {entry.output}
        ''')
    return text
