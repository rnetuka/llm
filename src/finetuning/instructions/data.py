import json

from pathlib import Path
from typing import TypedDict, NotRequired
from urllib import request


resource_directory = Path('..') / 'resources' / 'finetuning'


class EntryDict(TypedDict):
    instruction: str
    input: NotRequired[str]
    output: str


class Entry:

    instruction: str        # instruction that the LLM is supposed to learn

    input: str | None       # input for the instruction (optional)
                            # empty, if it's already part of the instruction text

    output: str | None      # expected LLM output

    def __init__(self, instruction: str, input: str | None = None, output: str | None = None):
        self.instruction = instruction
        self.input = input
        self.output = output

    @staticmethod
    def from_dict(dict: EntryDict):
        return Entry(dict['instruction'], dict['input'], dict['output'])


def download_data(url: str):
    filename = url.split('/')[-1]
    path = resource_directory / filename

    if path.exists():
        return

    with request.urlopen(url) as response:
        data = response.read().decode('utf-8')

    with open(path, 'w') as file:
        file.write(data)


def read_data() -> list[Entry]:
    path = resource_directory / 'instruction-data.json'

    if not path.exists():
        download_data('https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json')

    with open(path) as file:
        return [Entry.from_dict(dict) for dict in json.load(file)]
