import tiktoken

from tokens import Tokens


class Tokenizer:

    def __init__(self):
        self.encoding = tiktoken.get_encoding('gpt2')

    @property
    def vocabulary_size(self) -> int:
        return 50_257

    def __decode(self, token_id: int) -> str:
        return self.encoding.decode_single_token_bytes(token_id).decode('utf-8', errors='replace')

    def tokenize(self, text: str) -> Tokens:
        token_ids = self.encoding.encode(text, allowed_special={'<|endoftext|>'})
        tokens = [self.__decode(token_id) for token_id in token_ids]
        return Tokens(tokens, token_ids)
