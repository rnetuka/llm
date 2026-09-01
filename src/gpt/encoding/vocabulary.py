import tiktoken


class Vocabulary:

    def __init__(self):
        self.encoding = tiktoken.get_encoding('gpt2')

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def __getitem__(self, token_id: int) -> str:
        return self.encoding.decode_single_token_bytes(token_id).decode('utf-8', errors='replace')
