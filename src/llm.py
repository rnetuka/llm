import torch

from gpt import GptModel


class Llm:

    def __init__(self):
        self.model = GptModel()

    def crop_context(self, idx, context_size):
        return idx[:, -context_size:]

    def generate_text(self, idx, max_new_tokens: int, context_size: int):
        for _ in range(max_new_tokens):
            idx_cond = self.crop_context(idx, context_size)
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :]          # last vector, corresponding to the next token
            probas = torch.softmax(logits, dim=-1)  # converts vector into probability distribution
            idx_next = torch.argmax(probas, dim=-1, keepdim=True) # index of value with largest probability
            idx = torch.cat((idx, idx_next), dim=1) # appends sampled index to the running sequence

        return idx
