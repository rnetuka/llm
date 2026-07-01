import torch.nn

from attention import CasualAttention
from tokenizer import Tokenizer


if __name__ == '__main__':
    text = 'You journey begins with one step'
    embedding_dimensions = 3 # 256

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    length = len(tokens)

    embedding_layer = torch.nn.Embedding(tokenizer.vocabulary_size, embedding_dimensions)
    token_embeddings = embedding_layer(tokens.tensor)

    context_length = length
    context_layer = torch.nn.Embedding(context_length, embedding_dimensions)
    position_embeddings = context_layer(torch.arange(context_length))

    input_embeddings = token_embeddings + position_embeddings

    # Self-attention

    torch.manual_seed(123)
    batch = torch.stack((input_embeddings, input_embeddings), dim=0)
    context_length = batch.shape[1]
    attention = CasualAttention(d_in, d_out, context_length, dropout_rate=0.0)
    context_vectors = attention(input_embeddings)
    print(context_vectors.shape)