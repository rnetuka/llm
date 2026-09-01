import torch


VOCABULARY_SIZE = 50_257
CONTEXT_LENGTH = 1024
PAD_TOKEN = -100            # token for padding (i.e. 'empty token'), -100 is used by default in PyTorch

MODEL_SIZE_SMALL = 'gpt2-small'
MODEL_SIZE_MEDIUM = 'gpt2-medium'
MODEL_SIZE_LARGE = 'gpt2-large'
MODEL_SIZE_XL = 'gpt2-xl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GptConfig:

    vocabulary_size: int
    context_length: int
    embedding_dimensions: int
    attention_heads: int        # number of attention heads in the multi-head attention mechanism
    n_layers: int               # number of transformer blocks
    drop_rate: float
    qkv_bias: bool

    def __init__(self, model_name: str, model_size: str, embedding_dimensions: int, attention_heads: int, n_layers: int):
        self.model_name = model_name
        self.model_size = model_size
        self.vocabulary_size = VOCABULARY_SIZE
        self.context_length = CONTEXT_LENGTH
        self.embedding_dimensions = embedding_dimensions
        self.attention_heads = attention_heads
        self.n_layers = n_layers
        self.drop_rate = 0.1
        self.qkv_bias = True


GPT_2_SMALL = GptConfig(
    model_name='GPT-2 Small',
    model_size=MODEL_SIZE_SMALL,
    embedding_dimensions = 768,
    attention_heads = 12,
    n_layers = 12
)

GPT_2_MEDIUM = GptConfig(
    model_name='GPT-2 Medium',
    model_size=MODEL_SIZE_MEDIUM,
    embedding_dimensions=1024,
    attention_heads=16,
    n_layers=24
)

GPT_2_LARGE = GptConfig(
    model_name='GPT-2 Large',
    model_size=MODEL_SIZE_LARGE,
    embedding_dimensions=1280,
    attention_heads=20,
    n_layers=36
)

GPT_2_XL = GptConfig(
    model_name='GPT-2 XL',
    model_size=MODEL_SIZE_XL,
    embedding_dimensions=1600,
    attention_heads=25,
    n_layers=48
)
