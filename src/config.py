class GptConfig:

    vocabulary_size: int
    context_length: int
    embedding_dimensions: int

    def __init__(self, vocabulary_size: int, context_lenght: int, embedding_dimensions: int, attention_heads: int, n_layers: int, drop_rate: float, qkv_bias: bool):
        self.vocabulary_size = vocabulary_size
        self.context_length = context_lenght
        self.embedding_dimensions = embedding_dimensions
        self.attention_heads = attention_heads
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias


GPT_2_SMALL = GptConfig(
    vocabulary_size = 50_257,
    context_lenght = 1024,
    embedding_dimensions = 768,
    attention_heads = 12,
    n_layers = 12,
    drop_rate = 0.1,
    qkv_bias = False
)

GPT_2_MEDIUM = GptConfig(
    vocabulary_size=50_257,
    context_lenght=1024,
    embedding_dimensions=1024,
    attention_heads=16,
    n_layers=24,
    drop_rate=0.1,
    qkv_bias=False
)

GPT_2_LARGE = GptConfig(
    vocabulary_size=50_257,
    context_lenght=1024,
    embedding_dimensions=1280,
    attention_heads=20,
    n_layers=36,
    drop_rate=0.1,
    qkv_bias=False
)

GPT_2_XL = GptConfig(
    vocabulary_size=50_257,
    context_lenght=1024,
    embedding_dimensions=1600,
    attention_heads=25,
    n_layers=48,
    drop_rate=0.1,
    qkv_bias=False
)

VOCABULARY_SIZE = 50_257
CONTEXT_LENGTH = 1024
EMBEDDING_DIMENSIONS = 768
ATTENTION_HEADS = 12
N_LAYERS = 12
DROP_RATE = 0.1
QKV_BIAS = False