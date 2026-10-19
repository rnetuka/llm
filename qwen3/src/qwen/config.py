import torch


class QwenConfig:

    vocabulary_size: int
    context_length: int
    embedding_dimension: int

    # number of attention heads
    number_of_heads: int

    number_of_layers: int

    # size of intermediate dimension in FeedForward
    hidden_dimension: int

    # size of the heads in Grouped Query Attention
    head_dimension: int

    # whether to normalize queries & keys in Grouped Query Attention
    qk_normalization: bool

    # number of key-value groups for Grouped Query Attention
    number_of_kv_groups: int

    # the base in RoPE's "theta"
    rope_base: float

    # lower-precision dtype to reduce memory
    dtype: torch.dtype


    def __init__(self):
        self.vocabulary_size = 151_936
        self.context_length = 40_960
        self.embedding_dimension = 1024
        self.number_of_heads = 16
        self.number_of_layers = 28
        self.hidden_dimension = 3072
        self.head_dimension = 128
        self.qk_normalization = True
        self.number_of_kv_groups = 8
        self.rope_base = 1_000_000.
        self.dtype = torch.bfloat16


def get_device(enable_tensor_cores: bool = True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using NVIDIA CUDA GPU')

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split('.')[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
                torch.backends.cudnn.conv.fp32_precision = 'fp32'
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.xpu.is_available():
        device = torch.device('xpu')
        print('Using Intel GPU')

    else:
        device = torch.device('cpu')

    return device
