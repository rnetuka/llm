import torch
import torch.nn as nn

from qwen.normalization import RMSNorm
from qwen.rope import apply_rope


class GroupedQueryAttention(nn.Module):

    def __init__(self, d_in: int, num_heads: int, num_kv_groups: int, head_dim: int | None = None, qk_norm: bool = False, dtype: torch.dtype | None = None):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_projection = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_normalization = RMSNorm(head_dim)
            self.k_normalization = RMSNorm(head_dim)
        else:
            self.q_normalization = None
            self.k_normalization = None

    def forward(self, x: torch.Tensor, mask, cos, sin, start_pos: int = 0, cache=None) -> torch.Tensor:
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_normalization:
            queries = self.q_normalization(queries)
        if self.k_normalization:
            keys_new = self.k_normalization(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.head_dim ** 0.5, dim=-1)

        context = (attention_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        return self.out_projection(context), next_cache
