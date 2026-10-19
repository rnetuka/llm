import torch
import torch.nn as nn

from qwen.cache import KVCache
from qwen.config import QwenConfig
from qwen.normalization import RMSNorm
from qwen.resources import resources
from qwen.rope import compute_rope_params
from qwen.transformer import TransformerBlock


class Qwen3(nn.Module):

    def __init__(self, config: QwenConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.embedding_dimension, dtype=config.dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.number_of_layers)
        ])
        self.final_normalization = RMSNorm(config.embedding_dimension)
        self.output_head = nn.Linear(
            config.embedding_dimension,
            config.vocabulary_size,
            bias=False,
            dtype=config.dtype
        )

        if config.head_dimension is None:
            head_dimension = config.embedding_dimension // config.number_of_heads
        else:
            head_dimension = config.head_dimension
        cos, sin = compute_rope_params(
            head_dim=head_dimension,
            theta_base=config.rope_base,
            context_length=config.context_length
        )
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        self.config = config
        self.kv_cache = KVCache(config.number_of_layers)
        self.current_position = 0   # track current position in KV cache

    @staticmethod
    def pretrained():
        model_file = resources / 'qwen3-0.6B-base.pth'
        model = Qwen3(QwenConfig())

        checkpoint = torch.load(model_file, map_location='cpu')

        # If the weights are wrapped inside a key like 'state_dict' or 'model', extract them:
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # 2. Create a new state dict with renamed keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('tok_emb.'):
                new_key = key.replace('tok_emb.', 'token_embedding.')
            elif key.startswith('trf_blocks.'):
                new_key = (key
                   .replace('trf_blocks.', 'transformer_blocks.')
                   .replace('.att.', '.attention.')
                   .replace('.out_proj.', '.out_projection.')
                   .replace('.q_norm.', '.q_normalization.')
                   .replace('.k_norm.', '.k_normalization.')
                   .replace('.fc1.', '.layer1.')
                   .replace('.fc2.', '.layer2.')
                   .replace('.fc3.', '.layer3.')
                )
            elif key.startswith('final_norm.'):
                new_key = key.replace('final_norm.', 'final_normalization.')
            elif key.startswith('out_head.'):
                new_key = key.replace('out_head.', 'output_head.')

            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        return model


    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        token_embedding = self.token_embedding(in_idx)
        x = token_embedding

        num_tokens = x.shape[1]
        cache = self.kv_cache
        if cache is not None:
            pos_start = self.current_position
            pos_end = pos_start + num_tokens
            self.current_position = pos_end
            mask = torch.triu(torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1)[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        # Prefill (no cache): mask starts as (num_tokens, num_tokens)
        # Cached decoding: mask starts as (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # We add two leading dimensions so the mask becomes
        # (1, 1, num_tokens, num_tokens) during prefill and
        # (1, 1, num_tokens, total_key_tokens) during cached decoding.
        # These extra dimensions let PyTorch broadcast the same mask
        # across all batches and attention heads when applying it to
        # attn_scores of shape (batch, num_heads, num_tokens, total_key_tokens).
        mask = mask[None, None, :, :]

        for i, block in enumerate(self.transformer_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_position=pos_start, cache=blk_cache)

            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_normalization(x)
        logits = self.output_head(x.to(self.config.dtype))
        return logits

    def reset_kv_cache(self):
        self.kv_cache = KVCache(self.config.number_of_layers)
        self.current_position = 0

    @torch.inference_mode()
    def generate_text_basic_stream(self, tokens: torch.Tensor, max_new_tokens: int, eos_token_id: int) -> torch.Tensor:
        self.eval()
        self.reset_kv_cache()
        out = self(tokens)[:, -1]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if torch.all(next_token == eos_token_id):
                break

            yield next_token
            out = self(next_token)[:, -1]
            #tokens = torch.cat([tokens, next_token], dim=1)
