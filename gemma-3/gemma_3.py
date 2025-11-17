import abc
import dataclasses
import math
import sys
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from transformers import AutoTokenizer, PreTrainedTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMPILE_MODEL = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


@dataclasses.dataclass
class Gemma3_Config(abc.ABC):
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    query_pre_attn_scalar: int
    sliding_window: int
    max_position_embeddings: int
    pad_token_id: int
    eos_token_id: int
    rope_local_base_freq: float
    rope_theta: float
    layer_types: list[str]


@dataclasses.dataclass
class Gemma3_270M_Config(Gemma3_Config):
    vocab_size: int = 262_144
    hidden_size: int = 640
    num_hidden_layers: int = 18
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 2_048
    query_pre_attn_scalar: int = 256
    sliding_window: int = 512
    max_position_embeddings: int = 32_768
    pad_token_id: int = 0
    eos_token_id: int = 106
    rope_local_base_freq: float = 10_000.0
    rope_theta: float = 1_000_000.0
    layer_types: list[str] = dataclasses.field(
        default_factory=lambda: [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
    )


@dataclasses.dataclass
class Gemma3_1B_Config(Gemma3_Config):
    vocab_size: int = 262_144
    hidden_size: int = 1_152
    num_hidden_layers: int = 26
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 6_912
    query_pre_attn_scalar: int = 256
    sliding_window: int = 512
    max_position_embeddings: int = 32_768
    pad_token_id: int = 0
    eos_token_id: int = 106
    rope_local_base_freq: float = 10_000.0
    rope_theta: float = 1_000_000.0
    layer_types: list[str] = dataclasses.field(
        default_factory=lambda: [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ]
    )


class KVCache(object):
    def __init__(
        self,
        num_layers: int,
        layer_types: list[str],
        sliding_window: int,
        device: str | torch.device,
    ):
        self.num_layers = num_layers
        self.device = device
        self.sliding_window = sliding_window
        self.layer_types = layer_types

        self._cache_per_layer: list[dict[str, Tensor]] = [
            dict() for _ in range(num_layers)
        ]

    def get_kv_length(self) -> Tensor:
        """
        Returns:
            kv_len: LongTensor of shape (1,), length of cached key/value tensors.
        """
        kv_len = torch.tensor(
            [
                (
                    self._cache_per_layer[i]["k"].shape[2]
                    if "k" in self._cache_per_layer[i]
                    else 0
                )
                for i in range(self.num_layers)
            ],
            device=self.device,
        )  # (num_layers,)
        return kv_len.max().unsqueeze(0)  # (1,)

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Append new key and value tensors to existing cached tensors.

        Args:
            layer_idx: int, layer index
            k: Tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
            v: Tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
        Returns:
            k: Tensor of shape (batch_size, num_key_value_heads, total_seq_len, head_dim)
            v: Tensor of shape (batch_size, num_key_value_heads, total_seq_len, head_dim)
        """
        is_sliding = self.layer_types[layer_idx] == "sliding_attention"
        lcache = self._cache_per_layer[layer_idx]
        if "k" in lcache and "v" in lcache:
            lcache["k"] = torch.cat([lcache["k"], k], dim=2)
            lcache["v"] = torch.cat([lcache["v"], v], dim=2)
            if is_sliding and lcache["k"].shape[2] > self.sliding_window:
                lcache["k"] = lcache["k"][:, :, -self.sliding_window :, :].contiguous()
                lcache["v"] = lcache["v"][:, :, -self.sliding_window :, :].contiguous()
        else:
            lcache["k"] = k
            lcache["v"] = v
        return lcache["k"], lcache["v"]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Args:
            idx: int, layer index
        Returns:
            k: Tensor of shape (batch_size, num_key_value_heads, total_seq_len, head_dim)
            v: Tensor of shape (batch_size, num_key_value_heads, total_seq_len, head_dim)
        """
        return self._cache_per_layer[idx]["k"], self._cache_per_layer[idx]["v"]


class Gemma3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        output = x.float() * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma3TextScaledEmbeddings(nn.Embedding):
    def __init__(self, config: Gemma3_Config):
        super().__init__(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.embed_scale = math.sqrt(config.hidden_size)

        self._embed_scale: Tensor
        self.register_buffer(
            "_embed_scale",
            torch.tensor(self.embed_scale, dtype=torch.float),
            persistent=False,
        )

    def forward(self, input: LongTensor) -> Tensor:
        """
        Args:
            input: LongTensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        return super().forward(input) * self._embed_scale.type_as(self.weight)


class PyTorchGELUTanh(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate="tanh")


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3_Config):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = PyTorchGELUTanh()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq: Tensor  # (head_dim/2,)
        _demoninator = float(self.base) ** (
            torch.arange(0, self.head_dim, 2).float() / self.head_dim
        )  # (head_dim/2,)
        self.register_buffer("inv_freq", 1.0 / _demoninator, persistent=False)

        self.cos_cached: Tensor  # (1, 1, max_pos_embeddings, head_dim)
        self.sin_cached: Tensor  # (1, 1, max_pos_embeddings, head_dim)
        self._set_cos_sin_cache()

    def _set_cos_sin_cache(self):
        # Maximum position embeddings
        positions = torch.arange(
            self.max_position_embeddings,
            dtype=torch.float,
        )  # (max_pos_embeddings,)

        # Precompute cos and sin caches
        freqs = torch.einsum(
            "i,j->ij",
            positions,
            self.inv_freq,
        )  # (max_pos_embeddings, head_dim/2)

        emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos_embeddings, head_dim)
        cos_cached = emb.cos()  # (max_pos_embeddings, head_dim)
        sin_cached = emb.sin()  # (max_pos_embeddings, head_dim)

        self.register_buffer(
            "cos_cached",
            cos_cached[None, None, :, :],  # (1, 1, max_pos_embeddings, head_dim)
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            sin_cached[None, None, :, :],  # (1, 1, max_pos_embeddings, head_dim)
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        batch_size: int,
        seq_len: int,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            batch_size: int, batch size
            seq_len: int, sequence length
            position_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            cos: Tensor of shape (batch_size, 1, seq_len, head_dim)
            sin: Tensor of shape (batch_size, 1, seq_len, head_dim)
        """
        position_ids = position_ids[:, None, :, None]  # (batch_size, 1, seq_len, 1)
        position_ids = position_ids.expand(batch_size, 1, seq_len, self.head_dim)

        # (batch_size, 1, seq_len, head_dim)
        cos = self.cos_cached.gather(dim=2, index=position_ids)
        sin = self.sin_cached.gather(dim=2, index=position_ids)

        return cos, sin


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3_Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self._scaling = float(config.query_pre_attn_scalar) ** -0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

        self.q_norm = Gemma3RMSNorm(hidden_size=config.head_dim)
        self.k_norm = Gemma3RMSNorm(hidden_size=config.head_dim)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary_pos_emb(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Apply rotary position embeddings to query and key tensors and
        return the modified tensors.
        Args:
            q: Tensor of shape (batch_size, num_attention_heads, seq_len, head_dim)
            k: Tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
            cos: Tensor of shape (batch_size, 1, seq_len, head_dim)
            sin: Tensor of shape (batch_size, 1, seq_len, head_dim)
        Returns:
            q_embed: Tensor of shape (batch_size, num_attention_heads, seq_len, head_dim)
            k_embed: Tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim)
        """
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def _create_causal_mask(
        self,
        q_length: int,
        kv_length: int,
        past_len: Tensor,
    ) -> Tensor:
        """
        Create a causal additive mask for full attention.
        Args:
            q_length: int, query sequence length
            kv_length: int, key/value sequence length
            past_len: Tensor of shape (batch_size, 1), length of cached key/value tensors.
        Returns:
            attention_mask: Tensor of shape (batch_size, 1, q_length, kv_length)
        """
        device, dtype = past_len.device, self.q_proj.weight.dtype
        # i: (batch_size, q_length)
        i = past_len + (torch.arange(q_length, device=device)[None, :])
        # j: (1, kv_length)
        j = torch.arange(kv_length, device=device)[None, :]

        # mask: (batch_size, q_length, kv_length)
        mask = i[:, :, None] >= j[:, None, :]

        mask = torch.where(
            mask,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.finfo(dtype).min,
        )
        return mask[:, None, :, :]  # (batch_size, 1, q_length, kv_length)

    def _create_sliding_window_causal_mask(
        self,
        q_length: int,
        kv_length: int,
        past_len: Tensor,
    ) -> Tensor:
        """
        Create a sliding window causal mask for local attention.
        Args:
            q_length: int, query sequence length
            kv_length: int, key/value sequence length
            past_len: Tensor of shape (batch_size, 1), length of cached key/value tensors.
        Returns:
            attention_mask: Tensor of shape (batch_size, 1, q_length, kv_length)
        """
        device, dtype = past_len.device, self.q_proj.weight.dtype
        # i: (batch_size, q_length)
        i = past_len + (torch.arange(q_length, device=device)[None, :])

        # j: (1, kv_length)
        j_base = (past_len + q_length - kv_length).clamp(min=0)
        j = j_base + torch.arange(kv_length, device=device)[None, :]

        # mask: (batch_size, q_length, kv_length)
        mask = i[:, :, None] >= j[:, None, :]
        mask = mask & (
            j[:, None, :] >= (i[:, :, None] - self.config.sliding_window + 1)
        )

        mask = torch.where(
            mask,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.finfo(dtype).min,
        )
        return mask[:, None, :, :]  # (batch_size, 1, q_length, kv_length)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_values: KVCache,
        past_len: Tensor,
    ) -> Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: tuple of (cos, sin) each of shape (batch_size, 1, seq_len, head_dim)
            past_key_values: KVCache object containing cached key and value tensors.
            past_len: Tensor of shape (batch_size, 1), length of cached key/value tensors.
        Returns:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # (batch_size, num_attn_heads, seq_len, head_dim)
        q = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                seq_len,
                self.config.num_attention_heads,
                self.config.head_dim,
            )
            .transpose(1, 2)
        )

        # (batch_size, num_kv_heads, seq_len, head_dim)
        k = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            .transpose(1, 2)
        )

        v = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                seq_len,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            .transpose(1, 2)
        )

        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Update cached key, value tensors
        # k, v: # (batch_size, num_kv_heads, total_seq_len, head_dim)
        k, v = past_key_values.update(self.layer_idx, k, v)

        # Repeat k, v for grouped query attention heads
        k = k.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads,
            dim=1,
        )  # (batch_size, num_attn_heads, total_seq_len, head_dim)
        v = v.repeat_interleave(
            self.config.num_attention_heads // self.config.num_key_value_heads,
            dim=1,
        )  # (batch_size, num_attn_heads, total_seq_len, head_dim)

        causal_mask = (
            self._create_causal_mask(
                q_length=q.shape[-2],  # seq_len = q_length
                kv_length=k.shape[-2],  # total_seq_len = kv_length
                past_len=past_len,
            )
            if self.layer_type == "full_attention"
            else self._create_sliding_window_causal_mask(
                q_length=q.shape[-2],  # seq_len = q_length
                kv_length=k.shape[-2],  # total_seq_len = kv_length
                past_len=past_len,
            )
        )  # (batch_size, 1, seq_len, total_seq_len)

        # (batch_size, num_attn_heads, seq_len, total_seq_len)
        attn_weights: Tensor = (q @ k.transpose(-2, -1)) * self._scaling

        # attn_weights: (batch_size, num_attn_heads, seq_len, total_seq_len)
        attn_weights = F.softmax((attn_weights + causal_mask).float(), dim=-1)

        # (batch_size, num_attn_heads, seq_len, head_dim)
        attn_output = attn_weights.to(v.dtype) @ v

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.config.num_attention_heads * self.config.head_dim,
        )

        # (batch_size, seq_len, hidden_size)
        return self.o_proj(attn_output)


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3_Config, layer_idx: int):
        super().__init__()
        self.config = config

        self.layer_idx = layer_idx

        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config=config)

        self.input_layernorm = Gemma3RMSNorm(hidden_size=config.hidden_size)
        self.post_attention_layernorm = Gemma3RMSNorm(hidden_size=config.hidden_size)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(hidden_size=config.hidden_size)
        self.post_feedforward_layernorm = Gemma3RMSNorm(hidden_size=config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        past_key_values: KVCache,
        past_len: Tensor,
    ) -> Tensor:
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: tuple of (cos, sin) each of shape (batch_size, 1, seq_len, head_dim)
            past_key_values: KVCache object containing cached key and value tensors.
            past_len: Tensor of shape (batch_size, 1), length of cached key/value tensors.
        Returns:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            past_len=past_len,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3_Config):
        super().__init__()
        self.config = config

        self.embed_tokens = Gemma3TextScaledEmbeddings(config)

        self.rotary_emb = Gemma3RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.rotary_emb_local = Gemma3RotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_local_base_freq,
        )

        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config=config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = Gemma3RMSNorm(hidden_size=config.hidden_size)

    def forward(
        self,
        input_ids: LongTensor,
        past_key_values: KVCache,
        past_len: Tensor,
    ) -> Tensor:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            past_key_values: KVCache object containing cached key and value tensors.
            past_len: Tensor of shape (batch_size, 1), length of cached key/value tensors.
        Returns:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # (batch_size, seq_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)

        position_ids = (
            past_len
            + torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        ).long()  # (batch_size, seq_len)

        # Used for full_attention layers
        position_embeddings_global: tuple[Tensor, Tensor] = self.rotary_emb(
            batch_size,
            seq_len,
            position_ids,
        )

        # Used for sliding_attention layers
        position_embeddings_local: tuple[Tensor, Tensor] = self.rotary_emb_local(
            batch_size,
            seq_len,
            position_ids,
        )

        decoder_layer: Gemma3DecoderLayer
        for decoder_layer in self.layers:  # type: ignore
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=(
                    position_embeddings_global
                    if self.config.layer_types[decoder_layer.layer_idx]
                    == "full_attention"
                    else position_embeddings_local
                ),
                past_key_values=past_key_values,
                past_len=past_len,
            )  # (batch_size, seq_len, hidden_size)

        hidden_states = self.norm(hidden_states)  # (batch_size, seq_len, hidden_size)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    """
    Transformers-compatible Gemma-3 model for causal language modeling.
    """

    def __init__(self, config: Gemma3_Config):
        super().__init__()
        self.config = config

        self.model = Gemma3TextModel(config=config)

        # Tie weights
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: LongTensor,
        past_key_values: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            past_key_values: KVCache object containing cached key and value tensors.
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            present_key_values: KVCache object containing updated cached key and value tensors.
        """
        batch_size, _ = input_ids.shape

        # Initialize KVCache container if not provided
        if past_key_values is None:
            past_key_values = KVCache(
                num_layers=self.config.num_hidden_layers,
                layer_types=self.config.layer_types,
                sliding_window=self.config.sliding_window,
                device=self.lm_head.weight.device,
            )
        past_len = past_key_values.get_kv_length().expand(batch_size, 1)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            past_len=past_len,
        )  # (batch_size, seq_len, hidden_size)

        logits = self.lm_head(outputs)  # (batch_size, seq_len, vocab_size)

        return logits, past_key_values

    def stream_generate(
        self,
        input_ids: LongTensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> Iterator[Tensor]:
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            max_new_tokens: int, maximum number of new tokens to generate.
            temperature: float, temperature for sampling.
            top_k: int | None, if not None, use top-k sampling.
        Returns:
            Yields next_ids: LongTensor of shape (batch_size, 1) for each generated token.
        """
        self.eval()

        generated_ids = input_ids
        past_key_values: KVCache | None = None
        for _ in range(max_new_tokens):
            logits, past_key_values = self(
                input_ids=(
                    input_ids
                    if past_key_values is None
                    else generated_ids[:, -1:]  # Only feed the last token
                ),  # type: ignore
                past_key_values=past_key_values,
            )  # (batch_size, seq_length, vocab_size)

            last = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
            if top_k is not None:
                topv, _ = torch.topk(last, k=top_k)
                # Set logits below top-k to -inf (will become zero after softmax)
                last[last < topv[:, [-1]]] = torch.finfo(last.dtype).min

            next_ids = torch.multinomial(F.softmax(last, dim=-1), num_samples=1)
            generated_ids = torch.cat([generated_ids, next_ids], dim=-1)

            yield next_ids

            if torch.all(generated_ids[:, -1] == self.config.eos_token_id):
                break


@torch.inference_mode()
def _interactive_chat_loop(
    model: Gemma3ForCausalLM,
    tokenizer: PreTrainedTokenizer,
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
):
    assert temperature > 0.0, "Temperature must be greater than 0.0 for sampling."
    assert max_new_tokens > 0, "max_new_tokens must be greater than 0."
    assert top_k is None or top_k > 1, "top_k must be greater than 1."

    messages: list[dict[str, str]] = []
    token: str = str()
    while True:
        print(("\n" if token != "\n" else str()) + "=" * 40)
        user_input = input("<<< ").strip()
        if user_input.lower() in {"exit", "quit"}:
            return

        messages.append({"role": "user", "content": user_input})
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",  # PyTorch tensor
        )

        turn_response = str()
        print(">>> ", end="", flush=True)
        for next_ids in model.stream_generate(
            input_ids=inputs["input_ids"].to(DEVICE),  # type: ignore
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            print(
                token := tokenizer.decode(next_ids.squeeze(), skip_special_tokens=True),
                end="",
                flush=True,
            )
            turn_response += token

        messages.append({"role": "assistant", "content": turn_response.rstrip()})


def main(spec: str = "270M"):
    print(f"[i] Initializing Gemma-3-{spec} model...", end=" ", flush=True)
    # === Initialize model ===
    model: Gemma3ForCausalLM = Gemma3ForCausalLM(
        config=globals()[f"Gemma3_{spec}_Config"]()
    )
    ckpt_path = f"./gemma-3-{spec.lower()}-it.pt"
    print("[Done]")

    print(f"[i] Loading model weights from '{ckpt_path}'...", end=" ", flush=True)
    # === Load model weights ===
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(DEVICE)
    if COMPILE_MODEL:
        model = torch.compile(model)  # type: ignore
    # === Load tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(
        f"google/gemma-3-{spec.lower()}-it",
        cache_dir="./.cache",
    )
    print("[Done]")

    print(f"[i] Using AMP dtype: {AMP_DTYPE}")
    with torch.autocast(
        device_type=DEVICE.type,
        dtype=AMP_DTYPE,
        enabled=DEVICE.type == "cuda",
    ):
        messages = [{"role": "user", "content": "Who are you?"}]
        # === Print hello message ===
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",  # PyTorch tensor
        )
        for next_ids in model.stream_generate(
            input_ids=input_ids["input_ids"].to(DEVICE),
            max_new_tokens=512,
            temperature=1.0,
            top_k=2,
        ):
            print(
                tokenizer.decode(
                    next_ids.squeeze(),
                    skip_special_tokens=True,
                ),
                end="",
                flush=True,
            )

        # === Start interactive chat loop ===
        _interactive_chat_loop(
            model,
            tokenizer,
            temperature=1.0,
            top_k=5,
            max_new_tokens=1024,
        )

    print(f"[i] Done chatting with Gemma-3-{spec}. Bye!")


if __name__ == "__main__":
    main(sys.argv[1].upper() if len(sys.argv) > 1 else "270M")
