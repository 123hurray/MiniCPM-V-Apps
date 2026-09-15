#!/usr/bin/env python3
"""Export the MiniMind-O Thinker/Talker core for Android.

The upstream model is intentionally small, but its Python implementation keeps
KV tensors in Python objects.  Passing those tensors through JNI on every token
is prohibitively expensive.  This export-friendly implementation keeps a fixed
KV cache as mutable model buffers and updates it by ``input_pos``.  The Android
runtime therefore sends only the current token(s), aligned audio features and a
small attention mask.

Weights are downloaded at build time and are never packaged in the APK.  The
resulting ``.pte`` file is a separately downloadable model asset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (self.weight.float() * normalized).to(value.dtype)


class StaticKVCache(nn.Module):
    """Model-owned cache with an explicit position input.

    Old values do not need to be cleared between turns.  The caller restarts at
    position zero and masks every position after the current sequence end.
    """

    def __init__(self, heads: int, max_context: int, head_dim: int) -> None:
        super().__init__()
        shape = (1, heads, max_context, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(shape), persistent=False)

    def update(
        self,
        input_pos: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k_cache.index_copy_(2, input_pos, keys)
        self.v_cache.index_copy_(2, input_pos, values)
        return self.k_cache, self.v_cache


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_context: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_repeats = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.cache = StaticKVCache(num_kv_heads, max_context, head_dim)

    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        queries = self.q_proj(hidden).view(
            batch, sequence, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden).view(
            batch, sequence, self.num_kv_heads, self.head_dim
        )
        queries = self.q_norm(queries).transpose(1, 2)
        keys = self.k_norm(keys).transpose(1, 2)
        values = values.transpose(1, 2)

        rope_cos = cos.index_select(0, input_pos).unsqueeze(0).unsqueeze(0)
        rope_sin = sin.index_select(0, input_pos).unsqueeze(0).unsqueeze(0)
        queries = queries * rope_cos + self._rotate_half(queries) * rope_sin
        keys = keys * rope_cos + self._rotate_half(keys) * rope_sin

        keys, values = self.cache.update(input_pos, keys, values)
        keys = keys.repeat_interleave(self.num_repeats, dim=1)
        values = values.repeat_interleave(self.num_repeats, dim=1)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        scores = scores + attention_mask.unsqueeze(0).unsqueeze(0)
        probs = torch.softmax(scores.float(), dim=-1).to(queries.dtype)
        context = torch.matmul(probs, values)
        context = context.transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(context)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class Block(nn.Module):
    def __init__(self, config: dict, max_context: int) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.self_attn = Attention(
            hidden,
            int(config["num_attention_heads"]),
            int(config["num_key_value_heads"]),
            int(config["head_dim"]),
            max_context,
            float(config["rms_norm_eps"]),
        )
        self.input_layernorm = RMSNorm(hidden, float(config["rms_norm_eps"]))
        self.post_attention_layernorm = RMSNorm(
            hidden, float(config["rms_norm_eps"])
        )
        self.mlp = FeedForward(hidden, int(config["intermediate_size"]))

    def forward(
        self,
        hidden: torch.Tensor,
        input_pos: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), input_pos, cos, sin, attention_mask
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Transformer(nn.Module):
    def __init__(
        self,
        config: dict,
        max_context: int,
        layer_count: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        hidden = int(config["hidden_size"])
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            Block(config, max_context) for _ in range(layer_count)
        )
        self.norm = RMSNorm(hidden, float(config["rms_norm_eps"]))


class TalkerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, layers: int = 8) -> None:
        super().__init__()
        self.base = nn.Embedding(vocab_size, hidden_size)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Embedding(vocab_size, 256),
                nn.GELU(),
                nn.Linear(256, hidden_size, bias=False),
            )
            for _ in range(layers)
        )

    def forward(self, audio_ids: torch.Tensor) -> torch.Tensor:
        combined = self.base(audio_ids[:, 0, :]) + self.adapters[0](
            audio_ids[:, 0, :]
        )
        for index in range(1, 8):
            combined = combined + self.base(audio_ids[:, index, :])
            combined = combined + self.adapters[index](audio_ids[:, index, :])
        return combined / 8.0


class TalkerHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.base = nn.Linear(hidden_size, vocab_size, bias=False)
        self.adapters = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, 256, bias=False),
                nn.GELU(),
                nn.Linear(256, vocab_size, bias=False),
            )
            for _ in range(8)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden)
        return torch.stack(
            [base + adapter(hidden) for adapter in self.adapters], dim=0
        )


class Talker(nn.Module):
    def __init__(self, config: dict, max_context: int) -> None:
        super().__init__()
        talker_config = dict(config)
        talker_config["hidden_size"] = int(config["talker_hidden_size"])
        self.text_scale = nn.Parameter(torch.tensor(3.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.layers = nn.ModuleList(
            Block(talker_config, max_context)
            for _ in range(int(config["num_talker_hidden_layers"]))
        )
        hidden = int(config["talker_hidden_size"])
        self.norm = RMSNorm(hidden, float(config["rms_norm_eps"]))
        self.lm_head = TalkerHead(hidden, int(config["audio_vocab_size"]))
        self.embed_tokens = TalkerEmbedding(int(config["audio_vocab_size"]), hidden)
        self.codec_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            RMSNorm(hidden, float(config["rms_norm_eps"])),
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(int(config["hidden_size"]), int(config["hidden_size"])),
            nn.GELU(),
            nn.Linear(int(config["hidden_size"]), hidden),
            RMSNorm(hidden, float(config["rms_norm_eps"])),
        )
        # Kept for state-dict compatibility and future downloadable voices.
        self.spk_proj = nn.Linear(int(config["spk_emb_size"]), hidden, bias=False)


class MiniMindOMain(nn.Module):
    """Speech-native Thinker/Talker with mutable caches and aligned audio input."""

    def __init__(self, config: dict, max_context: int = 1024) -> None:
        super().__init__()
        self.max_context = max_context
        hidden = int(config["hidden_size"])
        self.model = Transformer(
            config,
            max_context,
            int(config["num_hidden_layers"]),
            int(config["vocab_size"]),
        )
        self.lm_head = nn.Linear(hidden, int(config["vocab_size"]), bias=False)
        self.talker = Talker(config, max_context)
        self.bridge_layer = int(config["bridge_layer"])
        head_dim = int(config["head_dim"])
        theta = float(config["rope_theta"])
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_context)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer(
            "rope_cos", torch.cat((freqs.cos(), freqs.cos()), dim=-1), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.cat((freqs.sin(), freqs.sin()), dim=-1), persistent=False
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        audio_ids: torch.Tensor,
        audio_features: torch.Tensor,
        audio_mask: torch.Tensor,
        input_pos: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model.embed_tokens(text_ids)
        hidden = torch.where(audio_mask.unsqueeze(-1) > 0.5, audio_features, hidden)
        bridge = hidden
        for index, layer in enumerate(self.model.layers):
            hidden = layer(
                hidden, input_pos, self.rope_cos, self.rope_sin, attention_mask
            )
            if index == self.bridge_layer:
                bridge = hidden
        thinker_hidden = self.model.norm(hidden)

        talker_hidden = (
            self.talker.embed_proj(bridge) * self.talker.text_scale
            + self.talker.codec_proj(self.talker.embed_tokens(audio_ids))
            * self.talker.audio_scale
        )
        for layer in self.talker.layers:
            talker_hidden = layer(
                talker_hidden,
                input_pos,
                self.rope_cos,
                self.rope_sin,
                attention_mask,
            )
        talker_hidden = self.talker.norm(talker_hidden)
        return (
            self.lm_head(thinker_hidden[:, -1, :]),
            self.talker.lm_head(talker_hidden[:, -1, :]),
        )


def causal_mask(positions: torch.Tensor, max_context: int) -> torch.Tensor:
    columns = torch.arange(max_context, device=positions.device)
    return torch.where(
        columns.unsqueeze(0) <= positions.unsqueeze(1),
        torch.tensor(0.0, device=positions.device),
        torch.tensor(-1.0e9, device=positions.device),
    )


def load_model(model_dir: Path, max_context: int) -> MiniMindOMain:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("use_moe"):
        raise ValueError("The Android runtime currently supports dense minimind-3o only")
    model = MiniMindOMain(config, max_context)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name
        for name in missing
        if ".cache." in name or name in {"rope_cos", "rope_sin"}
    }
    real_missing = sorted(set(missing) - allowed_missing)
    allowed_unexpected = {
        name for name in unexpected if name.startswith(("audio_proj.", "vision_proj."))
    }
    real_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"State-dict mismatch; missing={real_missing}, unexpected={real_unexpected}"
        )
    return model.float().eval()


def export_pte(
    model: MiniMindOMain,
    output: Path,
    max_context: int,
    quantize: bool = True,
) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.backends.xnnpack.utils.configs import (
        get_xnnpack_edge_compile_config,
    )
    from executorch.exir import to_edge_transform_and_lower

    # Export is inference-only.  Freezing parameters is also required because
    # the model-owned KV cache is represented by in-place buffer mutations.
    # Leaving parameters trainable makes the ExecuTorch legalization pass treat
    # copied cache values as autograd leaves and reject the graph.
    model.requires_grad_(False)

    sequence = 8
    positions = torch.arange(sequence, dtype=torch.long)
    example = (
        torch.zeros((1, sequence), dtype=torch.long),
        torch.full((1, 8, sequence), 2049, dtype=torch.long),
        torch.zeros((1, sequence, 768), dtype=torch.float32),
        torch.zeros((1, sequence), dtype=torch.float32),
        positions,
        causal_mask(positions, max_context),
    )
    seq = torch.export.Dim("sequence", min=1, max=max_context)
    dynamic_shapes = (
        {1: seq},
        {2: seq},
        {1: seq},
        {1: seq},
        {0: seq},
        {0: seq},
    )
    exported = torch.export.export(
        model, example, dynamic_shapes=dynamic_shapes, strict=False
    )
    if quantize:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        # Dynamic activations + per-channel int8 weights substantially reduce
        # both download size and memory bandwidth.  No representative audio
        # calibration set is needed for this mode.
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=True,
                is_dynamic=True,
            )
        )
        # Keep final classifier heads in FP32. Quantizing them saves only a
        # small fraction of the total model but can perturb near-tied token
        # ranks enough to noticeably change sampled speech/text.
        def quantize_non_head(node: torch.fx.Node) -> bool:
            stack = node.meta.get("nn_module_stack", {})
            paths = [value[0] for value in stack.values()]
            return "lm_head" not in paths and not any(
                path.startswith("talker.lm_head") for path in paths
            )

        quantizer.set_filter_function(quantize_non_head)
        prepared = prepare_pt2e(exported.module(), quantizer)
        prepared(*example)
        converted = convert_pt2e(prepared)
        exported = torch.export.export(
            converted,
            example,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=get_xnnpack_edge_compile_config(),
        constant_methods={
            "get_max_context_len": max_context,
            "get_audio_pad_id": 2049,
            "get_text_eos_id": 2,
        },
    )
    program = edge.to_executorch()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(program.buffer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Disable the default XNNPACK dynamic-int8 quantization",
    )
    args = parser.parse_args()
    model = load_model(args.model_dir, args.max_context)
    if args.check_only:
        positions = torch.arange(4)
        outputs = model(
            torch.zeros((1, 4), dtype=torch.long),
            torch.full((1, 8, 4), 2049, dtype=torch.long),
            torch.zeros((1, 4, 768)),
            torch.zeros((1, 4), dtype=torch.float32),
            positions,
            causal_mask(positions, args.max_context),
        )
        print(tuple(tuple(value.shape) for value in outputs))
        return
    export_pte(model, args.output, args.max_context, quantize=not args.fp32)


if __name__ == "__main__":
    main()
