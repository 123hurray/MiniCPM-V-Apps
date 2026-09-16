#!/usr/bin/env python3
"""Export MiniMind-O's Mimi codec decoder for Android ExecuTorch.

Only the first eight codebooks used by MiniMind-O are accepted.  The model is
decoded in short overlapping windows on Android, which avoids carrying the
Transformers cache through JNI and matches upstream's realtime WebUI path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ExportConv1d(nn.Module):
    """Export-safe equivalent of Transformers' causal MimiConv1d."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.conv = source.conv
        self.kernel_size = int(source.kernel_size)
        self.stride = int(source.stride)
        self.padding_total = int(source.padding_total)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[-1]
        numerator = length - self.kernel_size + self.padding_total
        frames = (numerator + self.stride - 1) // self.stride
        ideal = frames * self.stride + self.kernel_size - self.padding_total
        extra = ideal - length
        return self.conv(F.pad(hidden, (self.padding_total, extra)))


class ExportConvTranspose1d(nn.Module):
    """Export-safe equivalent of Transformers' MimiConvTranspose1d."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.conv = source.conv
        self.padding_left = int(source.padding_left)
        self.padding_right = int(source.padding_right)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(hidden)
        return hidden[..., self.padding_left : hidden.shape[-1] - self.padding_right]


def make_convolutions_export_safe(module: nn.Module) -> None:
    from transformers.models.mimi.modeling_mimi import (
        MimiConv1d,
        MimiConvTranspose1d,
    )

    for name, child in list(module.named_children()):
        if isinstance(child, MimiConv1d):
            setattr(module, name, ExportConv1d(child))
        elif isinstance(child, MimiConvTranspose1d):
            setattr(module, name, ExportConvTranspose1d(child))
        else:
            make_convolutions_export_safe(child)


class MimiAttention(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        head_dim: int,
        rope_theta: float,
        max_sequence: int = 12,
    ) -> None:
        super().__init__()
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.head_dim = head_dim
        self.heads = source.num_heads
        inv = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_sequence).float()
        freqs = torch.outer(positions, inv)
        self.register_buffer("cos", torch.cat((freqs.cos(), freqs.cos()), -1))
        self.register_buffer("sin", torch.cat((freqs.sin(), freqs.sin()), -1))

    @staticmethod
    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        query = self.q_proj(hidden).view(
            batch, sequence, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden).view(
            batch, sequence, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden).view(
            batch, sequence, self.heads, self.head_dim
        ).transpose(1, 2)
        cos = self.cos[:sequence].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:sequence].unsqueeze(0).unsqueeze(0)
        query = query * cos + self.rotate_half(query) * sin
        key = key * cos + self.rotate_half(key) * sin
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        causal = torch.full(
            (self.cos.shape[0], self.cos.shape[0]),
            -1.0e9,
            device=hidden.device,
        ).triu(1)
        scores = scores + causal[:sequence, :sequence].unsqueeze(0).unsqueeze(0)
        attended = torch.matmul(torch.softmax(scores.float(), -1), value)
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        return self.o_proj(attended)


class MimiTransformerLayer(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        head_dim: int,
        rope_theta: float,
        max_sequence: int = 12,
    ) -> None:
        super().__init__()
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        self.attention = MimiAttention(
            source.self_attn, head_dim, rope_theta, max_sequence
        )
        self.fc1 = source.mlp.fc1
        self.fc2 = source.mlp.fc2
        self.attention_scale = source.self_attn_layer_scale
        self.mlp_scale = source.mlp_layer_scale

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention_scale(
            self.attention(self.input_layernorm(hidden))
        )
        return hidden + self.mlp_scale(
            self.fc2(F.gelu(self.fc1(self.post_attention_layernorm(hidden))))
        )


class MimiTransformer(nn.Module):
    def __init__(
        self,
        source: nn.Module,
        head_dim: int,
        rope_theta: float,
        max_sequence: int = 12,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            MimiTransformerLayer(layer, head_dim, rope_theta, max_sequence)
            for layer in source.layers
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class MimiDecoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        semantic = model.quantizer.semantic_residual_vector_quantizer
        acoustic = model.quantizer.acoustic_residual_vector_quantizer
        self.semantic_codebook = nn.Embedding.from_pretrained(
            semantic.layers[0].codebook.embed.detach().clone(),
            freeze=True,
        )
        self.acoustic_codebooks = nn.ModuleList(
            nn.Embedding.from_pretrained(
                acoustic.layers[index].codebook.embed.detach().clone(),
                freeze=True,
            )
            for index in range(7)
        )
        self.semantic_output = semantic.output_proj
        self.acoustic_output = acoustic.output_proj
        self.upsample = model.upsample
        self.decoder_transformer = MimiTransformer(
            model.decoder_transformer,
            int(model.config.head_dim),
            float(model.config.rope_theta),
        )
        self.decoder = model.decoder
        make_convolutions_export_safe(self.upsample)
        make_convolutions_export_safe(self.decoder)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_codebook(codes[:, 0, :]).transpose(1, 2)
        acoustic = self.acoustic_codebooks[0](codes[:, 1, :])
        for index in range(1, 7):
            acoustic = acoustic + self.acoustic_codebooks[index](
                codes[:, index + 1, :]
            )
        acoustic = acoustic.transpose(1, 2)
        embeddings = self.semantic_output(semantic) + self.acoustic_output(acoustic)
        embeddings = self.upsample(embeddings)
        transformer_out = self.decoder_transformer(embeddings.transpose(1, 2))
        return self.decoder(transformer_out.transpose(1, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    from transformers import MimiModel

    model = MimiDecoder(MimiModel.from_pretrained(args.model_dir)).float().eval()
    model.requires_grad_(False)
    # Upstream realtime playback decodes four new frames with two frames of
    # overlap.  Six frames produce 480 ms at 12.5 Hz / 24 kHz.
    example = (torch.zeros((1, 8, 6), dtype=torch.long),)
    exported = torch.export.export(model, example, strict=False)

    if not args.fp32:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=True,
                is_dynamic=True,
            )
        )
        prepared = prepare_pt2e(exported.module(), quantizer)
        prepared(*example)
        converted = convert_pt2e(prepared)
        exported = torch.export.export(converted, example, strict=False)

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        constant_methods={
            "get_sample_rate": 24000,
            "get_codebooks": 8,
            "get_window_frames": 6,
            "get_overlap_frames": 2,
        },
    )
    program = edge.to_executorch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program.buffer)


if __name__ == "__main__":
    main()
