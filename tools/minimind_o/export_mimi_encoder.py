#!/usr/bin/env python3
"""Export a fixed-window Mimi encoder for on-device reference voice cloning.

The Android app records four seconds at 16 kHz and resamples to 24 kHz. This
model converts that fixed 96,000-sample waveform into the first eight Mimi
codebooks used by MiniMind-O. Voice models remain downloadable release assets
and are never packaged in the APK.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from export_mimi import MimiTransformer, make_convolutions_export_safe


REFERENCE_SAMPLES = 96_000
MAX_ENCODER_FRAMES = 128


class ResidualCodebooks(nn.Module):
    def __init__(self, source: nn.Module, count: int) -> None:
        super().__init__()
        self.input_proj = source.input_proj
        for index in range(count):
            self.register_buffer(
                f"codebook_{index}",
                source.layers[index].codebook.embed.detach().clone().float(),
            )
        self.count = count

    @staticmethod
    def nearest(hidden: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        vectors = hidden.transpose(1, 2).float()
        distances = (
            vectors.square().sum(dim=-1, keepdim=True)
            - 2.0 * torch.matmul(vectors, codebook.transpose(0, 1))
            + codebook.square().sum(dim=-1).view(1, 1, -1)
        )
        return distances.argmin(dim=-1)

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        if self.input_proj is not None:
            hidden = self.input_proj(hidden)
        residual = hidden
        result = []
        for index in range(self.count):
            codebook = getattr(self, f"codebook_{index}")
            codes = self.nearest(residual, codebook)
            residual = residual - F.embedding(codes, codebook).transpose(1, 2)
            result.append(codes)
        return result


class MimiEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.encoder_transformer = MimiTransformer(
            model.encoder_transformer,
            int(model.config.head_dim),
            float(model.config.rope_theta),
            MAX_ENCODER_FRAMES,
        )
        self.downsample = model.downsample
        quantizer = model.quantizer
        self.semantic = ResidualCodebooks(
            quantizer.semantic_residual_vector_quantizer, 1
        )
        self.acoustic = ResidualCodebooks(
            quantizer.acoustic_residual_vector_quantizer, 7
        )
        make_convolutions_export_safe(self.encoder)
        make_convolutions_export_safe(self.downsample)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(waveform)
        hidden = self.encoder_transformer(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.downsample(hidden)
        return torch.stack(self.semantic(hidden) + self.acoustic(hidden), dim=1)


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

    model = MimiEncoder(MimiModel.from_pretrained(args.model_dir)).float().eval()
    model.requires_grad_(False)
    example = (torch.zeros((1, 1, REFERENCE_SAMPLES), dtype=torch.float32),)
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
            "get_reference_samples": REFERENCE_SAMPLES,
            "get_codebooks": 8,
        },
    )
    program = edge.to_executorch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program.buffer)


if __name__ == "__main__":
    main()
