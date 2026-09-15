#!/usr/bin/env python3
"""Export MiniMind-O's SenseVoice encoder and audio projector.

The Android frontend pads LFR features to 256 frames (about 15.3 seconds of
16 kHz speech) and supplies the real frame count.  A fixed upper bound keeps
SANM export deterministic and avoids reallocating the model for every turn.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn


class AudioProjector(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mlp(value)


class SenseVoiceForMiniMindO(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(
        self,
        lfr_features: torch.Tensor,
        frame_length: torch.Tensor,
    ) -> torch.Tensor:
        # FunASR starts its encoder with `xs_pad *= sqrt(output_size)`.  Clone
        # here so that mutation stays internal; otherwise ExecuTorch correctly
        # exposes the modified user input as an extra public output.
        encoded, _ = self.encoder(lfr_features.clone(), frame_length)
        return self.projector(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensevoice-dir", type=Path, required=True)
    parser.add_argument("--main-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower
    # The encoder itself has no torchaudio dependency. FunASR imports optional
    # frontend modules eagerly, while the newest ExecuTorch toolchain can be
    # ahead of the latest published torchaudio wheel. Stub those unused modules
    # so export does not pin an incompatible binary package.
    for module_name in (
        "torchaudio",
        "torchaudio.compliance",
        "torchaudio.compliance.kaldi",
        "torchaudio.transforms",
        "torchaudio.functional",
    ):
        if module_name not in sys.modules:
            stub = types.ModuleType(module_name)
            stub.__path__ = []
            sys.modules[module_name] = stub
    from funasr import AutoModel

    loaded = AutoModel(
        model=str(args.sensevoice_dir),
        trust_remote_code=True,
        disable_update=True,
        device="cpu",
    )
    encoder = loaded.model.encoder.float().eval()
    config = json.loads((args.main_model_dir / "config.json").read_text())
    projector = AudioProjector(
        int(config["audio_hidden_size"]),
        int(config["hidden_size"]),
    )
    state = torch.load(
        args.main_model_dir / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    projector.load_state_dict(
        {
            name.removeprefix("audio_proj."): value.float()
            for name, value in state.items()
            if name.startswith("audio_proj.")
        }
    )
    model = SenseVoiceForMiniMindO(encoder, projector).eval()
    model.requires_grad_(False)
    example = (
        torch.zeros((1, args.frames, 560), dtype=torch.float32),
        torch.tensor([8], dtype=torch.long),
    )
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
            "get_max_audio_frames": args.frames,
            "get_lfr_width": 560,
            "get_output_width": int(config["hidden_size"]),
        },
    )
    program = edge.to_executorch()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(program.buffer)


if __name__ == "__main__":
    main()
