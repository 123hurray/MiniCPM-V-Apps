#!/usr/bin/env python3
"""Numerically smoke-test the published Mimi decoder PTE."""

from __future__ import annotations

import argparse
import math
import random
import wave
from pathlib import Path

import torch
from executorch.runtime import Runtime


def statistics(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1)
    finite = torch.isfinite(values)
    clean = values[finite]
    return {
        "samples": values.numel(),
        "finite": int(finite.sum()),
        "nan": int(torch.isnan(values).sum()),
        "inf": int(torch.isinf(values).sum()),
        "min": float(clean.min()) if clean.numel() else math.nan,
        "max": float(clean.max()) if clean.numel() else math.nan,
        "peak": float(clean.abs().max()) if clean.numel() else math.nan,
        "rms": float(clean.square().mean().sqrt()) if clean.numel() else math.nan,
    }


def save_wav(path: Path, values: torch.Tensor) -> None:
    pcm = (
        values.detach().float().reshape(-1).nan_to_num().clamp(-1, 1) * 32767
    ).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pte", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    program = Runtime.get().load_program(args.pte)
    method = program.load_method("forward")
    if method is None:
        raise RuntimeError("forward method missing")

    random.seed(7)
    cases = {
        "zeros": torch.zeros((1, 8, 6), dtype=torch.int64),
        "constant_100": torch.full((1, 8, 6), 100, dtype=torch.int64),
        "random": torch.tensor(
            [[[random.randrange(2048) for _ in range(6)] for _ in range(8)]],
            dtype=torch.int64,
        ),
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, codes in cases.items():
        outputs = method.execute((codes,))
        audio = outputs[-1]
        print(name, tuple(audio.shape), statistics(audio))
        if args.output_dir:
            save_wav(args.output_dir / f"{name}.wav", audio)


if __name__ == "__main__":
    main()
