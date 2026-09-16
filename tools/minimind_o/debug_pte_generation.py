#!/usr/bin/env python3
"""Reproduce Android MiniMind-O text/audio sampling with published PTEs."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import torch
from executorch.runtime import Runtime
from tokenizers import Tokenizer


CONTEXT = 1024
AUDIO_PAD = 2049
TEXT_EOS = 2
ENTER_TOKEN = 201


def causal_mask(positions: torch.Tensor) -> torch.Tensor:
    columns = torch.arange(CONTEXT)
    return torch.where(
        columns.unsqueeze(0) <= positions.unsqueeze(1),
        torch.tensor(0.0),
        torch.tensor(-1.0e9),
    )


def sample(
    logits: torch.Tensor,
    top_k: int,
    top_p: float,
    temperature: float,
    recent: list[int] | None = None,
    penalty: float = 1.0,
) -> int:
    scores = logits.detach().float().reshape(-1).clone() / temperature
    if recent and penalty != 1.0:
        for token in set(recent):
            score = scores[token]
            scores[token] = score / penalty if score > 0 else score * penalty
    values, indices = torch.topk(scores, top_k)
    probabilities = torch.softmax(values, dim=-1)
    if top_p < 1.0:
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = int(torch.nonzero(cumulative >= top_p)[0]) + 1
        probabilities = probabilities[:keep]
        indices = indices[:keep]
    return int(indices[torch.multinomial(probabilities, 1)])


def stats(audio: torch.Tensor) -> dict[str, float | int]:
    audio = audio.detach().float().reshape(-1)
    finite = torch.isfinite(audio)
    clean = audio[finite]
    return {
        "samples": audio.numel(),
        "finite": int(finite.sum()),
        "nan": int(torch.isnan(audio).sum()),
        "peak": float(clean.abs().max()) if clean.numel() else math.nan,
        "rms": float(clean.square().mean().sqrt()) if clean.numel() else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--mimi", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=128)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt = (
        '<|im_start|>user\n请用中文说一句“语音测试成功”。<|im_end|>\n'
        '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    print("prompt_tokens", len(prompt_ids), prompt_ids)

    runtime = Runtime.get()
    main_method = runtime.load_program(args.main).load_method("forward")
    mimi_method = runtime.load_program(args.mimi).load_method("forward")
    if main_method is None or mimi_method is None:
        raise RuntimeError("forward method missing")

    text_ids = torch.tensor([prompt_ids], dtype=torch.int64)
    sequence = len(prompt_ids)
    positions = torch.arange(sequence, dtype=torch.int64)
    outputs = main_method.execute(
        (
            text_ids,
            torch.full((1, 8, sequence), AUDIO_PAD, dtype=torch.int64),
            torch.zeros((1, sequence, 768), dtype=torch.float32),
            torch.zeros((1, sequence), dtype=torch.float32),
            positions,
            causal_mask(positions),
        )
    )

    generated: list[int] = []
    codes: list[list[int]] = [[] for _ in range(8)]
    stops: list[int | None] = [None] * 8
    frames: list[list[int]] = []
    text_finished = False
    first_finished = True
    position = sequence

    for step in range(args.max_new):
        text_logits = outputs[-2]
        audio_logits = outputs[-1].reshape(8, 2112)
        if text_finished:
            text_token = ENTER_TOKEN if first_finished else 0
            first_finished = False
        else:
            text_token = sample(text_logits, 50, 0.90, 0.75)
        audio_step = step - 1
        for layer in range(8):
            if audio_step < layer:
                code = AUDIO_PAD
            else:
                code = sample(
                    audio_logits[layer], 50, 1.0, 0.20,
                    codes[layer][-3:], 1.05,
                )
            codes[layer].append(code)
            if audio_step >= layer and stops[layer] is None and code >= 2048:
                stops[layer] = len(codes[layer]) - 1

        if not text_finished:
            if text_token == TEXT_EOS:
                text_finished = True
            else:
                generated.append(text_token)
        if audio_step >= 7:
            frame = [codes[layer][step - 7 + layer] for layer in range(8)]
            active = sum(
                stops[layer] is None or step - 7 + layer < stops[layer]
                for layer in range(8)
            )
            if active == 8:
                frames.append([code if 0 <= code < 2048 else 0 for code in frame])

        if text_finished and all(stop is not None for stop in stops):
            break
        next_audio = torch.full((1, 8, 1), AUDIO_PAD, dtype=torch.int64)
        for layer in range(min(audio_step + 1, 8)):
            next_audio[0, layer, 0] = codes[layer][-1]
        next_position = torch.tensor([position], dtype=torch.int64)
        outputs = main_method.execute(
            (
                torch.tensor([[text_token]], dtype=torch.int64),
                next_audio,
                torch.zeros((1, 1, 768), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                next_position,
                causal_mask(next_position),
            )
        )
        position += 1

    print("generated_text", tokenizer.decode(generated, skip_special_tokens=False))
    print("steps", len(codes[0]), "stops", stops, "playable_frames", len(frames))
    for layer in range(8):
        print("codes", layer, codes[layer][:20])
    if len(frames) < 6:
        raise RuntimeError(f"only {len(frames)} playable frames")

    chunks: list[torch.Tensor] = []
    for start in range(0, len(frames) - 5, 4):
        window = frames[start : start + 6]
        flattened = (
            torch.tensor(window, dtype=torch.int64)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        audio = mimi_method.execute((flattened,))[-1].detach().float().reshape(-1)
        if start > 0:
            audio = audio[3840:]
        chunks.append(audio)
    combined = torch.cat(chunks)
    print("decoded", stats(combined))
    pcm = (combined.nan_to_num().clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
    args.wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(pcm)


if __name__ == "__main__":
    torch.manual_seed(7)
    main()
