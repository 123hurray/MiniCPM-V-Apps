# Audio8 NPU / INT4 experimental model build

This branch contains static-model preparation, not an integrated NPU Android TTS engine.
The Android application remains the 4.3 CPU baseline. Do not relabel it as an NPU APK.
The stable `audio8-system-tts` branch is unchanged.

## Build and validation

The dedicated `Audio8 NPU INT4 model preparation` workflow runs on pushes to
`audio8-npu-int4`. It verifies the SHA-256 of a pinned, public, independent Realistic
ONNX export, rewrites the two single-token graphs and publishes separate model artifacts.
Model weights and compiled output are never committed to git or bundled into an APK.

Run locally:

```bash
python -m pip install -r scripts/requirements-audio8-npu.txt
python scripts/test_audio8_qnn_model.py
python scripts/build_audio8_npu_models.py --inputs /tmp/audio8-source --output /tmp/audio8-static-models
```

The conversion tools use NumPy and ONNX protobuf/shape inference only. They do not
import ONNX Runtime, initialize QNN, or execute an inference provider. Earlier QNN
execution was paused by a safety review after a telemetry attempt. Moving that
execution into CI is not an approved workaround; this workflow does not do so.

## Transformations

- Keep all MatMulNBits weight bytes, scales and quantization unchanged (INT4).
- Replace unsupported INT4 GatherBlockQuantized lookup with the equivalent FP16 table.
  This increases embedding storage; the artifact is not an all-INT4 model.
- Fix batch=1 and single-token sequence=1 for decode.
- Replace dynamic prefix KV Gather/Concat with fixed-capacity Where updates and a
  position mask. Capacities: slow=2048, fast=10. Keep the original CPU prefill separate.
- Reject incompatible embedding/KV contracts instead of silently converting another graph.

Static tests cover both 24-layer and 4-layer graph structures, exact embedding values,
unchanged INT4 projection weight bytes, invalid contracts, and masked-attention math.
They are not full-model numerical comparison or NPU performance tests.

Before the execution pause, the rewritten Fast AR graph passed SM8550/v73 offline
compilation with CPU fallback disabled, and four CPU comparison cases had maximum
absolute error 0. Those limited results do not establish whole-model or device quality.
The converted slow graph still requires execution validation. The generated manifest
records that validation is pending.

## Remaining integration work

Android ORT/QNN backend integration, CPU prefill, codec placement, model downloader,
NPU delegation/profiling, memory/cancellation behavior, and phone latency/voice-quality
comparison are not completed by this model build. Continue reusing the documented
persistent Audio8 signing certificate for any future APK update.

## Provenance

Source: https://huggingface.co/stephvax/audio8-tts-realistic-0.6b-onnx-int4

Revision: `04313606782af6b9a1aa93724555bc7b64ba9197`.

This is an independent community export, not an official Realistic ONNX release.
The build retains its source LICENSE, NOTICE, README, and provenance alongside the output.
