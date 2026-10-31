# Audio8 NPU / INT4 experimental model build

This branch contains the experimental Android v4.4 ORT/QNN bridge and model preparation.
The stable `audio8-system-tts` branch remains the 4.3 CPU baseline.

## On-device setup

1. Update using the same Audio8 certificate. Keep the existing base GGUF: its tokenizer,
   prompt builder, reference encoder and CPU codec are still used.
2. Download the extra NPU package inside the app. It contains original INT4 CPU prefill
   graphs plus adapted fixed-capacity NPU decode graphs, about 0.9 GB in total.
   The model Release origin can be replaced by an HTTPS mirror; the APK pins the exact
   catalog SHA-256 and verifies every file. Reserve at least 5 GB of free storage.
3. Enable the NPU checkbox and save. Reconnect the system engine. Initial QNN compilation
   may take minutes and uses additional memory; subsequent loads reuse the context cache.
4. Slow single-token decode uses SM8550/v73 HTP with CPU fallback disabled. CPU handles
   prefill and codec. Fast AR defaults to CPU INT4 and can independently use HTP.
5. Turn PCM cache off when comparing CPU and NPU inference. Use the same reference voice,
   compare warm runs separately from cold model loading/compilation, and export diagnostics.
   QNN placement, call counts, stage timings and available QNN CSV profiles are included.

If initialization fails, the request errors instead of silently synthesizing through CPU.
Disable NPU to return to the baseline. No actual SM8550 latency, audio-quality or memory
validation is claimed from a cloud compilation.

## Build and validation

The dedicated `Audio8 NPU INT4 model preparation` workflow runs on pushes to
`audio8-npu-int4`. It verifies the SHA-256 of a pinned, public, independent Realistic
ONNX export, rewrites the two single-token graphs and publishes separate model assets
in a prerelease. The dependent workflow builds the Android APK with the catalog hash.
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
The Android bridge disables ORT telemetry and, before creating its environment or loading
QNN, requires an additional synchronized seccomp filter denying new AF_INET/AF_INET6
sockets in the separate synthesis process. UNIX driver RPC remains allowed. The UI
downloader is a different process and retains Internet access. NPU initialization fails
closed if the filter cannot be installed. This isolation is not a substitute for device
validation and does not resume the previously blocked host inference test.

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

## Remaining validation

The bridge, downloader and settings are implemented, but full numerical parity, first
audible PCM, steady-state latency, peak PSS, thermal behavior, voice similarity, driver
compatibility and cancellation require phone validation. Prefill/model compilation and
an in-flight codec kernel remain non-interruptible. Shared-memory/zero-copy KV transfer
is not implemented; transfer overhead may reduce or erase an NPU speed advantage.
Continue reusing the documented persistent Audio8 signing certificate for updates.

## Provenance

Source: https://huggingface.co/stephvax/audio8-tts-realistic-0.6b-onnx-int4

Revision: `04313606782af6b9a1aa93724555bc7b64ba9197`.

This is an independent community export, not an official Realistic ONNX release.
The build retains its source LICENSE, NOTICE, README, and provenance alongside the output.
