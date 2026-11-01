# Audio8 Android system TTS

## Version 5.2 — persistent LiteRT INT8 codec

The default codec now uses the pinned 132 MB community LiteRT INT8 model with persistent buffers and weight caching. AR remains Preview 0.1B. Download the new LiteRT codec in settings; the default path skips QNN graph preparation. Same-code cold/warm CPU thread benchmarks, optional FP16/GPU comparisons, opt-in real-code export and offline folding/QDQ calibration tools are included. See [setup, validation limits and offline tools](docs/audio8-codec-optimization.md). Version code 48 retains the signing certificate. The ten-second target requires measurement on the target phone.

## Version 5.1 — bound NPU codec graph memory

A v5.0 device report failed after 48.6 s preparing 256 frames with QNN error 1002
MEM_ALLOC during graph finalization. This does not establish which allocator
failed. Preparation no longer loads/retains GGUF, AR or CPU codec weights. It
releases an existing engine before independently preparing QNN codec contexts.

The exact codec graph is partitioned into 14 sequential graph domains per capacity,
cut at decoder residual boundaries. Every cross-part live tensor is preserved;
there is no speech chunking, changed weights or CPU fallback. Graph composition
is checked against the static source, including external weight byte offsets,
boundary shapes/dtypes, and the single-consumer rule for large boundary buffers.
Only one QNN session is resident during preparation or execution. Large consumed
boundary buffers are released after copying outputs safely (including aliases).
Tiny shape metadata can remain live. Existing downloaded external weights are reused.
Update the small graph headers through the codec download button and prepare again;
old monolithic contexts are not used. Per-part timings and RSS/peak virtual-memory
samples identify any remaining allocator failure. NPU execution, memory limits,
audio parity and the under-10-second target still need real-device verification.
Version code 47 retains the signing certificate and the user's NPU selection.

## Version 5.0 — separate NPU preparation and improve Falcon AR scheduling

Pure ONNX reference constant folding freezes 128/256-frame graphs and reduces
3341 nodes to 1646. Two pinned static headers are published with the APK; the
original external weight file is reused without redownloading. No ORT/QNN host
inference is used to build them.

The device log showed 1.12 s loading, 23.89 s AR and at least 109 s blocked in
request-time QNN session creation. NPU choice is retained, with no CPU fallback.
Download the codec, enable/save codec NPU, then explicitly prepare 128 or 256-frame
graphs before preview. Preparation has its own 240 s process watchdog. Synthesis
uses prepared contexts only and fails quickly on a cache miss. Graph finalization
uses mode 0 instead of mode 3 to prioritize preparation time; actual inference
performance must be measured. Two capacity contexts are retained, not evicted at
every different utterance length. Longer than 256 frames fails explicitly.

Codebooks are right-padded to capacity and PCM cropped to real frame length.
The pinned source has causal window attention, causal convolutions and per-feature
normalization; future padding should preserve the output prefix. This is an
architectural inference, not completed full-model numeric/audio validation.
No independent utterance splitting or overlapping-audio approximation is used.
Do not claim successful NPU execution or measured speedup from a cloud build.

v4.9 initialized QNN before AR. v5.0 defers it until codec decode and releases
it afterwards; explicit preparation also releases its runtime on completion.
This isolates AR from idle provider threads/resources; cached contexts are reloaded
at codec time, whose overhead is logged. This is a suspected interference fix,
not an established cause without an on-device A/B test.

AR has request-scoped ggml threadpool reuse (idle workers park), correct active
slow-thread controls in the Falcon branch, cancellation at every prompt/generated
token, progress and separated prefill/decode/compute/build/allocation/copy timings.
The old slow_ms=0 omitted Falcon computation. v4.8 and v4.9 used the same Falcon
math path; the log alone cannot establish a regression against a previous run.
CI checks pool numerical parity, thread-count changes and exception teardown.
Version code 46 retains the signing certificate. Device parity and latency pending.

## Version 4.9 — opt-in codec-only QNN experiment

AR remains CPU. In advanced settings download the separate pinned 261 MB official
Preview 0.1B ONNX FP16 codec, then enable and save the codec NPU option. Default is
CPU. Restart the engine after changing backends. SM8550 only. The FP16 HTP attempt
is NOT validated on-device; unsupported operators/precision fail without CPU fallback.
The published CPU-oriented ONNX codec is not claimed to be an HTP-ready model.
Exact frame length overrides batch/frames/samples; no zero-padding, audio splitting,
quantization or approximate activation has been introduced. A single in-memory session
and the latest embedded QNN context are retained, keyed by source revision and frame
length. First compilation can time out; test a short phrase first. Disable PCM cache
when benchmarking. Logs separate graph preparation, input conversion, Run, output
copy and total wall time, frames/audio duration/RTF. Existing timeouts and process
network isolation apply. AR and reference encoding still need the GGUF model.

CPU codec retains the GEMM transpose-convolution and fused Snake paths. Snake now
hoists channel indexing and alpha lookup out of the sample loop, preserving sin and
division rather than introducing an approximate activation. CPU graph preparation is
logged separately from execution. Actual gains require device measurements.

ONNX source: Edge0/audio8-TTS-0.1B-ONNX-INT8 revision
`317c12d4e0da83847b594fcf8bd74bf2c76615ec`. File size and SHA-256 are pinned.
See the source model card for original license and attribution; no model weights in
APK/git. Release code 45 reuses the persistent certificate. Cloud builds prove
compilation only; first successful HTP execution and audio parity remain pending.

## Version 4.8 — Preview 0.1B CPU evaluation

The APK now downloads the pinned 851,018,912-byte Preview 0.1B Q8 GGUF
from the `audio8-preview01-q8-v1` GitHub Release, with SHA-256 verification.
It uses distinct final/partial filenames, retaining the previous Realistic model.
Old NPU preferences are ignored; the Realistic NPU export is incompatible.
HF mirror settings do not apply to this release. CPU inference and the
existing voice-reference controls remain available. This is a Preview checkpoint
under Audio8 Community License v1.0, not the Apache-licensed Realistic model.
Only native x86 CPU smoke validation exists; Android quality/speed need testing.
Version code 44 reuses the persistent release signing certificate.

## Version 4.7 — CPU codec kernels and independent decode threads

Codec defaults to 5 independently configurable CPU threads, no longer inheriting
a slow-AR/prefill setting of 2. Severe thermal status caps it to 3. A separate
switch enables GEMM plus parallel overlap-add for F32/F16 transposed convolutions
and single-pass CPU Snake activation. Weight dtype and sample rate are unchanged;
no text/audio splitting or independent per-chunk voice generation is introduced.
Disable the codec fast-path switch to compare the old kernels at the same thread
count. The native tests compare both paths at 1/5 threads, multiple stride/length
configurations and both weight dtypes; whole-model phone quality/performance still
requires testing. Graph reuse now requires the exact frame count to avoid decoding
the padded capacity of a previous longer utterance. Diagnostic output records
codec threads, selected optimization, frame/capacity count and compute milliseconds.
Existing model files, QNN caches and signing certificate remain unchanged.

## Version 4.6 — pinned prefill output contract and QNN exception cleanup

The pinned slow graph emits only the last position: FP32 logits `[1,1,4097]`
and FP16 hidden `[1,1,896]`, even for a multi-token CPU prefill. Read these
outputs without a prompt-length offset; retain full prompt-length KV updates.
Unexpected shapes still fail with actual and expected dimensions in diagnostics.
QNN registration is now an owned RAII member: both normal teardown and a failed
constructor release sessions before unregistering the provider, then release Env.
Failed registration never unregisters somebody else's provider. Native tests
exercise the shape contract, rejected layouts, teardown order and repeated retries
with fake provider/session objects, without executing ORT/QNN on the build host.
Models, model hashes, compiled-cache paths and signing certificate are unchanged.
The user's v4.5 log confirms HTP initialization/compilation, not successful NPU
token generation, audible output, numerical parity or a speed improvement.

## Version 4.5 — bounded requests and concise status

Automatic voice-selection warmup has been removed, including for old preferences.
The home screen shows only readiness, the latest phase with elapsed time, and preview /
stop / recovery controls. Model, voice and performance details are collapsed by default.
TTS binder operations run off the UI thread. A separate-process watchdog ends only the
synthesis process after 240 seconds of initialization or 180 seconds of synthesis;
stop allows 15 seconds for native cancellation before the same process-only recovery.
A private receiver provides manual recovery without deleting models. Reconnect the
engine after a reset. Initialization phases and CPU AR / codec stages are persisted
before execution, so a failure can be localized. This bounds waiting; it does not
establish a successful NPU run or a measured latency improvement on the phone.

## Experimental branch: audio8-npu-int4

Version 4.4 integrates an experimental ORT/QNN bridge: CPU INT4 prefill, HTP INT4
single-token slow decode, configurable CPU INT4 / HTP fast decode, and the existing
CPU codec. It is not an all-NPU or phone-validated engine. The workflow publishes
separate downloadable models and binds their catalog hash into the APK. Model weights
are not bundled. Existing GGUF and reference files are retained.
See [setup, security constraints, and validation limits](docs/audio8-npu-int4.md).
The stable `audio8-system-tts` branch remains unchanged.

## Version 4.3 — Snapdragon 8 Gen 2 CPU latency optimization

- ARMv8.6-A FP16/dotprod/I8MM kernels and release LTO when supported. This build targets
  SM8550; it is not a portable ARMv8.2 APK.
- Project only the model-metadata semantic token rows and EOS (4096+1 instead of the
  full 155776-token head). Input embeddings remain intact; original token IDs, EOS,
  top-k/top-p/RAS and seed remain unchanged. Unsupported shapes fall back to the full head.
- Reuse the prefill graph only while both shape and target KV allocation match; reuse
  the single codec capacity graph. Turning graph retention off restores release-after-use.
- Independently configurable slow/codec and fast AR threads (default 5/2); optional
  affinity to the five highest-frequency allowed cores, discovered from sysfs instead
  of hard-coded CPU IDs. Severe thermal state caps active slow threads to 3.
- Background model/short-utterance warmup when selecting a voice; a real request cancels
  warmup at an AR frame boundary. Native cancellation between AR frames and fast steps.
- LRU PCM cache in the synthesis process, bounded to 16 MiB / 32 entries / 4 MiB per
  utterance. Keys include text, voice/reference, transcript, rate/pitch and native settings.
  Speech and PCM are not persisted. Settings use atomic files so the separate synthesis
  process sees changes without stale SharedPreferences.
- Phase timings and first PCM / total / cache hit / threads / thermal state diagnostics.
  UI exposes all switches and baseline/recommended presets. Changing native settings
  reloads the session on the next request; it does not mutate an existing graph's shape.

These are implemented optimizations, not measured SM8550 speedup claims. CI tests compare
F32 and Q8 full/compact projection at batch 1 and 4 (including EOS mapping), cancellation,
PCM cache LRU/limits, source+merged TTS manifests, Android Java/native compilation and lint.
On-device end-to-end audio quality, peak PSS and latency still need measurement.

Not silently enabled: INT4 changes the quantization/quality tradeoff; the available LiteRT
export is Preview rather than this Realistic model. QNN/HTP needs a new supported static
quantized graph export and calibration. Existing GGUF CPU graphs do not become NPU graphs
through a build flag. Vulkan/codec correctness is not validated for this app on Adreno 740.
The current build deliberately remains CPU-only. Real streaming codec state is not implemented;
no independent text chunking or overlapping-window approximation is used to hide latency.

Version 4.2 fixes engine discovery: the TTS service filter now declares
`android.intent.category.DEFAULT`, required by Android TtsEngines' use of
`PackageManager.MATCH_DEFAULT_ONLY`. Source and merged manifests are checked in CI.
The app also includes a package-manager discovery check for on-device diagnosis.

Version 4.1 adds an editable HTTPS Hugging Face mirror origin, defaulting to
`https://hf-mirror.com`, with an official-direct shortcut. Both download the identical
pinned model path and must pass the same SHA-256 check. Switching endpoints keeps
the existing partial model. Settings take effect on the next download attempt.
HF-Mirror is a third-party mirror, not a Hugging Face-operated service.

Audio8 release certificate SHA-256 (reuse for every future update):
`42bf8536d97d3a3e6ccb42252a91cd471296045c6455eb7d55d202bb868a8905`.
Private key recovery is kept separately, never in this repository.

This branch replaces the previous multimodal/agent/ASR application with an Android-only
Audio8-TTS-Realistic-0.6B engine. Other branches are unchanged. No model is bundled.

## Scope

- Android 13+, arm64, ARMv8.6-A FP16/dot-product/I8MM; aimed at Snapdragon 8 Gen 2.
- Standalone application ID `com.example.audio8tts`, coexists with the old app.
- Android `TextToSpeechService`, Chinese/English advertised initially, offline synthesis.
- Pinned audio.cpp CPU backend; no untested GPU switch.
- Download/HTTP resume/SHA-256 validation for the 1,429,460,800-byte Q8 model.
- User-imported reference WAV and accurate transcript; no automatic speaker changes.
- Diagnostics export through SAF, including native stderr and available Android exit traces.
- No chat, ASR, agents, shell, Python, camera, vision, or other TTS model.

## Experimental limitations — do not represent this as phone-tested

Cloud compilation is not proof of audible output, quality, memory safety, or real-time
performance on the phone. Validate on an SM8550 before production use.
Current upstream synthesis completes an utterance before returning PCM. A request is never
split independently to work around OOM. Requests over 500 Unicode code points fail explicitly.
Stopping suppresses subsequent PCM immediately and cancels at the next AR/fast-AR boundary.
Model loading, prefill and one in-flight codec compute remain non-interruptible.
Independent pitch control/time-stretch is not implemented;
speech rate uses resampling and changes pitch. Use default 100/100 settings for natural output.
Reference-free output is not guaranteed to preserve the same speaker across requests.
Not yet suitable for low-latency screen-reader navigation.

## Build

Run `bash scripts/prepare-runtime.sh`, then `cd Audio8-Android && bash gradlew :app:assembleRelease`.
Runtime is pinned to `3a0dd44293fc706f5c375ec96d3e6fd9c54b6c45`.
`scripts/audio8-mobile.patch` applies the audited mobile changes to that exact revision;
the setup fails instead of guessing if the patch no longer applies. It is idempotent.
The dedicated GitHub Actions workflow builds on pushes to `audio8-system-tts`.
Only `audio8_tts` is selected in the native model composite.

## Persistent signing

The historical release certificate is documented in the parent branch; its private key is not
recoverable from an APK or certificate fingerprint. Configure repository Actions secrets:
`ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`.
Reuse that same key for all future builds. Without it, the workflow deliberately uploads unsigned
signing inputs only, not an installable APK. Never generate an ephemeral debug signing key.

## Phone validation

1. Install the signed APK, download the model, import an authorized clean reference and transcript.
2. Open system text-to-speech settings and choose Audio8; default speed and pitch.
3. Test short Chinese, English, 100/300/500-character text, queue/flush/stop, and service restart.
4. Test through a second application (e.g. a reader using Android TTS), not only the preview UI.
5. Measure peak PSS, first audible PCM, audio duration/generation duration, temperature over 10 min.
6. Export diagnostics after errors. No claim of completed device testing without results.

## Sources and license

- https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b (Apache-2.0)
- https://huggingface.co/audio-cpp/Audio8-TTS-Realistic-0.6b-GGUF
- https://github.com/0xShug0/audio.cpp
- https://developer.android.com/reference/android/speech/tts/TextToSpeechService

Retain upstream license/NOTICE when distributing runtime or model weights. Voice cloning requires
permission from the speaker. Native/download logs may contain paths or upstream diagnostic data;
review exported logs before sharing.
