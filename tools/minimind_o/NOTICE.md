# MiniMind-O Android model assets

The Android APK contains only runtime and download code. Model weights are
exported by `export-*.py` and published as separate `.pte` downloads.

- `minimind-o-main-int8.pte` is derived from
  [`jingyaogong/minimind-3o`](https://huggingface.co/jingyaogong/minimind-3o)
  under Apache-2.0.
- `minimind-o-sensevoice-int8.pte` contains the SenseVoiceSmall speech encoder
  (FunAudioLLM, Apache-2.0) plus MiniMind-O's audio projector.
- `minimind-o-mimi-int8.pte` is derived from Kyutai Mimi under CC-BY-4.0.
  The Mimi architecture and weights remain attributed to Kyutai.

The export scripts perform dynamic-activation, per-channel INT8 XNNPACK
quantization. They do not change the upstream licenses.
