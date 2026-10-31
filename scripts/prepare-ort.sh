#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ort="$root/Audio8-Android/ort-sdk"
version=1.29.0
mkdir -p "$ort/include"
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_ep_c_api.h onnxruntime_error_code.h; do
  curl --fail --location --silent --show-error --retry 3 \
    "https://raw.githubusercontent.com/microsoft/onnxruntime/v$version/include/onnxruntime/core/session/$header" -o "$ort/include/$header"
done
curl --fail --location --silent --show-error --retry 3 \
  "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$version/onnxruntime-android-$version.aar" -o "$ort/ort.aar"
unzip -oq "$ort/ort.aar" 'jni/arm64-v8a/libonnxruntime.so' -d "$ort"
test -s "$ort/jni/arm64-v8a/libonnxruntime.so"
licenses="$root/Audio8-Android/app/src/main/assets/licenses"
mkdir -p "$licenses"
curl --fail --location --silent --show-error --retry 3 \
  "https://raw.githubusercontent.com/microsoft/onnxruntime/v$version/LICENSE" -o "$licenses/onnxruntime-LICENSE"
curl --fail --location --silent --show-error --retry 3 \
  'https://repo.maven.apache.org/maven2/com/qualcomm/qti/qnn-runtime/2.51.0/qnn-runtime-2.51.0.aar' -o "$ort/qnn-runtime.aar"
unzip -p "$ort/qnn-runtime.aar" LICENSE.pdf > "$licenses/qnn-runtime-LICENSE.pdf"
unzip -p "$ort/qnn-runtime.aar" NOTICE.txt > "$licenses/qnn-runtime-NOTICE.txt"
echo 'Pinned ORT headers and Android library prepared; no inference executed'
