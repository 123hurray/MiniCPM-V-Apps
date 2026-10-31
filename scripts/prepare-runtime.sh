#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
runtime="$root/Audio8-Android/runtime"
revision=3a0dd44293fc706f5c375ec96d3e6fd9c54b6c45
if [[ ! -d "$runtime/.git" ]]; then git clone --no-checkout https://github.com/0xShug0/audio.cpp.git "$runtime"; fi
git -C "$runtime" checkout --detach "$revision"
test "$(git -C "$runtime" rev-parse HEAD)" = "$revision"
patch="$root/scripts/audio8-mobile.patch"
if git -C "$runtime" apply --check "$patch"; then
  git -C "$runtime" apply "$patch"
elif git -C "$runtime" apply --reverse --check "$patch"; then
  echo 'Pinned mobile optimization patch already applied'
else
  echo 'Runtime does not match the pinned optimization patch; refusing to continue' >&2
  exit 1
fi
sha256sum "$patch"
