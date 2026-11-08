#!/usr/bin/env bash
# Retrieve pinned upstream sources and apply this workspace's Studio changes.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

checkout() {
  local dir="$1" url="$2" revision="$3"
  if [ ! -e "$WS/$dir" ]; then
    git clone "$url" "$WS/$dir"
    git -C "$WS/$dir" checkout --detach "$revision"
  fi
  if [ "$(git -C "$WS/$dir" rev-parse HEAD)" != "$revision" ]; then
    echo "$dir is not at the expected revision; refusing to modify it" >&2
    exit 1
  fi
}

checkout PinPoint3D https://github.com/Quit123/PinPoint3D.git 40cb81ec81bb9b525009abbee0f727d92d80db44
checkout MinkowskiEngine https://github.com/NVIDIA/MinkowskiEngine.git 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
checkout interSeg3D-Studio https://github.com/zh-plus/interSeg3D-Studio.git 4605802f547aa17e67e58e4331cb6e64c2bb0bf7

studio="$WS/interSeg3D-Studio"
patch="$WS/patches/interseg3d-studio.patch"
if git -C "$studio" apply --reverse --check "$patch" 2>/dev/null; then
  echo "Studio changes already present"
else
  git -C "$studio" apply --check "$patch"
  git -C "$studio" apply "$patch"
fi
component="$studio/src/frontend/components/MockBackendBanner.vue"
if [ -e "$component" ] && ! cmp -s "$WS/patches/MockBackendBanner.vue" "$component"; then
  echo "Refusing to overwrite modified $component" >&2
  exit 1
fi
cp "$WS/patches/MockBackendBanner.vue" "$component"
echo "Sources ready; see README.md for the checkpoint and environment."
