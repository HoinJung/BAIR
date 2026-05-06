#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p third_party

clone_if_missing() {
  local url="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    echo "[skip] $dest already exists"
    return
  fi
  git clone "$url" "$dest"
}

clone_if_missing https://github.com/hiyamdebary/EarthDial.git third_party/EarthDial
clone_if_missing https://github.com/mbzuai-oryx/GeoChat.git third_party/GeoChat
clone_if_missing https://github.com/Norman-Ou/GeoPix.git third_party/GeoPix
clone_if_missing https://github.com/ChenDelong1999/RemoteCLIP.git third_party/RemoteCLIP
clone_if_missing https://github.com/Luo-Z13/SkySense-Chat.git third_party/SkySense-Chat

echo "External repositories are ready in third_party/."
