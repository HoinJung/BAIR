#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${1:-/home/jw-server3/a/jung414/data}"

mkdir -p "$ROOT/data/raw"

link_one() {
  local src="$1"
  local dest="$2"
  if [[ ! -e "$src" ]]; then
    echo "[missing] $src"
    return
  fi
  if [[ -e "$dest" || -L "$dest" ]]; then
    echo "[skip] $dest already exists"
    return
  fi
  ln -s "$src" "$dest"
  echo "[link] $dest -> $src"
}

link_one "$SRC_ROOT/iuchest" "$ROOT/data/raw/iuchest"
link_one "$SRC_ROOT/mimic-cxr" "$ROOT/data/raw/mimic-cxr"
link_one "$SRC_ROOT/NWPU" "$ROOT/data/raw/NWPU"
link_one "$SRC_ROOT/facet" "$ROOT/data/raw/facet"
