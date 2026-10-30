#!/usr/bin/env bash
# Sets up the in-process llama.cpp engine ([agent] engine=llama):
#   1. checks out llama.cpp at the tag memagent/Mem.Llama.pas is bound to
#   2. applies packaging/llama-patches (LoRA training in-process, Mem.Train)
#   3. builds it as shared libraries into cache/llama (RUNPATH $ORIGIN, so the
#      directory can be moved or shipped next to the binary)
#   4. installs the GGUF converters' Python deps and converts each profile's
#      base model to cache/gguf/<profile>.gguf (the agent can also download one)
#
# usage: scripts/llama_setup.sh [profile ...]     (default: tiny)
#   LLAMA_CUDA=1      build the CUDA backend (inference GPU = [gpu] infer)
#   LLAMA_PORTABLE=1  no -march=native, no OpenMP: libraries that run on any x86-64 with AVX2
#                     and need nothing beyond libc/libstdc++ (release builds)
#   LLAMA_LIBS_ONLY=1 stop after the libraries (no Python deps, no model conversion)
#   JOBS=N            parallel build jobs (default: all cores)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG=b11985   # must match LLAMA_TAG in memagent/Mem.Llama.pas
SRC="$REPO/cache/llama.cpp"
LIB="$REPO/cache/llama"
PY="${PYTHON:-python3}"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi
have="$(git -C "$SRC" describe --tags --exact-match 2>/dev/null || echo unknown)"
if [ "$have" != "$TAG" ]; then
  echo "llama_setup: $SRC is at $have, Mem.Llama.pas is bound to $TAG" >&2
  exit 1
fi

# memagent's patches: LoRA training in-process (Mem.Train)
for p in "$REPO"/packaging/llama-patches/*.patch; do
  if git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then
    echo "llama_setup: $(basename "$p") already applied"
  else
    git -C "$SRC" apply "$p"
    echo "llama_setup: applied $(basename "$p")"
  fi
done

cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  ${LLAMA_CUDA:+-DGGML_CUDA=ON} ${LLAMA_PORTABLE:+-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF} >/dev/null
cmake --build "$SRC/build" -j"${JOBS:-$(nproc)}" --target llama
mkdir -p "$LIB"
cp -a "$SRC"/build/bin/lib*.so* "$LIB"/
echo "llama_setup: libraries in $LIB"
[ -n "${LLAMA_LIBS_ONLY:-}" ] && exit 0

"$PY" -m pip install -q sentencepiece "$SRC/gguf-py"
for p in "${@:-tiny}"; do
  if [ -f "$REPO/cache/gguf/$p.gguf" ]; then
    echo "llama_setup: cache/gguf/$p.gguf exists"
  else
    "$PY" "$REPO/trainer/gguf_export.py" base --profile "$p" --llama-src "$SRC"
  fi
done
