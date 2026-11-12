#!/usr/bin/env bash
# Builds a self-contained release folder + tarball:
#
#   dist/memagent-<version>-linux-x64[-cuda]/
#     memagent            the agent (Free Pascal)
#     lib/                libllama + libggml* (llama.cpp b11985, RUNPATH $ORIGIN); with a CUDA
#                         build (LLAMA_CUDA=1) also libggml-cuda and the CUDA runtime and cuBLAS
#                         it needs, so only the NVIDIA driver has to be installed (-cuda name)
#     memagent.ini        release defaults: engine=auto, CPU, data/ and cache/ in this folder
#     models.json priors.md LICENSE README.md
#     trainer/            the LoRA trainer (Python, optional: long-term memory)
#     body/               the LeRobot body driver (Python, optional: [body] driver=lerobot)
#     llama.cpp/          the GGUF converters the trainer uses (MIT, from llama.cpp)
#
# Run ./memagent: the first start downloads the model (270 MB for tiny) and the
# retrieval models into cache/, then chats. Training needs Python with
# trainer/requirements.txt; without it the agent runs on short-term memory.
#
# usage: scripts/package.sh [version]      (default: dev)
#   needs: make (fpc), and llama.cpp built by scripts/llama_setup.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-dev}"
SRC="$REPO/cache/llama.cpp"
LIB="${LLAMA_LIB_DIR:-$REPO/cache/llama}"
CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
SUFFIX=""
compgen -G "$LIB/libggml-cuda.so*" >/dev/null && SUFFIX="-cuda"
NAME="memagent-$VERSION-linux-x64$SUFFIX"
OUT="$REPO/dist/$NAME"

cd "$REPO"
make all
[ -f "$LIB/libllama.so" ] || { echo "package: no $LIB/libllama.so; run scripts/llama_setup.sh" >&2; exit 1; }
[ -f "$SRC/convert_lora_to_gguf.py" ] || { echo "package: no llama.cpp checkout in $SRC" >&2; exit 1; }

rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/lib" "$OUT/trainer" "$OUT/body" "$OUT/llama.cpp"
cp build/memagent "$OUT/"
cp scripts/run.sh "$OUT/"   # one-command launcher: best settings for the machine
strip "$OUT/memagent" 2>/dev/null || true
cp -a "$LIB"/lib*.so* "$OUT/lib/"
if [ -n "$SUFFIX" ]; then
  # the CUDA runtime libraries libggml-cuda links (its RUNPATH $ORIGIN finds them in lib/);
  # libcuda.so.1 is the driver's and stays the system's
  for l in libcudart libcublas libcublasLt; do
    compgen -G "$CUDA_LIB/$l.so.*" >/dev/null || { echo "package: no $l in $CUDA_LIB (set CUDA_HOME)" >&2; exit 1; }
    cp -a "$CUDA_LIB"/$l.so.* "$OUT/lib/"
  done
  missing=$(cd "$OUT/lib" && LD_LIBRARY_PATH= ldd ./libggml-cuda.so | awk '/not found/ {print $1}' | grep -v '^libcuda\.so' || true)
  [ -z "$missing" ] || { echo "package: libggml-cuda.so needs $missing" >&2; exit 1; }
fi
cp models.json priors.md LICENSE "$OUT/"
cp trainer/memlib.py trainer/worker.py trainer/consolidate.py trainer/gguf_export.py \
   trainer/requirements.txt "$OUT/trainer/"
cp body/lerobot_body.py body/README.md "$OUT/body/"
cp -a "$SRC/convert_lora_to_gguf.py" "$SRC/convert_hf_to_gguf.py" "$SRC/conversion" "$SRC/gguf-py" "$OUT/llama.cpp/"
cp "$SRC/LICENSE" "$OUT/llama.cpp/LICENSE"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +

# release ini: the repo's, with CPU defaults and the bundled converter
sed -e 's/^engine=.*/engine=auto/' -e 's/^infer=.*/infer=-1/' -e 's/^train=.*/train=-1/' \
    -e 's#^;src_dir=.*#src_dir=llama.cpp#' memagent.ini > "$OUT/memagent.ini"
cp packaging/README.release.md "$OUT/README.md"

tar -C "$REPO/dist" -czf "$OUT.tar.gz" "$NAME"
echo "package: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
