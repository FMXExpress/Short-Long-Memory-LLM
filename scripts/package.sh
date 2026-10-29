#!/usr/bin/env bash
# Builds a self-contained release folder + tarball:
#
#   dist/memagent-<version>-linux-x64/
#     memagent            the agent (Free Pascal)
#     lib/                libllama + libggml* (llama.cpp b11985, RUNPATH $ORIGIN)
#     memagent.ini        release defaults: engine=auto, CPU, data/ and cache/ in this folder
#     models.json priors.md LICENSE README.md
#     trainer/            the LoRA trainer (Python, optional: long-term memory)
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
NAME="memagent-$VERSION-linux-x64"
OUT="$REPO/dist/$NAME"
SRC="$REPO/cache/llama.cpp"
LIB="${LLAMA_LIB_DIR:-$REPO/cache/llama}"

cd "$REPO"
make all
[ -f "$LIB/libllama.so" ] || { echo "package: no $LIB/libllama.so; run scripts/llama_setup.sh" >&2; exit 1; }
[ -f "$SRC/convert_lora_to_gguf.py" ] || { echo "package: no llama.cpp checkout in $SRC" >&2; exit 1; }

rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT/lib" "$OUT/trainer" "$OUT/llama.cpp"
cp build/memagent "$OUT/"
strip "$OUT/memagent" 2>/dev/null || true
cp -a "$LIB"/lib*.so* "$OUT/lib/"
cp models.json priors.md LICENSE "$OUT/"
cp trainer/memlib.py trainer/worker.py trainer/consolidate.py trainer/gguf_export.py \
   trainer/requirements.txt "$OUT/trainer/"
cp -a "$SRC/convert_lora_to_gguf.py" "$SRC/convert_hf_to_gguf.py" "$SRC/conversion" "$SRC/gguf-py" "$OUT/llama.cpp/"
cp "$SRC/LICENSE" "$OUT/llama.cpp/LICENSE"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +

# release ini: the repo's, with CPU defaults and the bundled converter
sed -e 's/^engine=.*/engine=auto/' -e 's/^infer=.*/infer=-1/' -e 's/^train=.*/train=-1/' \
    -e 's#^;src_dir=.*#src_dir=llama.cpp#' memagent.ini > "$OUT/memagent.ini"
cp packaging/README.release.md "$OUT/README.md"

tar -C "$REPO/dist" -czf "$OUT.tar.gz" "$NAME"
echo "package: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
