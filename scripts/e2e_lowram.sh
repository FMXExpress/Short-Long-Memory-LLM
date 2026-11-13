#!/usr/bin/env bash
# Training on a small machine: memagent under a memory cap picks the model as loaded
# for training (an f32 copy would not fit beside the chat model), finishes a deep sleep,
# and answers from the weights alone.
#
#   scripts/e2e_lowram.sh [profile]          (default qwen3-0.6b; needs make all)
#
# Run it under a cap, or it only checks the plan the machine's own RAM gives:
#   docker run --memory 3g ...  scripts/e2e_lowram.sh
#   systemd-run --user --scope -p MemoryMax=3G scripts/e2e_lowram.sh
# E2E_TRAIN_BASE=f32 forces the f32 copy instead, for comparison under the same cap.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-qwen3-0.6b}"
DIR="${E2E_DIR:-$(mktemp -d)}/lowram"
rm -rf "$DIR"; mkdir -p "$DIR"
cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
engine=llama
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
[gpu]
layout=cpu
[consolidate]
auto=0
train_base=${E2E_TRAIN_BASE:-auto}
[retrieval]
cache_dir=$PWD/cache
INI
fail() { echo "E2E LOWRAM FAIL: $1"; tail -30 "$DIR/out.txt"; exit 1; }
start=$(date +%s)
./build/memagent --config "$DIR/memagent.ini" > "$DIR/out.txt" 2>&1 <<'CHAT' || true
My dog's name is Biscuit.
My sister's name is Ana.
I live in Reykjavik.
My favorite color is teal.
My boss is Dr. Okafor.
/sleep
/wait
/weights What is my dog's name?
/weights What is my sister's name?
/weights Where do I live?
/weights What is my favorite color?
/weights Who is my boss?
/quit
CHAT
secs=$(( $(date +%s) - start ))
grep "^layout" "$DIR/out.txt" || fail "no layout line"
if [ -z "${E2E_TRAIN_BASE:-}" ] || [ "$E2E_TRAIN_BASE" = auto ]; then
  if grep -q "^layout.*the model as loaded.*in RAM" "$DIR/out.txt"; then plan=model
  elif grep -q "^layout.*an f32 copy.*fits" "$DIR/out.txt"; then plan=f32
  else fail "the plan names neither choice"; fi
else
  plan=$E2E_TRAIN_BASE
fi
grep -q "hot-swapped LoRA" "$DIR/out.txt" || fail "the deep sleep did not finish (killed for memory?)"
hits=0
for v in Biscuit Ana Reykjavik teal Okafor; do
  grep -q "^weights> .*$v" "$DIR/out.txt" && hits=$((hits + 1))
done
grep "^weights>" "$DIR/out.txt"
echo "E2E LOWRAM: plan=$plan, deep sleep done, $hits/5 answered from the weights, ${secs}s ($DIR)"
[ "$hits" -ge 3 ] || fail "only $hits/5 facts answered from the weights"
