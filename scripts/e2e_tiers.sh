#!/usr/bin/env bash
# End-to-end test of the memory tiers on the llama engine (tiny profile, CPU):
# a deep sleep bakes three facts, then two new facts go into a fast LoRA that
# trains on top of the frozen deep one, and both tiers answer from the weights.
# Usage: scripts/e2e_tiers.sh [profile]     (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-tiny}"
DIR="${E2E_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"

cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
engine=llama
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
recent_turns=2
[gpu]
infer=-1
train=-1
[consolidate]
trainer=llama
tiers=on
auto=1
interval_min=60
deep_after=8
threads=${THREADS:-0}
[retrieval]
cache_dir=$PWD/cache
[llama]
${LLAMA_LIB_DIR:+lib_dir=$LLAMA_LIB_DIR}
threads=${THREADS:-0}
INI

./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$DIR/transcript.txt"
My dog's name is Biscuit.
I live in Reykjavik.
Remember that the wifi password is tulip42.
/sleep
/wait
My sister's name is Ana.
My brother's name is Leo.
/wait
/facts
/ab What's my sister's name?
/ab What is my dog's name?
/status
/quit
CHAT

T="$DIR/transcript.txt"
fail() { echo "E2E TIERS FAIL: $1"; exit 1; }
lora() { grep "^LoRA, no RAG" "$T" | sed -n "${1}p"; }
grep -q "memory tiers: a fast LoRA" "$T"                       || fail "tiers not enabled (rebuild llama.cpp: scripts/llama_setup.sh)"
grep -q "deep sleep (/sleep)" "$T"                            || fail "/sleep starts a deep sleep"
grep -q "hot-swapped LoRA none -> mem-" "$T"                  || fail "deep LoRA promoted"
grep -q "fast LoRA .* -> fast-" "$T"                          || fail "fast LoRA trained on top of the deep one and promoted"
grep -q "baked: fast-" "$T"                                   || fail "new facts marked as in the fast LoRA"
grep -q "baked: mem-" "$T"                                    || fail "older facts still in the deep LoRA"
lora 1 | grep -qi "ana"                                       || fail "fast tier: sister from the weights"
lora 2 | grep -qi "biscuit"                                   || fail "deep tier kept: dog from the weights"
grep -q "^tiers: *fast LoRA holds 2 facts" "$T"               || fail "/status tier line"
echo "E2E TIERS PASS ($T)"
