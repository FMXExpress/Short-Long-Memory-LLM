#!/usr/bin/env bash
# End-to-end demo/test: drives the real memagent binary with a scripted
# conversation on the tiny profile (CPU only), through two consolidations.
# Usage: scripts/e2e_demo.sh [profile]     (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-tiny}"
DIR="${E2E_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"

cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
python=${PYTHON:-python3}
recent_turns=2
rag_k=3
[gpu]
infer=${INFER_GPU:--1}
train=${TRAIN_GPU:--1}
[consolidate]
auto=1
min_new_facts=3
interval_min=0
INI

./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$DIR/transcript.txt"
Hi! I am testing your memory today.
My dog's name is Biscuit.
I live in Reykjavik.
Remember that the wifi password is tulip42.
What is my dog's name?
/wait
/facts
/ab Do you know what my dog's name is?
/ab Do you know what the wifi password is?
My dog's name is Max.
/forget 3
/sleep
/wait
/facts
/ab What's my dog's name?
/ab Do you know what the wifi password is?
/adapter
/rollback
/ab What's my dog's name?
/adapter
/status
/quit
CHAT

# Assertions on the transcript
T="$DIR/transcript.txt"
fail() { echo "E2E FAIL: $1"; exit 1; }
grep -q "memory +#1\] Your dog's name is Biscuit" "$T"       || fail "fact extraction"
grep -q "hot-swapped LoRA none -> mem-" "$T"                  || fail "first consolidation/hot swap"
grep -q "baked: mem-" "$T"                                     || fail "facts marked baked"
lora() { grep "^LoRA, no RAG" "$T" | sed -n "${1}p"; }
lora 1 | grep -qi "biscuit"                                    || fail "LoRA recall of Biscuit without RAG"
lora 2 | grep -qi "tulip42"                                    || fail "LoRA recall of the wifi password"
grep -q "memory ~#.*Max.*replaces" "$T"                       || fail "supersede"
[ "$(grep -c 'hot-swapped LoRA' "$T")" -ge 3 ]                 || fail "second consolidation + rollback"
lora 3 | grep -qi "max"                                        || fail "LoRA learned the corrected value"
lora 4 | grep -qi "tulip42"                                    && fail "forgotten fact still in the weights"
lora 5 | grep -qi "biscuit"                                    || fail "rollback restored the older adapter"
grep -q "rolledback" "$T"                                      || fail "rollback recorded in adapter history"
echo "E2E PASS ($T)"
