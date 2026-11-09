#!/usr/bin/env bash
# End-to-end test of idle-time training ([consolidate] train_when = idle, the one-gpu-idle
# layout) on the CPU, tiny profile: a run waits for idle time, then unloads the chat model
# and trains; a message stops it and the chat model loads back to answer; /wait trains now.
# Usage: scripts/e2e_idle.sh                (DATA dir: $E2E_DIR or a temp dir)
# LAYOUT=one-gpu-idle PROFILE=small runs it on a card.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${E2E_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"
cat > "$DIR/memagent.ini" <<INI
[agent]
profile=${PROFILE:-tiny}
engine=llama
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
recent_turns=2
[gpu]
layout=${LAYOUT:-cpu}
[consolidate]
trainer=llama
tiers=on
auto=1
train_when=idle
idle_min=0.1
threads=${THREADS:-0}
[retrieval]
cache_dir=$PWD/cache
[llama]
threads=${THREADS:-0}
INI

T="$DIR/transcript.txt"
{
  echo "My dog's name is Biscuit."
  echo "/status"          # the run waits for idle time
  sleep 25                # 6 s of quiet: the run unloads the chat model and trains
  echo "/status"
  echo "What is my dog's name?"   # stops the run, loads the chat model back, answers
  echo "/wait"            # trains now, without waiting for idle time
  echo "/ab What is my dog's name?"
  echo "/quit"
} | ./build/memagent --config "$DIR/memagent.ini" | tee "$T"

fail() { echo "E2E IDLE FAIL: $1"; exit 1; }
grep -q "training on .* when idle" "$T"                     || fail "startup line names idle training"
grep -q "run is waiting for idle time" "$T"                         || fail "/status: the run waits for idle time"
# on a fast device the idle run can finish within the quiet spell: then it must have been promoted
# before the question, and the question is answered by the reloaded chat model
if grep -q "the chat model unloaded for it" "$T"; then
  grep -q "stopping the idle-time .* run to answer" "$T"            || fail "a message stops the idle-time run"
else
  # it finished: the second /status shows nothing running, and the next line promotes it
  grep -q "^sleeping: no" "$T"                                      || fail "/status: the idle run neither unloaded the chat model nor finished"
  echo "(the idle run finished within the quiet spell; stopping a run was not exercised)"
fi
grep -q "^pal> .*" "$T"                                             || fail "the reloaded chat model answers"
grep -q "hot-swapped LoRA none -> mem-" "$T"                        || fail "a LoRA was trained and promoted"
grep -q "^LoRA, no RAG.*[Bb]iscuit" "$T"                            || fail "the LoRA answers from the weights"
echo "E2E IDLE PASS ($T)"
