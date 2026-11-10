#!/usr/bin/env bash
# End-to-end test of a chat-only profile (models.json "trainer": "none", e.g. qwen3.8-27b) on the CPU,
# with the tiny model made chat-only: facts are stored and retrieved, nothing ever trains, and
# /sleep says why. Usage: scripts/e2e_chatonly.sh          (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${E2E_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"
python3 -c "
import json; d = json.load(open('models.json'))
d['profiles']['tiny']['trainer'] = 'none'
json.dump(d, open('$DIR/models.json', 'w'), indent=1)"
cat > "$DIR/memagent.ini" <<INI
[agent]
profile=tiny
engine=llama
models=$DIR/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
recent_turns=2
[gpu]
layout=cpu
[consolidate]
auto=1
min_new_facts=1
interval_min=0
[retrieval]
cache_dir=$PWD/cache
[llama]
threads=${THREADS:-0}
INI

T="$DIR/transcript.txt"
./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$T"
My dog's name is Biscuit.
I live in Reykjavik.
/sleep
/nap
/facts
/status
/quit
CHAT

fail() { echo "E2E CHATONLY FAIL: $1"; exit 1; }
grep -q "no training (a chat-only profile)" "$T"            || fail "startup layout line says chat-only"
grep -q "^training: off (profile tiny is chat-only" "$T"    || fail "startup says training is off"
grep -q "memory +#1\] Your dog's name is Biscuit" "$T"      || fail "facts are still stored"
[ "$(grep -c 'training is off: profile tiny is chat-only' "$T")" = 2 ] || fail "/sleep and /nap say training is off"
grep -q "deep sleep\|\[nap\]\|consolidating" "$T"            && fail "nothing may start training"
[ -d "$DIR/data/adapters" ] && [ -n "$(ls -A "$DIR/data/adapters" 2>/dev/null)" ] && fail "no adapter may be written"
grep -q "Biscuit" <(sed -n '/^you> \/facts/,/^you> \/status/p' "$T") || fail "/facts lists the stored fact"
echo "E2E CHATONLY PASS ($T)"
