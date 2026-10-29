#!/usr/bin/env bash
# Smoke test for the profiles above `tiny`: the code paths tiny turns off
# (LLM fact extraction, rolling summaries, tool calls), then optionally one
# consolidation. Usage: scripts/profile_smoke.sh <profile> [--train]
# GPUs: INFER_GPU / TRAIN_GPU (default -1 = CPU).
set -euo pipefail
cd "$(dirname "$0")/.."
export MEMAGENT_TEMPERATURE="${MEMAGENT_TEMPERATURE:-0}"   # greedy, so the run is repeatable
PROFILE="${1:?profile}"
DIR="${SMOKE_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"
cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
python=${PYTHON:-python3}
recent_turns=1
[gpu]
infer=${INFER_GPU:--1}
train=${TRAIN_GPU:--1}
[consolidate]
auto=0
[retrieval]
cache_dir=$PWD/cache
INI
./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$DIR/transcript.txt"
Hi! I'm planning a trip and could use your help over the next few messages.
My sister Ana just moved to Lisbon, she works there as a nurse.
Please use your remember tool to store this: my passport expires in March 2027.
What time is it right now? Use your tool.
We will visit Ana in Lisbon in May, and I want to see the Belem tower and eat pasteis de nata.
/facts
/recall Where does my sister live?
/recall When does my passport expire?
/status
/quit
CHAT
T="$DIR/transcript.txt"
fail() { echo "SMOKE FAIL: $1"; exit 1; }
grep -qiE "memory \+#.*(Lisbon|nurse)" "$T"   || fail "LLM fact extraction (no rule matches the Lisbon sentence)"
grep -q "^\[tool\] " "$T"                     || fail "no tool call was made"
grep -A3 "^you> /recall Where does my sister live" "$T" | grep -q "fact #.*Lisbon" || fail "fact recall by relation (sister -> Lisbon)"
python3 -c "import sqlite3,sys; r=sqlite3.connect(sys.argv[1]).execute('select text from summary').fetchall(); sys.exit(0 if r and r[0][0].strip() else 1)" \
  "$DIR/data/memory.db" || fail "no rolling summary"
echo "summary: $(python3 -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('select text from summary').fetchone()[0][:300])" "$DIR/data/memory.db")"
if [ "${2:-}" = "--train" ]; then
  CUDA_VISIBLE_DEVICES="$( [ "${TRAIN_GPU:--1}" = -1 ] && echo '' || echo "$TRAIN_GPU")" \
    python3 trainer/consolidate.py --db "$DIR/data/memory.db" --out "$DIR/data/adapters" \
      --profile "$PROFILE" --priors priors.md 2>&1 | grep -E "^\[consolidate\]|^\[probe\]|^\{" | tee -a "$T"
  tail -1 "$T" | grep -q '"exit": 0' || fail "consolidation did not promote"
fi
echo "SMOKE PASS ($T)"
