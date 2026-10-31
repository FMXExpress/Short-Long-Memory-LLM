#!/usr/bin/env bash
# End-to-end test of priors (standing instructions) on the real binary: picked
# up from the user's words, confirmed, contradicted, set and dropped with
# /prior, and present in the system prompt (/prompt). No training: runs in
# seconds once the tiny model is in ./cache.
# Usage: scripts/e2e_priors.sh [profile]     (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-tiny}"
DIR="${E2E_DIR:-$(mktemp -d)}"
rm -rf "$DIR/data"; mkdir -p "$DIR"

cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
engine=${MEMAGENT_ENGINE:-llama}
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
recent_turns=2
[gpu]
infer=-1
train=-1
[consolidate]
auto=0
[retrieval]
cache_dir=$PWD/cache
INI

./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$DIR/transcript.txt"
From now on, keep your answers to one sentence.
Please never use emojis.
I prefer short answers.
Don't use emojis.
/prompt What is the sea?
Actually, you can use emojis.
/prior Call me Eli.
/priors
/prior drop 3
/prompt What is the sea?
/status
/quit
CHAT

T="$DIR/transcript.txt"
fail() { echo "E2E PRIORS FAIL: $1"; exit 1; }
prompt() { awk -v n="$1" '/^you> \/prompt /{c++; next} /^you> /{if(c==n) exit} c==n' "$T"; }
grep -q "^\[prior +#1\] Keep your answers to one sentence\." "$T"   || fail "instruction picked up"
grep -q "^\[prior +#2\] Never use emojis\." "$T"                   || fail "negative instruction picked up"
grep -q "^\[prior +#3\] The user prefers short answers\." "$T"     || fail "preference about answers picked up"
grep -q "^\[prior ~#2 Never use emojis\. (0.75" "$T"              || fail "restating confirms (2:1 -> 3:1)"
prompt 1 | grep -q "^- Never use emojis\."                         || fail "a prior is in the system prompt"
grep -q "^\[prior -#2 Never use emojis\. (0.60" "$T"              || fail "the reverse contradicts (3:1 -> 3:2)"
grep -q "^\[prior +#4\] Use emojis\." "$T"                         || fail "the reverse is a prior of its own"
grep -q "^\[prior +#5\] Call me Eli\." "$T"                        || fail "/prior sets one"
grep -q "#5 Call me Eli\. (0.75, 0 messages, set with /prior)" "$T" || fail "/priors lists a set prior at 3:1"
grep -q "^\[prior #3\] deprecated" "$T"                            || fail "/prior drop"
prompt 2 | grep -q "^- Call me Eli\."                              || fail "a set prior is in the prompt"
prompt 2 | grep -q "prefers short answers"                         && fail "a dropped prior left the prompt"
prompt 2 | grep -q "^- Use emojis\."                               || fail "the newer, more confident of two opposites is in"
prompt 2 | grep -q "Never use emojis"                              && fail "...and the one it contradicts is not"
grep -q "#2 Never use emojis\..*\[not in the prompt\]" "$T"        || fail "/priors says which are left out"
grep -q "^priors: *3 in the prompt, 5 in all" "$T"                 || fail "/status priors line"
echo "E2E PRIORS PASS ($T)"
