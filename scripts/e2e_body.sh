#!/usr/bin/env bash
# End-to-end test of the body layer on the real binary, twice: the in-process
# mock driver, and body/lerobot_body.py --stub through the process protocol.
# Covers /do + /confirm + /cancel, a task refused by a rule, a subtask stopped
# by a rule, a body-only prior kept out of the chat prompt, and "put X in Y"
# steps becoming facts. No training, no robot: runs in seconds.
# Usage: scripts/e2e_body.sh [profile]     (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-tiny}"
BASE="${E2E_DIR:-$(mktemp -d)}"

run() {   # $1 = label, $2 = [body] lines
  local DIR="$BASE/$1"
  rm -rf "$DIR"; mkdir -p "$DIR"
  cat > "$DIR/memagent.ini" <<INI
[agent]
profile=$PROFILE
engine=${MEMAGENT_ENGINE:-llama}
models=$PWD/models.json
priors=$PWD/priors.md
data_dir=$DIR/data
repo_dir=$PWD
python=${PYTHON:-python3}
recent_turns=2
[gpu]
infer=-1
train=-1
[consolidate]
auto=0
[retrieval]
cache_dir=$PWD/cache
[body]
$2
INI
  ./build/memagent --config "$DIR/memagent.ini" <<'CHAT' | tee "$DIR/transcript.txt"
Never put the cast-iron pan in the dishwasher.
/prior --body Don't touch the knives.
/do put the mug in the left cabinet, then put the keys on the hall table
/confirm
/do wipe the counter
/cancel
/do put the cast-iron pan in the dishwasher
/do tidy the kitchen
/confirm
/facts
/prompt Where is my mug?
/status
/quit
CHAT
  local T="$DIR/transcript.txt"
  fail() { echo "E2E BODY FAIL ($1): $2"; exit 1; }
  grep -q "^body: " "$T"                                                   || fail "$1" "body started"
  grep -q "^\[prior +#1\] Never put the cast-iron pan in the dishwasher\." "$T" || fail "$1" "a rule from the chat"
  grep -q "^\[prior +#2\] Don't touch the knives\.  (body)" "$T"          || fail "$1" "/prior --body"
  grep -q "will: put the mug in the left cabinet, then put the keys on the hall table (2 rule(s) apply)" "$T" \
                                                                           || fail "$1" "/do asks first"
  grep -q "^\[body\] subtask: put the mug in the left cabinet" "$T"       || fail "$1" "first subtask"
  grep -q "^\[body\] subtask: put the keys on the hall table" "$T"        || fail "$1" "second subtask"
  grep -q "^\[memory +#[0-9]*\] The mug is in the left cabinet\." "$T"    || fail "$1" "a step became a fact"
  grep -q "^\[memory +#[0-9]*\] The keys are on the hall table\." "$T" \
                                                                           || fail "$1" "second step became a fact"
  grep -q "^\[body\] dropped: wipe the counter" "$T"                      || fail "$1" "/cancel"
  grep -q "^\[body\] refused: \"put the cast-iron pan in the dishwasher\" breaks prior #1" "$T" \
                                                                           || fail "$1" "a task that breaks a rule is refused"
  grep -q "^\[body\] stopping: \"put the cast-iron pan in the dishwasher\" breaks prior #1" "$T" \
                                                                           || fail "$1" "a subtask that breaks a rule stops the body"
  grep -q "^\[body\] subtask: wipe the counter" "$T"                      && fail "$1" "nothing runs after the stop"
  grep -q "^\[body\] stopped: " "$T"                                      || fail "$1" "the stop is reported"
  grep -q "The plate is" "$T"                                             && fail "$1" "a stopped act stores no facts for it"
  awk '/^you> \/prompt /{p=1; next} /^you> /{p=0} p' "$T" | grep -q "touch the knives" \
                                                                           && fail "$1" "a body-only rule stays out of the chat prompt"
  awk '/^you> \/prompt /{p=1; next} /^you> /{p=0} p' "$T" | grep -q "^- Never put the cast-iron pan" \
                                                                           || fail "$1" "a chat + body rule is in the chat prompt"
  grep -q "^body: .*2 rule(s) apply" "$T"                                 || fail "$1" "/status body line"
  echo "E2E BODY PASS ($1: $T)"
}

# "tidy the kitchen" stands in for a goal a planning policy decomposes on its own
PLAN="tidy the kitchen => pick up the plate; put the cast-iron pan in the dishwasher; wipe the counter"
run mock "driver=mock
args=$PLAN
confirm=1"
run lerobot-stub "driver=lerobot
args=--stub --stub-delay 0.05 --stub-plans '$PLAN'
confirm=1"
