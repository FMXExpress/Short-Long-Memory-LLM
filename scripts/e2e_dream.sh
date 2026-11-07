#!/usr/bin/env bash
# End-to-end test of dreaming (TAgent.Dream) on the real binary: a planted wrong fact,
# one whose message never said it, is set aside, the true facts survive, /review lists
# it and /keep brings it back. No judge (the CPU path: the value must be in the message).
# Usage: scripts/e2e_dream.sh [profile]     (DATA dir: $E2E_DIR or a temp dir)
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-tiny}"
DIR="${E2E_DIR:-$(mktemp -d)}/dream"
rm -rf "$DIR"; mkdir -p "$DIR"
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
fail() { echo "E2E DREAM FAIL: $1"; exit 1; }
./build/memagent --config "$DIR/memagent.ini" > "$DIR/one.txt" <<'CHAT'
My sister's name is Ana.
My dog's name is Biscuit.
I live in Reykjavik.
/quit
CHAT
# the planted error: a fact that claims to come from the first message, which never said it
python3 - "$DIR/data/memory.db" <<'PY'
import sqlite3, sys, time
db = sqlite3.connect(sys.argv[1])
src = db.execute("SELECT id FROM messages WHERE content = ? ORDER BY id LIMIT 1", ("My sister's name is Ana.",)).fetchone()[0]
cur = db.execute("INSERT INTO facts(text, subject, value, questions, kind, confidence, source_msg, created_at)"
                 " VALUES(?,?,?,?,?,?,?,?)", ("Your brother's name is Bob.", "my brother's name", "Bob", "[]",
                                              "user", 0.9, src, int(time.time())))
db.execute("INSERT INTO facts_fts(rowid, text, subject) VALUES(?,?,?)", (cur.lastrowid, "Your brother's name is Bob.",
                                                                        "my brother's name"))
db.commit()
print("planted #%d" % cur.lastrowid)
PY
./build/memagent --config "$DIR/memagent.ini" > "$DIR/two.txt" <<'CHAT'
/recall What is my brother's name?
/dream
/review
/recall What is my brother's name?
/facts
/keep 4
/facts
/quit
CHAT
T="$DIR/two.txt"
# the output of the k-th use of a command, up to the next prompt
sect() { awk -v c="$1" -v k="$2" 'index($0, "you> " c) == 1 {n++; p = (n == k); next} /^you> /{p = 0} p' "$T"; }
sect /recall 1 | grep -q "Your brother's name is Bob" || fail "the planted fact is in memory before"
grep -q "^\[dream\] set aside #4 \"Your brother's name is Bob\.\": message #[0-9]* does not say it" "$T" \
                                                                 || fail "the planted fact is set aside"
grep -q "^\[dream\] checked 4 fact(s) against their messages: 1 set aside, 0 duplicate(s) merged" "$T" \
                                                                 || fail "the summary"
grep -q "^  #4   Your brother's name is Bob\.   (/keep 4 brings it back)" "$T" || fail "/review lists it"
sect /recall 2 | grep -q "Bob" && fail "a set-aside fact is out of retrieval"
sect /facts 1 | grep -q "Your sister's name is Ana" || fail "true facts survive"
sect /facts 1 | grep -q "Biscuit" || fail "true facts survive (2)"
sect /facts 1 | grep -q "Bob" && fail "not in /facts while set aside"
grep -q "^\[memory\] #4 is back in memory" "$T"                  || fail "/keep"
sect /facts 2 | grep -q "Your brother's name is Bob" || fail "kept fact is back"
echo "E2E DREAM PASS ($T)"
