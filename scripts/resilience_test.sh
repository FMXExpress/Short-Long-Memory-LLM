#!/usr/bin/env bash
# Resilience checks on top of an e2e run (pass its data dir):
#   A  restart: a new process reloads the active adapter and the facts
#   B  worker crash: the agent restarts the Python worker and re-applies the LoRA
#   C  no retrieval models: falls back to the static embedder and still recalls
#   D  /quit during a consolidation waits for it and promotes the adapter
# Usage: scripts/resilience_test.sh <e2e dir containing memagent.ini and data/>
set -euo pipefail
cd "$(dirname "$0")/.."
E2E="$1"
W=$(mktemp -d)
fails=0
ok()   { echo "  ok    $1"; }
bad()  { echo "  FAIL  $1"; fails=$((fails+1)); }

# drive memagent through a FIFO so we can act between lines
start() {  # ini, logfile
  mkfifo "$W/in"
  ./build/memagent --config "$1" < "$W/in" > "$2" 2>&1 &
  AGENT=$!
  exec 3>"$W/in"
}
send() { echo "$1" >&3; }
waitfor() {  # pattern, file, seconds
  local t=0
  until grep -q -- "$1" "$2"; do
    sleep 1; t=$((t+1))
    [ $t -ge "$3" ] && return 1
  done
  return 0   # an until loop returns its last body command's status, not the condition's
}
stop() { send /quit; exec 3>&-; wait $AGENT || true; rm -f "$W/in"; }

echo "A/B: restart + worker crash (data from $E2E)"
L="$W/ab.log"
start "$E2E/memagent.ini" "$L"
ACTIVE=$(grep -o 'hot-swapped LoRA [^ ]* -> mem-[0-9T]*' "$E2E/transcript.txt" | tail -1 | awk '{print $NF}')
waitfor "long-term adapter: $ACTIVE" "$L" 120 && ok "restart reloads the active adapter ($ACTIVE, the rollback target)" \
  || bad "restart reloads the active adapter"
sleep 2
grep -q "hot-swapped" "$L" && bad "restart re-promoted an adapter the user rolled back from" || ok "a rollback survives a restart"
send "/facts"
waitfor "baked: mem-" "$L" 30 && ok "facts and their baked state persist" || bad "facts persist"
send "/ab What's my dog's name?"
waitfor "^LoRA, no RAG.*Biscuit" "$L" 120 && ok "reloaded LoRA answers from the weights" || bad "reloaded LoRA answers"
pkill -f "worker.py.*$E2E/data/logs/worker.log" || true
sleep 2
send "/ab What's my dog's name?"
waitcount() {  # pattern, file, count, seconds
  local t=0
  until [ "$(grep -c -- "$1" "$2")" -ge "$3" ]; do
    sleep 1; t=$((t+1))
    [ $t -ge "$4" ] && return 1
  done
  return 0
}
if waitfor "worker is not running, restarting" "$L" 60 && waitcount "^LoRA, no RAG.*Biscuit" "$L" 2 180; then
  ok "killed worker is restarted with the LoRA re-applied"
else
  bad "worker restart"
fi
stop

echo "C: no retrieval models -> static embedder"
sed -e "s|^data_dir=.*|data_dir=$W/c/data|" -e "s|^cache_dir=.*|cache_dir=$W/empty-cache|" "$E2E/memagent.ini" > "$W/c.ini"
L="$W/c.log"
start "$W/c.ini" "$L"
waitfor "lexical fallback" "$L" 120 && ok "falls back to the static embedder" || bad "static fallback"
send "My dog's name is Biscuit."
send "/recall dog name"
waitfor "fact #1 .*Biscuit" "$L" 120 && ok "static tier still recalls" || bad "static recall"
stop

echo "D: /quit during a consolidation"
sed -e "s|^data_dir=.*|data_dir=$W/d/data|" "$E2E/memagent.ini" > "$W/d.ini"
L="$W/d.log"
start "$W/d.ini" "$L"
send "/remember My cat's name is Pixel."
send "/sleep"
waitfor "consolidating" "$L" 120 || true
stop
grep -q "waiting for the running consolidation" "$L" && grep -q "hot-swapped LoRA none -> mem-" "$L" \
  && ok "quit waits for training and promotes the adapter" || bad "quit during consolidation"

[ $fails = 0 ] && { echo "all resilience checks passed"; rm -rf "$W"; } || { echo "$fails resilience check(s) failed; logs in $W"; exit 1; }
