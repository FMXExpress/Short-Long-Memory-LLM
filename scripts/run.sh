#!/usr/bin/env bash
# One command to run memagent at the best settings for this machine.
#
#   scripts/run.sh                 detect CPU, RAM and GPUs, pick a preset, set up, run
#   scripts/run.sh gpu12           force a preset (list: scripts/run.sh --list)
#   scripts/run.sh --dry-run       show the preset and the config, change nothing
#   scripts/run.sh auto -- --offline     anything after -- goes to memagent
#
# Works in a clone (scripts/run.sh) and in an extracted release (./run.sh). In a clone it
# builds memagent and llama.cpp (CUDA when there is a GPU and nvcc) if they are missing.
# It installs the Python trainer when the preset trains in Python, writes
# run/<preset>.ini, and starts memagent with its memory in data/<profile>/. Without a
# CUDA llama.cpp (a release, or no nvcc) a GPU preset answers through the Python engine.
# Windows: scripts\run.cmd (or run.cmd in a release) does the same.
# Training runs in the background on its own ([consolidate] auto=1): a fast LoRA after
# each new fact, a deep sleep after 8.
#
# Why these presets (bench/README.md, "One-shot presets"):
#   - retrieval does most of the work at every size; retrieving twice ([retrieval]
#     hops=2) took the 1.5B agent from 40 to 58 of 96 integration questions.
#   - the trained memory paid off from 4B up (bridge questions 4 -> 20 of 24 at 4B,
#     3 -> 23 at 24B). At 1.5B it lost in the agent (32/96 against 58/96 for retrieval),
#     so the 1.5B presets do not train in the background ([consolidate] auto=0).
#   - the Python trainer on a GPU learns 112 facts in 2 min (1.5B) to 15 min (24B);
#     llama.cpp on 4 CPU cores took 70 min, so CPU presets train on few cores, slowly.
#   - VRAM (measured on an RTX A6000): 24B Q4 inference ~14 GB + 16k context ~5 GB,
#     24B QLoRA training 23.7 GB. 4B and 1.5B are estimates from their sizes.
set -euo pipefail
# a release has the binary next to this script; a clone has scripts/ under the repo
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -x "$HERE/memagent" ]; then ROOT=$HERE; BIN=$ROOT/memagent; CLONE=0
else ROOT=$(cd "$HERE/.." && pwd); BIN=$ROOT/build/memagent; CLONE=1; fi
cd "$ROOT"

PRESETS="cpu-lite cpu gpu8 gpu12 gpu40 2gpu"
describe() {
  case $1 in
    cpu-lite) echo "tiny (SmolLM2-135M) on CPU; trains on 1 core. Under 4 cores or 8 GB RAM: a demo, not a useful memory." ;;
    cpu)      echo "small (Qwen2.5-1.5B) on CPU, retrieval only: at 1.5B trained memory scored 32/96 against 58/96 for retrieval." ;;
    gpu8)     echo "small (Qwen2.5-1.5B) on one 8-11 GB GPU, retrieval only: at 1.5B trained memory scored 32/96 against 58/96 for retrieval." ;;
    gpu12)    echo "medium (Qwen3-4B) on one 12-39 GB GPU, answering and training on it: the smallest size where trained memory helped." ;;
    gpu40)    echo "large (Magistral-Small 24B) on one 40 GB+ GPU, answering and training on it (~43 GB at peak)." ;;
    2gpu)     echo "two GPUs: answer on GPU 0, train on GPU 1; large if both have 24 GB+, else medium if 12 GB+, else small." ;;
  esac
}

PRESET=auto DRY=0 EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --list) for p in $PRESETS; do printf '  %-9s %s\n' "$p" "$(describe "$p")"; done; exit 0 ;;
    --dry-run) DRY=1 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PRESET=$1 ;;
  esac
  shift
done

# ---------------------------------------------------------------- the machine
CORES=$(nproc 2>/dev/null || echo 2)
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 4)
GPU_MB=()
if command -v nvidia-smi >/dev/null 2>&1; then
  while read -r mb; do [ -n "$mb" ] && GPU_MB+=("$mb"); done < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
fi
NGPU=${#GPU_MB[@]}
MIN_GB=0
if [ "$NGPU" -gt 0 ]; then
  MIN_GB=$(printf '%s\n' "${GPU_MB[@]}" | sort -n | head -1 | awk '{printf "%d", $1/1024}')
fi
echo "[run] machine: $CORES cores, $RAM_GB GB RAM, $NGPU NVIDIA GPU(s)$( [ "$NGPU" -gt 0 ] && echo ", smallest ${MIN_GB} GB")"

if [ "$PRESET" = auto ]; then
  if [ "$NGPU" -ge 2 ] && [ "$MIN_GB" -ge 7 ]; then PRESET=2gpu
  elif [ "$NGPU" -ge 1 ] && [ "$MIN_GB" -ge 40 ]; then PRESET=gpu40
  elif [ "$NGPU" -ge 1 ] && [ "$MIN_GB" -ge 12 ]; then PRESET=gpu12
  elif [ "$NGPU" -ge 1 ] && [ "$MIN_GB" -ge 7 ]; then PRESET=gpu8
  elif [ "$CORES" -ge 4 ] && [ "$RAM_GB" -ge 8 ]; then PRESET=cpu
  else PRESET=cpu-lite
  fi
fi
case " $PRESETS " in *" $PRESET "*) ;; *) echo "unknown preset '$PRESET' (have: $PRESETS)" >&2; exit 2 ;; esac

# ---------------------------------------------------------------- the preset
TRAIN_THREADS=1 INFER=-1 TRAIN=-1 TRAINER=llama GPU_BUILD=0 ENGINE=llama AUTO=1
case $PRESET in
  cpu-lite) PROFILE=tiny ;;
  cpu)      PROFILE=small; AUTO=0; TRAIN_THREADS=$(( CORES / 4 > 0 ? CORES / 4 : 1 )) ;;
  gpu8)     PROFILE=small;  AUTO=0; INFER=0; TRAIN=0; TRAINER=python; GPU_BUILD=1 ;;
  gpu12)    PROFILE=medium; INFER=0; TRAIN=0; TRAINER=python; GPU_BUILD=1 ;;
  gpu40)    PROFILE=large;  INFER=0; TRAIN=0; TRAINER=python; GPU_BUILD=1 ;;
  2gpu)
    INFER=0; TRAIN=1; TRAINER=python; GPU_BUILD=1
    if [ "$MIN_GB" -ge 24 ]; then PROFILE=large; elif [ "$MIN_GB" -ge 12 ]; then PROFILE=medium; else PROFILE=small; fi ;;
esac
# answering on the GPU needs a CUDA llama.cpp: one already built, or nvcc to build it in a
# clone. Otherwise the Python engine (PyTorch, CUDA) answers on the GPU instead.
cuda_llama() { ls "$ROOT"/cache/llama/libggml-cuda.so* "$ROOT"/lib/libggml-cuda.so* >/dev/null 2>&1; }
if [ "$GPU_BUILD" = 1 ] && ! cuda_llama && { [ "$CLONE" = 0 ] || ! command -v nvcc >/dev/null 2>&1; }; then
  ENGINE=python
fi
echo "[run] preset $PRESET: $(describe "$PRESET")"
[ "$AUTO" = 0 ] && echo "[run] background training off for $PROFILE (it scored below retrieval); /sleep trains on demand, auto=1 in run/$PRESET.ini turns it on"
echo "[run] profile $PROFILE, answer on $( [ "$INFER" -ge 0 ] && echo "GPU $INFER" || echo CPU), train on $( [ "$TRAIN" -ge 0 ] && echo "GPU $TRAIN" || echo "CPU ($TRAIN_THREADS thread(s))") with the $TRAINER trainer; $ENGINE engine"

INI=run/$PRESET.ini
TMP=$(mktemp)
cat > "$TMP" <<EOF
; written by scripts/run.sh for preset $PRESET ($(date -u +%Y-%m-%dT%H:%MZ)); rerun it to regenerate.
; Paths are relative to this file. Settings not listed take memagent's defaults (memagent.ini).
[agent]
profile=$PROFILE
engine=$ENGINE
models=../models.json
priors=../priors.md
data_dir=../data/$PROFILE
repo_dir=..
rag_k=3
auto_provision=1

[llama]
threads=0
gpu_layers=-1
$( [ -d llama.cpp ] && echo 'src_dir=../llama.cpp' || echo '; src_dir: the converters in cache/llama.cpp' )

[gpu]
infer=$INFER
train=$TRAIN

[consolidate]
trainer=$TRAINER
threads=$TRAIN_THREADS
; 0 at 1.5B: trained memory lost to retrieval there (bench/README.md); /sleep still trains on demand
auto=$AUTO
tiers=auto

[retrieval]
cache_dir=../cache
; retrieve again for each name in the first hits: two-fact questions get both facts
hops=2
EOF

if [ "$DRY" = 1 ]; then
  echo "[run] --dry-run: would write $INI:"; sed 's/^/    /' "$TMP"; rm -f "$TMP"; exit 0
fi
mkdir -p run "data/$PROFILE"
mv "$TMP" "$INI"

# ---------------------------------------------------------------- set up what is missing
if [ "$CLONE" = 1 ] && [ ! -x build/memagent ]; then
  command -v fpc >/dev/null || { echo "[run] Free Pascal is missing: sudo apt install fpc cmake (or use a release build)" >&2; exit 1; }
  echo "[run] building memagent"; make
fi
NEED_LLAMA=0
if [ "$CLONE" = 1 ]; then
  LLAMA_LIB=$(ls cache/llama/libllama.so* 2>/dev/null | head -1 || true)
  [ -z "$LLAMA_LIB" ] && NEED_LLAMA=1
  if [ "$GPU_BUILD" = 1 ] && [ -n "$LLAMA_LIB" ] && ! cuda_llama && command -v nvcc >/dev/null 2>&1; then
    echo "[run] llama.cpp has no CUDA backend: rebuilding it with CUDA"; NEED_LLAMA=1
  fi
fi
if [ "$NEED_LLAMA" = 1 ]; then
  if [ "$GPU_BUILD" = 1 ] && command -v nvcc >/dev/null 2>&1; then
    echo "[run] building llama.cpp with CUDA (one time, several minutes)"; LLAMA_CUDA=1 make llama
  else
    echo "[run] building llama.cpp for CPU (one time)"; make llama
  fi
fi
if [ "$TRAINER" = python ] || [ "$ENGINE" = python ]; then
  PY=$(command -v python3 || command -v python || true)
  [ -n "$PY" ] || { echo "[run] python3 is missing; the $PROFILE profile trains in Python" >&2; exit 1; }
  if ! "$PY" -c "import torch, transformers, peft; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[run] installing the Python trainer (torch with CUDA, transformers, peft)"
    "$PY" -m pip install -q -r trainer/requirements.txt
  fi
  if [ "$PROFILE" != small ] && ! "$PY" -c "import bitsandbytes" 2>/dev/null; then
    echo "[run] installing bitsandbytes ($PROFILE trains 4-bit)"; "$PY" -m pip install -q bitsandbytes
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'" || {
    echo "[run] PyTorch cannot see the GPU; install a CUDA build of torch (https://pytorch.org)" >&2; exit 1; }
fi

echo "[run] starting: $BIN --config $INI ${EXTRA[*]:-}"
exec "$BIN" --config "$INI" "${EXTRA[@]}"
