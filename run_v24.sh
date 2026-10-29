#!/usr/bin/env bash
# V24 -- ONE properly-budgeted full run.
#
# WHY V24 IS A SINGLE ARM (not another A/B)
# --------------------------------------------
# The corrected A/B (run_fixed_ab_corrected.sh) proved the supervision
# fix works and produced three usable artifacts -- but it ran each arm
# for only 6,000 steps.  One epoch over the 320,000-row corpus is
# 320,000/16 = 20,000 steps, so 6,000 steps = 0.30 EPOCHS.  The LABEL
# curve was still climbing (+1.42 pp over the last 500 steps) when the
# runs were cut off:
#
#   step  500 -> 37.55 %
#   step 3000 -> 43.55 %
#   step 6000 -> 50.92 %
#
# The same corrected trainer on a 500-row corpus, with 125 epochs of
# budget, reached 91.29 % direct label agreement.  So the architecture
# learns; the full arms were simply stopped far too early.
#
# Every "the model is weak" number measured on fx_a/fx_b/fx_c is
# therefore a statement about a 0.30-epoch model, not about the
# architecture.  V24 exists to settle that question with one
# convergence run instead of three short ones.
#
# WHAT CHANGED VS THE CORRECTED A/B
# ---------------------------------
#   --steps      6000 -> 100000   (5 epochs; see V24_EPOCHS below)
#   --eval-every 500  -> 2000     (hard export + 4,000-position label
#                                   scoring was a large share of the
#                                   measured 1.316 s/step)
#   --eval-games 0    -> 24       (0 produced a vacuous W0L0D0 readout)
#   --save-every 500  -> 2000
#
# Everything that defines the CORRECTED experiment is unchanged:
#   seed 32, ctx 192, batch 16, transcript v2, dense lags,
#   --max-legacy-frac 0, --table-bias 0.0, --hard-start 0,
#   --col-prior 0.0, --move-loss 7way, --soft-w 1.0,
#   --selfplay-games 0 --teacher-games 0, direct-label promotion gate.
#
# USAGE
#   bash run_v24.sh                     # default 100,000 steps
#   V24_STEPS=200000 bash run_v24.sh     # 10 epochs
#   V24_RESUME=runs/_probe/v24/final_float.pt bash run_v24.sh
set -u
cd "$(dirname "$0")"
PY=./.venv-c4/Scripts/python.exe
DENSE="0,1,2,3,4,5,6,7,8,9,10,20"
SEED=32
TAG=${V24_TAG:-v24}
STEPS=${V24_STEPS:-100000}
EVAL_EVERY=${V24_EVAL_EVERY:-2000}
SAVE_EVERY=${V24_SAVE_EVERY:-2000}
EVAL_GAMES=${V24_EVAL_GAMES:-24}
RESUME=${V24_RESUME:-}

# ---- refuse to double-write a run dir ----
RUNNING=$($PY -c "
import psutil
out=[]
for p in psutil.process_iter(['pid','cmdline']):
    try:
        cl=p.info['cmdline'] or []
        if 'src.c4.train' in cl and any(t in c for c in cl for t in ('$TAG',)):
            out.append(str(p.info['pid']))
    except Exception: pass
print(' '.join(out))
")
if [ -n "$RUNNING" ]; then
  echo "REFUSING: a trainer is already running against '$TAG': $RUNNING"
  exit 1
fi

# ---- one-writer lock: two V24 chains would corrupt each other's log ----
# Same pattern as the regression sweep lock: a DIRECTORY lock, with the
# owner's PID recorded so a crashed chain does not wedge the launcher
# forever. Do NOT clear this lock with `rm -rf` while a chain may still
# be running -- that defeats the one-writer guard and two chains would
# then write the same run dir.
LOCK="runs/_probe/.v24.lock"
mkdir -p runs/_probe
if ! mkdir "$LOCK" 2>/dev/null; then
  v24_pid=""
  [ -f "$LOCK/pid" ] && v24_pid=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$v24_pid" ] && ! kill -0 "$v24_pid" 2>/dev/null; then
    echo "reclaiming stale V24 lock from dead pid $v24_pid" >&2
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo "FATAL: could not reclaim $LOCK" >&2; exit 1; }
  else
    echo "ABORT: another V24 chain holds $LOCK (pid ${v24_pid:-unknown})" >&2
    echo "       Wait for it, or remove the lock dir ONLY if that chain is dead." >&2
    echo "       Do NOT use 'rm -rf' to clear it while a chain may still be running." >&2
    exit 1
  fi
fi
echo $$ > "$LOCK/pid"
cleanup() { rm -f "$LOCK/pid" 2>/dev/null || true; rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
echo "lock acquired: $LOCK"

# ---- the artifacts must exist and be clean before we start ----
CORPUS=runs/_probe/perply_v23_full/corpus_320k_fixed.txt
SIDECAR=runs/_probe/perply_v23_full/merged_repaired.npz
for f in "$CORPUS" "$SIDECAR"; do
  [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

# ---- the corpus must pass the integrity gate before any GPU is touched ----
$PY - <<'EOF'
import sys
sys.path.insert(0, '.')
from src.c4.corpus_integrity import assert_corpus_consistent
r = assert_corpus_consistent("runs/_probe/perply_v23_full/corpus_320k_fixed.txt",
                              context="pre-flight")
print(f"PRE-FLIGHT OK: {r['rows_checked']} rows, 0 violations")
EOF
[ $? -eq 0 ] || { echo "FATAL: corpus integrity pre-flight failed"; exit 1; }

ROWS=$($PY -c "
import sys
sys.path.insert(0,'.')
from src.c4.corpus_integrity import scan_corpus
r = scan_corpus('$CORPUS')
print(r.get('rows_checked', 0))
" 2>/dev/null || echo 320000)
BATCH=16
# One epoch = rows / batch STEPS (each step consumes `batch` corpus rows).
# Epoch-equivalents = steps / (rows / batch) = steps * batch / rows.
# The earlier `rows * batch / steps` was inverted and printed 51 epochs
# for a 100,000-step run when the true figure is 5.12.
STEPS_PER_EPOCH=$(awk -v r="$ROWS" -v b="$BATCH" 'BEGIN{printf "%d", r/b}')
EPOCHS=$(awk -v s="$STEPS" -v e="$STEPS_PER_EPOCH" 'BEGIN{printf "%.2f", s/e}')
echo "V24 CONFIG: rows=$ROWS steps=$STEPS batch=$BATCH"
echo "           one epoch = $STEPS_PER_EPOCH steps -> this run = $EPOCHS epoch-equivalents"
echo "           eval_every=$EVAL_EVERY save_every=$SAVE_EVERY eval_games=$EVAL_GAMES"
echo "           resume=${RESUME:-<fresh>}"

# ---- the DISJOINT holdout (the review, confirmed gap) -------------
# The promotion key's PRIMARY term is direct label agreement, but the
# scorer samples the TRAINING corpus.  That measures fit, not
# generalisation.  The holdout corpus + sidecar below are canonical-board
# DISJOINT from the training corpus, so the trainer can score the same
# objective on data the model has never seen.
#
# If the holdout artifact is missing the run still starts -- but then the
# metrics are in-sample only and the abort rules cannot fire, so say so
# loudly rather than quietly degrading the run.
HOLD_CORPUS=runs/_probe/disjoint_holdout.corpus.txt
HOLD_TARGETS=runs/_probe/disjoint_holdout.npz
HOLD_FLAGS=""
if [ -f "$HOLD_CORPUS" ] && [ -f "$HOLD_TARGETS" ]; then
  HOLD_FLAGS="--holdout-corpus $HOLD_CORPUS --holdout-targets $HOLD_TARGETS"
  echo "           holdout: $HOLD_CORPUS (disjoint) -> holdout label agreement"
else
  echo "WARNING: no disjoint holdout found -> in-sample metrics only," >&2
  echo "         the abort rules stay disabled, and label agreement is" >&2
  echo "         a FIT metric rather than a generalisation metric." >&2
fi

if [ -z "$RESUME" ]; then
  rm -rf "runs/_probe/$TAG"
fi
mkdir -p "runs/_probe/$TAG"
cp "$CORPUS" "runs/_probe/$TAG/corpus.txt"
cp "$SIDECAR" "runs/_probe/$TAG/perply_targets.npz"

COMMON="--load-corpus --transcript v2 --move-loss 7way --soft-w 1.0 \
  --steps $STEPS --batch 16 --ctx 192 --lr 0.0006 --seed $SEED \
  --eval-every $EVAL_EVERY --save-every $SAVE_EVERY --eval-games $EVAL_GAMES \
  --max-legacy-frac 0 --corpus-source teacher --seed-games 0 \
  --selfplay-games 0 --teacher-games 0 --selfplay-every 1000000 \
  --hard-start 0 --hard-ramp 1000 --hard-w 0.5 --table-bias 0.0 \
  --lags $DENSE --col-prior 0.0 \
  --lr-schedule ${V24_LR_SCHEDULE:-none} \
  --ema-decay ${V24_EMA_DECAY:-0.0} --keep-last ${V24_KEEP_LAST:-3} \
  $HOLD_FLAGS"

# Same as $COMMON but WITHOUT --lags: on resume the checkpoint supplies the
# lag set (see the refusal message in src/c4/train.py).
COMMON_NO_LAGS=$(printf '%s' "$COMMON" | sed 's/--lags [^ ]*//')

if [ -n "$RESUME" ]; then
  # On resume, DO NOT pass --lags. The lag set is part of the model's
  # structure; the trainer takes it from the checkpoint itself and refuses
  # a --lags that disagrees. Passing it here would only create a way for
  # the launcher to contradict the checkpoint.
  $PY -u -m src.c4.train --run-dir "runs/_probe/$TAG" \
    --perply-targets "runs/_probe/$TAG/perply_targets.npz" \
    --resume "$RESUME" \
    $COMMON_NO_LAGS > "runs/_probe/$TAG.log" 2>&1 &
else
  $PY -u -m src.c4.train --run-dir "runs/_probe/$TAG" \
    --perply-targets "runs/_probe/$TAG/perply_targets.npz" \
    $COMMON > "runs/_probe/$TAG.log" 2>&1 &
fi
PID=$!
echo "$TAG pid=$PID requested_steps=$STEPS"
echo "log: runs/_probe/$TAG.log"

wait $PID
RC=$?
echo "TRAINER_EXIT=$RC"

# ---- VERIFY: "the process exited" is NOT "the arm finished" ----------
if ! grep -qE "^\s*step\s+$STEPS" "runs/_probe/$TAG.log"; then
  echo "ARM FAILED  $TAG : never reached step $STEPS"
  tail -20 "runs/_probe/$TAG.log"
  exit 1
fi
if grep -q "Traceback" "runs/_probe/$TAG.log"; then
  echo "ARM FAILED  $TAG : traceback in log"
  tail -20 "runs/_probe/$TAG.log"
  exit 1
fi
for f in "runs/_probe/$TAG/final_float.pt" "runs/_probe/$TAG/hard_model_final.pt" \
         "runs/_probe/$TAG/checkpoints/best_hard.pt"; do
  if [ ! -s "$f" ]; then
    echo "ARM FAILED  $TAG : missing/empty $f"
    exit 1
  fi
done
# the final export must carry the transcript stamp, or a loader reads it
# as v1 and mis-scores it
$PY -c "
import sys, torch
sys.path.insert(0, '.')
raw = torch.load('runs/_probe/$TAG/hard_model_final.pt', map_location='cpu', weights_only=True)
v = raw.get('transcript') if isinstance(raw, dict) else None
if v != 'v2':
    print('BAD transcript stamp:', v); sys.exit(1)
print('transcript stamp OK:', v)
" 2>/dev/null || { echo "ARM FAILED  $TAG : final export missing transcript=v2 stamp"; exit 1; }

LBL=$(grep -oE "LABEL [0-9.]+%" "runs/_probe/$TAG.log" | tail -1)
EL=$(grep -oE "elapsed: [0-9]+s" "runs/_probe/$TAG.log" | tail -1)
echo "ARM OK       $TAG : reached step $STEPS, $EL, $LBL, artifacts present, stamp v2"

# ---- the promoted artifact must be MEASURED, not just exist ----
$PY -u tmp/measure_promotion_scorer.py "runs/_probe/$TAG" \
  > "runs/_probe/${TAG}_promotion_scorer.log" 2>&1
echo "--- shipped promotion scorer (authoritative) ---"
tail -20 "runs/_probe/${TAG}_promotion_scorer.log"

echo "V24 DONE"
