#!/usr/bin/env bash
# Serial regression sweep.  Run serially: the UI parity tests share one
# live server, and concurrent runs race on it.
set -u
cd "$(dirname "$0")/.."
LOG=runs/_probe/regression_sweep.log
mkdir -p runs/_probe

# ---- one-writer lock ---------------------------------------------------
# The live-parity tests all drive ONE ui_server on ONE port and reset the
# board under each other. Two concurrent sweeps therefore corrupt each
# other's live state and write into the same LOG, producing interleaved
# output that looks like random failures. Each sweep writes its own
# per-test logs too (`_sweep_<test>.log`), so a concurrent pair also
# overwrites the evidence for the same test name.
#
# This is the SAME failure mode as the post-training battery, which is
# why it gets the same lock.
LOCK=runs/_probe/.regression_sweep.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  # A stale lock (owner died) must not block forever. The lock records the
  # owner's PID; if that PID is gone, reclaim it. Note the lock is a
  # DIRECTORY, so `rm -f` does not remove it -- `rm -rf` does, and using
  # `rm -rf` to "clear" the lock silently DEFEATS the one-writer guard
  # and lets several sweeps race on port 7860. Never do that.
  lock_pid_file="$LOCK/pid"
  lock_pid=""
  [ -f "$lock_pid_file" ] && lock_pid=$(cat "$lock_pid_file" 2>/dev/null)
  if [ -n "$lock_pid" ] && ! kill -0 "$lock_pid" 2>/dev/null; then
    echo "reclaiming stale sweep lock from dead pid $lock_pid" >&2
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo "ABORT: could not reclaim $LOCK" >&2; exit 1; }
  else
    echo "ABORT: another regression sweep holds $LOCK (pid ${lock_pid:-unknown})" >&2
    echo "       Wait for it, or remove the lock dir ONLY if that sweep is dead." >&2
    echo "       Do NOT use 'rm -rf' to clear it while a sweep may still be running." >&2
    exit 1
  fi
fi
echo $$ > "$LOCK/pid"

# The sweep summary must be sweep-unique. A fixed summary path means two
# concurrent sweeps overwrite each other's PASS/FAIL lines, and the result
# looks like random per-test failures even though each sweep was fine.
SWEEP_ID="$$"
LOG="runs/_probe/_sweep_${SWEEP_ID}_summary.log"
: > "$LOG"

# A sweep writes its per-test evidence to `_sweep_<test>.log`.  If two
# sweeps ever ran at once they would overwrite each other's evidence --
# and because the live-parity tests drive ONE shared ui_server, the
# second sweep's board resets would make the first sweep's parity tests
# report FAIL while their own stdout still said "PARITY_OK = True".
# The lock above prevents concurrency; this suffix makes the evidence
# file unique per sweep so a stale file can never be mistaken for this
# run's result.
PASS=0; FAIL=0; FAILED=""

# ---- own the UI server -------------------------------------------------
# The live-parity tests drive ONE ui_server on port 7860 and reset the
# board under each other.  Previously the sweep ASSUMED a server was
# already listening there -- which is how a STALE server from an earlier
# session could shadow the current code and make the parity tests fail
# at random (on Windows a second bind silently wins).
#
# The sweep now starts its own server from the CURRENT source, against
# the SAME checkpoint the parity tests compare against, and stops it
# afterwards.  That makes the sweep self-contained and reproducible.
UI_PORT="${SWEEP_UI_PORT:-7860}"
UI_CKPT="${SWEEP_UI_CKPT:-runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt}"
UI_LOG="runs/_probe/_sweep_${SWEEP_ID}_ui_server.log"
UI_PID=""
stop_ui() {
  if [ -n "$UI_PID" ] && kill -0 "$UI_PID" 2>/dev/null; then
    kill "$UI_PID" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$UI_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$UI_PID" 2>/dev/null || true
  fi
}
cleanup() {
  stop_ui
  # The lock dir now holds a `pid` file, so a bare `rmdir` fails and the
  # lock would survive the sweep and block the next one. Remove the
  # contents first, then the dir. (A stale lock is harmless anyway -- the
  # next sweep reclaims it when the recorded PID is dead -- but leaving it
  # behind is confusing and hides a real concurrent run.)
  rm -f "$LOCK/pid" 2>/dev/null || true
  rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ ! -f "$UI_CKPT" ]; then
  echo "ABORT: sweep UI checkpoint missing: $UI_CKPT" >&2
  exit 1
fi

# If something already holds the port, the sweep refuses to run rather
# than measure against a server it does not control.
if curl -s -o /dev/null -w '%{http_code}' --max-time 4 \
     "http://127.0.0.1:${UI_PORT}/state" 2>/dev/null | grep -q '^200$'; then
  echo "ABORT: a server is already listening on ${UI_PORT}." >&2
  echo "       Stop it first, or set SWEEP_UI_PORT to a free port." >&2
  exit 1
fi

PYTHONPATH="$PWD" ./.venv-c4/Scripts/python.exe -u -m src.c4.ui_server \
    --model "$UI_CKPT" --port "$UI_PORT" --device cpu > "$UI_LOG" 2>&1 &
UI_PID=$!
ui_up=0
for _ in $(seq 1 120); do
  if ! kill -0 "$UI_PID" 2>/dev/null; then break; fi
  if curl -s -o /dev/null -w '%{http_code}' --max-time 4 \
       "http://127.0.0.1:${UI_PORT}/state" 2>/dev/null | grep -q '^200$'; then
    ui_up=1; break
  fi
  sleep 1
done
if [ "$ui_up" -ne 1 ]; then
  echo "ABORT: sweep UI server did not come up on ${UI_PORT}" >&2
  tail -20 "$UI_LOG" >&2 || true
  exit 1
fi
echo "sweep: ui server pid=$UI_PID on ${UI_PORT} serving ${UI_CKPT}"

for t in tmp/test_*.py; do
  name=$(basename "$t")
  case "$name" in
    test_is_full.py|test_ui_isfull.py) continue ;;   # superseded by _v2
  esac
  out="runs/_probe/_sweep_${SWEEP_ID}_$(basename "$t" .py).log"
  # The live-parity tests compare the UI server against an independent
  # probe. They must talk to THIS sweep's server, not to whatever
  # happens to be listening on the default port -- otherwise the test
  # silently measures a different model and reports a false mismatch.
  # UI_PORT is exported so the tests (which read os.environ) follow
  # the sweep's port.
  UI_PORT="$UI_PORT" timeout 900 ./.venv-c4/Scripts/python.exe -u "$t" > "$out" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    PASS=$((PASS+1)); echo "PASS  $name" >> "$LOG"
  else
    FAIL=$((FAIL+1)); FAILED="$FAILED $name"
    echo "FAIL  $name  (exit $rc)  -> $out" >> "$LOG"
  fi
  echo "" >> "$LOG"
done
echo "=== $PASS PASS, $FAIL FAIL ===" >> "$LOG"
if [ -n "$FAILED" ]; then echo "failed:$FAILED" >> "$LOG"; fi
# Stable pointer for humans and scripts that want "the most recent sweep".
# Copy (not symlink -- Windows) so it survives toolchains that do not
# follow symlinks.
cp -f "$LOG" runs/_probe/regression_sweep.log
tail -30 "$LOG"
