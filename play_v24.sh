#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# play_v24.sh -- watch a training run's promoted checkpoint and keep the UI
#                serving the newest one.
#
# Why this exists
# ---------------
# A long training run improves the model continuously. The point of the UI
# is to let a human play THAT model while it is still learning. Without
# this, the server loads a checkpoint once at startup and serves a frozen
# model for the whole run.
#
# What it does
# ------------
#   * starts the UI server on the run's promoted checkpoint
#       runs/_probe/$TAG/checkpoints/best_hard.pt
#   * polls that file's content hash
#   * when the trainer promotes a better checkpoint, it calls GET /reload
#     so the server hot-swaps the model WITHOUT restarting and WITHOUT
#     dropping the game in progress
#   * writes a small JSON status file so the cadence is auditable
#
# The trainer writes the checkpoint atomically (write temp -> rename), so a
# reload never sees a half-written file.
#
# Usage
# -----
#   bash play_v24.sh                      # uses $V24_TAG or the newest run
#   V24_TAG=my_run bash play_v24.sh
#   PLAY_PORT=7861 bash play_v24.sh
#   PLAY_POLL=15 bash play_v24.sh         # poll every 15 s (default 20 s)
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TAG="${V24_TAG:-}"
if [[ -z "$TAG" ]]; then
  # newest run directory that has a promoted checkpoint
  TAG="$(ls -1dt runs/_probe/*/checkpoints/best_hard.pt 2>/dev/null | head -1 \
        | sed 's#runs/_probe/##; s#/checkpoints/best_hard.pt##')"
fi
if [[ -z "$TAG" ]]; then
  echo "play_v24: no run with checkpoints/best_hard.pt found under runs/_probe/" >&2
  exit 2
fi

CKPT="runs/_probe/$TAG/checkpoints/best_hard.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "play_v24: $CKPT does not exist yet" >&2
  exit 2
fi

PORT="${PLAY_PORT:-7860}"
HOST="${PLAY_HOST:-127.0.0.1}"
DEVICE="${PLAY_DEVICE:-cpu}"
POLL="${PLAY_POLL:-20}"
STATUS="runs/_probe/$TAG/play_status.json"
LOG="runs/_probe/$TAG/play_ui.log"

mkdir -p "runs/_probe/$TAG"

# ---------------------------------------------------------------------------
# HTTP readiness probe.
#
# WHY NOT `curl -o /dev/null`:
# On THIS host curl is the native Windows build (8.18.0 mingw32).  Its
# `-o /dev/null` fails with exit code 23 ("write error") on an otherwise
# PERFECTLY SUCCESSFUL request -- the response body is written, the HTTP
# status is 200, but curl reports a failure while disposing of the body.
# Measured: `curl -s -o /dev/null .../state` -> rc=23 on a 200 response,
# while `curl -s -o - .../state` -> rc=0 and `curl -s -o <file>` -> rc=0.
#
# A readiness probe that returns non-zero on success is worse than no
# probe at all: the watcher would wait the full 90 s, declare the server
# dead, and exit 4 -- while a perfectly good server was running.  So the
# probe checks the HTTP STATUS CODE, which curl reports faithfully, and
# discards the body with `-o` to a temp file instead of /dev/null.
# ---------------------------------------------------------------------------
HTTP_STATUS() {
  local url="$1" timeout_s="${2:-4}"
  local tmp
  tmp="$(mktemp 2>/dev/null || echo "runs/_probe/.curl_probe.$$")"
  local code
  code="$(curl -s -o "$tmp" -w '%{http_code}' --max-time "$timeout_s" "$url" 2>/dev/null)"
  rm -f "$tmp" 2>/dev/null || true
  [[ "$code" == "200" ]]
}

# one-writer lock: two watchers on the same run would fight over the port
LOCK="runs/_probe/$TAG/.play.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "play_v24: another watcher already holds $LOCK -- refusing to start a second" >&2
  exit 3
fi
cleanup() {
  rmdir "$LOCK" 2>/dev/null || true
  # Kill the UI server THIS watcher started. Leaving it behind means the
  # port stays occupied and the next watcher silently reuses a stale
  # server that is serving the OLD checkpoint.
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "play_v24: tag=$TAG"
echo "play_v24: checkpoint=$CKPT"
echo "play_v24: serving http://$HOST:$PORT/  (poll every ${POLL}s, hot reload on change)"

# start the UI server if it is not already up on that port
if HTTP_STATUS "http://$HOST:$PORT/state" 4; then
  echo "play_v24: a server is already answering on $PORT -- reusing it"
  SERVER_PID=""
else
  PY="${PY:-./.venv-c4/Scripts/python.exe}"
  [[ -x "$PY" ]] || PY="python3"
  # PYTHONPATH must contain ONLY the project root. Inheriting an outer
  # PYTHONPATH (e.g. from a tool runner) can shadow the venv's numpy with a
  # build for a different Python version, which fails at `import torch`
  # with a confusing "compiled module files exist, but seem incompatible"
  # error. The project root is the only thing this server needs.
  PYTHONPATH="$ROOT" "$PY" -u -m src.c4.ui_server --model "$CKPT" \
      --port "$PORT" --device "$DEVICE" > "$LOG" 2>&1 &
  SERVER_PID=$!
  echo "play_v24: started UI server pid=$SERVER_PID (log=$LOG)"
  for _ in $(seq 1 90); do
    if HTTP_STATUS "http://$HOST:$PORT/state" 4; then break; fi
    sleep 1
  done
  if ! HTTP_STATUS "http://$HOST:$PORT/state" 4; then
    echo "play_v24: UI server did not come up -- see $LOG" >&2
    tail -30 "$LOG" >&2 || true
    exit 4
  fi
fi

# --------------------------------------------------------------- watch loop
# The SERVER already knows whether the checkpoint changed on disk (it
# tracks size+mtime and falls back to a content hash). So the watcher
# just asks /modelinfo, and calls /reload when the answer is "yes".
# That keeps the hashing logic in one place -- the server -- instead of
# duplicating it here, and removes a shell->python dependency from the
# polling path.
last_step=""
reloads=0
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
  local state="$1" detail="${2:-}"
  # `last_step` is EMPTY until the server reports a step.  Emitting it
  # bare produced `"step": ,` -- which is NOT valid JSON, so every
  # consumer of the status file (and the watcher's own test) failed to
  # parse it.  An unknown step is `null`, not an empty token.
  local step_json="null"
  if [[ -n "$last_step" ]]; then step_json="$last_step"; fi
  # The detail text is interpolated into a JSON string, so any quote or
  # backslash in it would break the JSON.  Escape it before writing.
  local detail_json
  detail_json="$(printf '%s' "$detail" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
  cat > "$STATUS" <<EOF
{
  "tag": "$TAG",
  "checkpoint": "$CKPT",
  "state": "$state",
  "detail": "$detail_json",
  "step": $step_json,
  "reloads": $reloads,
  "started_at": "$started_at",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "port": $PORT,
  "url": "http://$HOST:$PORT/"
}
EOF
}

while true; do
  if [[ ! -f "$CKPT" ]]; then
    write_status "waiting_for_checkpoint" "checkpoint not written yet"
    sleep "$POLL"
    continue
  fi

  info="$(curl -s --max-time 120 "http://$HOST:$PORT/modelinfo" 2>/dev/null || echo "")"
  if [[ -z "$info" ]]; then
    write_status "server_unreachable" "the UI server did not answer /modelinfo"
    sleep "$POLL"
    continue
  fi

  changed="$(printf '%s' "$info" | sed -n 's/.*"changed_on_disk": *\(true\|false\).*/\1/p')"
  if [[ "$changed" == "true" ]]; then
    resp="$(curl -s --max-time 180 "http://$HOST:$PORT/reload" 2>/dev/null || echo '{"ok":false}')"
    if [[ "$resp" == *'"ok": true'* ]]; then
      reloads=$((reloads + 1))
      step="$(printf '%s' "$resp" | sed -n 's/.*"step": *\([0-9]*\).*/\1/p')"
      if [[ -n "$step" ]]; then last_step="$step"; fi
      write_status "serving" "hot reloaded to step ${step:-unknown}"
      echo "play_v24: hot reload -> step ${step:-unknown}"
    else
      # the file may have been mid-write; try again next tick
      write_status "reload_failed" "server refused the reload; will retry"
      echo "play_v24: reload failed -- retrying" >&2
    fi
  elif [[ -z "$last_step" ]]; then
    # first observation, nothing changed yet: report what is served
    step="$(printf '%s' "$info" | sed -n 's/.*"step": *\([0-9]*\).*/\1/p')"
    if [[ -n "$step" ]]; then last_step="$step"; fi
    write_status "serving" "watching for a newer checkpoint"
  else
    write_status "serving" "watching for a newer checkpoint"
  fi
  sleep "$POLL"
done
