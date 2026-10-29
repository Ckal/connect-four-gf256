"""End-to-end test of play_v24.sh (the play-while-training watcher).

Uses a SCRATCH run directory with copies of real checkpoints, so no
historical artifact is ever overwritten. This test exists because an
earlier version of this flow accidentally clobbered a promoted artifact.

What it proves:
  1. the watcher starts and writes a status file,
  2. the watcher picks up a NEW checkpoint written to the watched path
     and hot-reloads the server,
  3. the status file records the reload count and the new step,
  4. a second watcher refuses to start (one-writer lock),
  5. the original historical artifacts are UNTOUCHED.
"""
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import psutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS = []
PASSES = 0


def check(name, cond, detail=""):
    global PASSES
    if cond:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


SRC_A = ROOT / "runs/_probe/fx_a_s6000/checkpoints/step6000_hard.pt"
SRC_B = ROOT / "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt"
if not SRC_A.exists() or not SRC_B.exists():
    print("SKIP: source checkpoints missing")
    sys.exit(0)

# guard the originals
src_a_sha, src_b_sha = sha(SRC_A), sha(SRC_B)

scratch = pathlib.Path(tempfile.mkdtemp(prefix="play_v24_test_",
                                        dir=str(ROOT / "runs" / "_probe")))
tag = scratch.name
# The watcher expects the PRODUCTION layout:
#   runs/_probe/$TAG/checkpoints/best_hard.pt
run_dir = scratch / "checkpoints"
run_dir.mkdir(parents=True)
served = run_dir / "best_hard.pt"
shutil.copy2(SRC_A, served)

port = free_port()
env = {**os.environ, "PYTHONPATH": str(ROOT)}

# `bash` on this host resolves to WSL when invoked from Python, which has
# no /bin/bash. Git Bash is the shell this project actually uses.
BASH = r"C:\Program Files\Git\bin\bash.exe"
if not pathlib.Path(BASH).exists():
    BASH = "/usr/bin/bash"

watcher = subprocess.Popen(
    [BASH, "play_v24.sh"],
    cwd=str(ROOT),
    env={**env, "V24_TAG": tag, "PLAY_PORT": str(port), "PLAY_POLL": "2",
         # The watcher must not inherit an outer PYTHONPATH: it can shadow
         # the venv's numpy with a build for a different Python version.
         "PYTHONPATH": str(ROOT)},
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
watcher_out = []

try:
    # ---------------------------------------------- wait for the status file
    status = scratch / "play_status.json"
    for _ in range(90):
        if status.exists() and status.stat().st_size > 0:
            break
        time.sleep(1)
    check("watcher wrote a status file", status.exists(),
          f"missing {status}")
    if status.exists():
        s1 = json.loads(status.read_text())
        check("status says it is serving", s1.get("state") == "serving",
              json.dumps(s1))
        check("status records the checkpoint path",
              s1.get("checkpoint", "").endswith("best_hard.pt"), json.dumps(s1))

    # --------------------------------------------- the UI is answering
    ok = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/modelinfo", timeout=5) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            time.sleep(1)
    check("the UI server is up", ok)

    # ------------------------------------- a NEW checkpoint appears
    tmp = served.with_name(served.name + ".tmp")
    shutil.copy2(SRC_B, tmp)
    os.replace(tmp, served)          # exactly how the trainer promotes

    reloaded = False
    for _ in range(60):
        try:
            s2 = json.loads(status.read_text())
        except Exception:
            time.sleep(1)
            continue
        if s2.get("reloads", 0) >= 1:
            reloaded = True
            break
        time.sleep(1)
    check("the watcher hot-reloaded on a new checkpoint", reloaded,
          status.read_text() if status.exists() else "(no status)")

    if reloaded:
        s3 = json.loads(status.read_text())
        check("status records the reload", s3.get("reloads", 0) >= 1,
              json.dumps(s3))
        check("status reports the new step", s3.get("step") not in (None, "", 0),
              json.dumps(s3))
        check("status state is serving", s3.get("state") == "serving",
              json.dumps(s3))

    # ------------------------------------- the served model really changed
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/modelinfo", timeout=10) as r:
        mi = json.loads(r.read().decode())
    check("the server reports no pending change after the reload",
          mi.get("changed_on_disk") is False, json.dumps(mi))

    # ------------------------------------- one-writer lock
    second = subprocess.run(
        [BASH, "play_v24.sh"],
        cwd=str(ROOT),
        env={**env, "V24_TAG": tag, "PLAY_PORT": str(free_port()),
             "PLAY_POLL": "2"},
        capture_output=True, text=True, timeout=120,
    )
    check("a second watcher refuses to start",
          second.returncode != 0 and "another watcher" in
          (second.stdout + second.stderr),
          f"rc={second.returncode} out={second.stdout[:200]}{second.stderr[:200]}")

finally:
    # ---- kill the watcher AND its whole process tree --------------------
    # The old cleanup called watcher.terminate() and then swept the UI
    # server with PowerShell. That leaked the WATCHER itself: on Windows
    # terminate() on an MSYS bash does not reliably run the script's EXIT
    # trap, and the PowerShell sweep only matched `*ui_server*`, never the
    # `play_v24.sh` bash.
    #
    # Measured consequence: 24 leaked watcher/UI processes
    # survived the watcher regression suite, holding 1107 CPU-seconds and
    # 1.2 GB RSS -- enough contention to make an 18-hour training run and
    # the live-parity tests flaky. So kill the tree explicitly.
    def _kill_tree(proc):
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:
            children = []
        for c in children:
            try:
                c.kill()
            except Exception:
                pass
        try:
            proc.kill()
        except Exception:
            pass

    _kill_tree(watcher)
    try:
        watcher.wait(timeout=15)
    except Exception:
        pass

    # Sweep any straggler that still references this test's tag: the UI
    # server, or a second bash that outlived its parent.
    try:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cl = p.cmdline() or []
                j = " ".join(cl)
                if tag in j and ("ui_server" in j or "play_v24.sh" in j):
                    if "-c" in cl[:2]:
                        continue
                    p.kill()
            except Exception:
                pass
    except Exception:
        pass

    shutil.rmtree(scratch, ignore_errors=True)

# --------------------------------------------------- originals untouched
check("source checkpoint A is untouched", sha(SRC_A) == src_a_sha)
check("source checkpoint B is untouched", sha(SRC_B) == src_b_sha)

print()
if FAILS:
    print(f"FAIL: {len(FAILS)} check(s) failed: {FAILS}")
    sys.exit(1)
print(f"PASS: {PASSES} checks -- the play-while-training watcher works")
