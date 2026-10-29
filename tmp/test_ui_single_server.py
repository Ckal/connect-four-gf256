"""Regression test: ONE ui_server per port.

The original failure was not a flaky test.  On Windows, setting
SO_REUSEADDR (which `allow_reuse_address = True` does) lets a SECOND
live socket bind an address that is ALREADY bound by a live socket.
A stale ui_server from an earlier session therefore kept running while
a freshly started one also printed "serving http://127.0.0.1:7860/".

The observable damage:

  * requests were answered by the OLD process, so `/modelinfo` lacked
    the `step` field that the current code returns;
  * the live-parity tests intermittently failed with connection resets,
    and then "recovered" at random as the shadowing process died;
  * a measurement taken against the "new" server was in fact taken
    against the old checkpoint.

This test pins the fix:

  1. a second server on the SAME port must REFUSE to start (exit code
     3, non-zero) instead of silently binding;
  2. the first server must still be the one answering requests;
  3. the port must be free again once the first server exits.

Run standalone:
    ./.venv-c4/Scripts/python.exe tmp/test_ui_single_server.py
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt"
PORT = 7867

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def port_is_open(port: int, timeout: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def start_server(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "src.c4.ui_server",
         "--model", CKPT, "--port", str(port), "--device", "cpu"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def wait_up(port: int, proc: subprocess.Popen, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if port_is_open(port, timeout=0.5):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    print("=== single-server-per-port regression ===")

    if not (ROOT / CKPT).exists():
        print(f"SKIP: checkpoint missing: {CKPT}")
        return 0

    # ---------------------------------------------------------------
    print("\n[1] start the FIRST server")
    p1 = start_server(PORT)
    try:
        up = wait_up(PORT, p1)
        check("first server comes up", up,
              "" if up else (p1.stdout.read() if p1.stdout else "no output"))
        if not up:
            return 1

        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/modelinfo",
                                    timeout=30) as r:
            info1 = json.loads(r.read().decode("utf-8"))
        check("first server reports a step", "step" in info1, str(info1)[:120])

        # -----------------------------------------------------------
        print("\n[2] a SECOND server on the same port must REFUSE")
        p2 = start_server(PORT)
        try:
            out = p2.communicate(timeout=120)[0] or ""
        except subprocess.TimeoutExpired:
            p2.kill()
            out = p2.communicate()[0] or ""
            out += "\n[TIMEOUT: second server kept running -- it shadowed the first]"
        rc2 = p2.returncode
        check("second server exits non-zero", rc2 not in (0, None),
              f"rc={rc2}")
        check("second server reports the bind conflict",
              "cannot bind" in out or "already" in out or "10048" in out,
              out[-300:].replace("\n", " | "))

        # -----------------------------------------------------------
        print("\n[3] the FIRST server must still be answering")
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/modelinfo",
                                    timeout=30) as r:
            info3 = json.loads(r.read().decode("utf-8"))
        check("first server still serves the same checkpoint",
              info3.get("checkpoint") == info1.get("checkpoint"),
              f"{info3.get('checkpoint')} vs {info1.get('checkpoint')}")
        check("first server still serves the same step",
              info3.get("step") == info1.get("step"),
              f"{info3.get('step')} vs {info1.get('step')}")
    finally:
        try:
            p1.terminate()
            p1.wait(timeout=30)
        except Exception:
            try:
                p1.kill()
            except Exception:
                pass

    # ---------------------------------------------------------------
    print("\n[4] the port is free again after the first server exits")
    deadline = time.time() + 20
    while time.time() < deadline and port_is_open(PORT, timeout=0.5):
        time.sleep(0.5)
    check("port released on exit", not port_is_open(PORT, timeout=0.5))

    # ---------------------------------------------------------------
    print("\n[5] a fresh server can bind the freed port")
    p3 = start_server(PORT)
    try:
        up3 = wait_up(PORT, p3)
        check("fresh server binds the freed port", up3,
              "" if up3 else (p3.stdout.read() if p3.stdout else "no output"))
    finally:
        try:
            p3.terminate()
            p3.wait(timeout=30)
        except Exception:
            try:
                p3.kill()
            except Exception:
                pass

    print("\n=== %s ===" % ("ALL PASS" if not FAILS else
                            f"{len(FAILS)} FAILURES: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
