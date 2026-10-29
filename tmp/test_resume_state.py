"""REGRESSION: a resume must restore optimiser + RNG state.

WHY
---
Claude's review flagged that the trainer's checkpoint carries weights and
step but NOT the optimiser moments or the RNG states.  That is real: a
resume restarts Adam from empty moments (loss spike) and continues with a
different sampling sequence, so a resumed run is NOT the same trajectory
as an uninterrupted one.

This test asserts the FIX:

  1. `capture_train_state` returns Adam moments and three RNG states.
  2. The float checkpoint writers attach that state:
       - `final_float.pt`
       - `checkpoints/best_soft.pt`
       - `checkpoints/step<N>_soft.pt`
  3. The hard export does NOT carry it (it must stay a lean integer
     payload -- it is the artifact that ships).
  4. A real resume REPORTS restoring them.

The trainer is exercised end-to-end on a small corpus, because an AST
check cannot prove the payload is actually loadable.

Run: ./.venv-c4/Scripts/python.exe tmp/test_resume_state.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = str(ROOT / ".venv-c4/Scripts/python.exe")
CORPUS = "runs/_probe/perply_v23_full/corpus_320k_fixed.txt"
SIDECAR = "runs/_probe/perply_v23_full/merged_repaired.npz"
DENSE = "0,1,2,3,4,5,6,7,8,9,10,20"
RUN = ROOT / "runs/_probe/_resume_state_test"

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def train_cmd(run_dir: pathlib.Path, steps: int, resume: str = "") -> list:
    cmd = [
        PY, "-u", "-m", "src.c4.train",
        "--run-dir", str(run_dir),
        "--perply-targets", str(run_dir / "perply_targets.npz"),
        "--load-corpus", "--transcript", "v2", "--move-loss", "7way",
        "--soft-w", "1.0",
        "--steps", str(steps), "--batch", "16", "--ctx", "192",
        "--lr", "0.0006", "--seed", "32",
        "--eval-every", "1000000", "--save-every", "10",
        "--eval-games", "0",
        "--max-legacy-frac", "0", "--corpus-source", "teacher",
        "--seed-games", "0", "--selfplay-games", "0", "--teacher-games", "0",
        "--selfplay-every", "1000000",
        "--hard-start", "0", "--hard-ramp", "1000", "--hard-w", "0.5",
        "--table-bias", "0.0", "--lags", DENSE, "--col-prior", "0.0",
    ]
    if resume:
        cmd += ["--resume", resume]
    return cmd


# Windows can fail to start a Python child outright under load:
#   STATUS_DLL_INIT_FAILED = 0xC0000142 = 3221225794
# The process dies before running any code, so there is NO output to
# inspect and NO logic error -- it is a transient environment failure.
# Retrying is the correct response; reporting it as a test failure is not.
_DLL_INIT_FAILED = 3221225794


def run_trainer(cmd, *, cwd, env, timeout, attempts: int = 3):
    """Run the trainer, retrying a transient Windows DLL-init failure."""
    last: Optional[subprocess.CompletedProcess] = None
    for attempt in range(1, attempts + 1):
        last = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                               text=True, timeout=timeout)
        if last.returncode != _DLL_INIT_FAILED:
            return last
        print(f"  [retry] trainer died with STATUS_DLL_INIT_FAILED "
              f"(attempt {attempt}/{attempts}); retrying in 10s")
        time.sleep(10)
    assert last is not None
    return last


def main() -> int:
    if not (ROOT / CORPUS).exists() or not (ROOT / SIDECAR).exists():
        print("SKIP: corpus/sidecar missing")
        return 0
    shutil.rmtree(RUN, ignore_errors=True)
    RUN.mkdir(parents=True)
    shutil.copy2(ROOT / CORPUS, RUN / "corpus.txt")
    shutil.copy2(ROOT / SIDECAR, RUN / "perply_targets.npz")

    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    # ---------------- phase 1: train 20 steps, write a checkpoint ---------
    print("\n=== phase 1: train 20 steps ===")
    p1 = run_trainer(train_cmd(RUN, 20), cwd=str(ROOT), env=env,
                     timeout=5400)
    print(f"  rc={p1.returncode}")
    if p1.returncode != 0:
        print(p1.stdout[-3000:])
        print(p1.stderr[-3000:])
        check("phase 1 trainer exits 0", False, f"rc={p1.returncode}")
        return 1
    check("phase 1 trainer exits 0", True)

    # ---------------- phase 2: inspect what was written ------------------
    print("\n=== phase 2: inspect the written checkpoints ===")
    import torch

    final = RUN / "final_float.pt"
    check("final_float.pt exists", final.exists())
    # weights_only=False is REQUIRED here: the checkpoint now carries the
    # Adam moments (tensors) AND the Python/NumPy RNG states (a tuple and a
    # dict of plain Python objects), which weights_only=True refuses to
    # unpickle. These are our own files, written by our own trainer.
    raw = torch.load(final, map_location="cpu", weights_only=False)
    check("final_float.pt has 'state'", isinstance(raw.get("state"), dict))
    check("final_float.pt carries opt_state", "opt_state" in raw,
          f"keys={sorted(raw.keys())}")
    check("final_float.pt carries rng_state", "rng_state" in raw)
    if "rng_state" in raw:
        rs = raw["rng_state"]
        check("rng_state has python/numpy/torch",
              all(k in rs for k in ("python", "numpy", "torch")),
              f"keys={sorted(rs.keys())}")

    step_soft = sorted((RUN / "checkpoints").glob("step*_soft.pt"))
    check("a step*_soft.pt snapshot exists", len(step_soft) > 0,
          f"n={len(step_soft)}")
    if step_soft:
        sraw = torch.load(step_soft[-1], map_location="cpu", weights_only=False)
        check("step*_soft.pt carries opt_state", "opt_state" in sraw)
        check("step*_soft.pt carries rng_state", "rng_state" in sraw)

    best_soft = RUN / "checkpoints" / "best_soft.pt"
    if best_soft.exists():
        braw = torch.load(best_soft, map_location="cpu", weights_only=False)
        check("best_soft.pt carries opt_state", "opt_state" in braw)
    else:
        # Promotion may legitimately not fire on a 20-step run; the file is
        # then absent, which is not a failure of this feature.
        print("  [INFO] best_soft.pt absent (promotion did not fire)")

    hard = RUN / "hard_model_final.pt"
    check("hard_model_final.pt exists", hard.exists())
    if hard.exists():
        hraw = torch.load(hard, map_location="cpu", weights_only=False)
        check("hard export does NOT carry opt_state (lean payload)",
              "opt_state" not in hraw, f"keys={sorted(hraw.keys())[:12]}")
        check("hard export does NOT carry rng_state",
              "rng_state" not in hraw)
        check("hard export keeps transcript stamp",
              isinstance(hraw, dict) and hraw.get("transcript") == "v2",
              f"transcript={hraw.get('transcript') if isinstance(hraw, dict) else None}")

    # ---------------- phase 3: resume and check the report ----------------
    print("\n=== phase 3: resume from final_float.pt ===")
    RUN2 = ROOT / "runs/_probe/_resume_state_test2"
    shutil.rmtree(RUN2, ignore_errors=True)
    RUN2.mkdir(parents=True)
    shutil.copy2(ROOT / CORPUS, RUN2 / "corpus.txt")
    shutil.copy2(ROOT / SIDECAR, RUN2 / "perply_targets.npz")

    # On resume the lag set comes from the checkpoint: pass --lags and let
    # the trainer verify they agree (it refuses a mismatch).
    p2 = run_trainer(train_cmd(RUN2, 25, resume=str(final)),
                     cwd=str(ROOT), env=env, timeout=5400)
    out = p2.stdout + p2.stderr
    print(f"  rc={p2.returncode}")
    check("phase 3 resume exits 0", p2.returncode == 0, out[-500:] if p2.returncode else "")
    check("resume reports restoring Adam moments",
          "restored Adam moments" in out)
    check("resume reports restoring RNG state",
          "restored RNG states" in out)

    # ---------------- phase 4: hard resume says so -----------------------
    print("\n=== phase 4: resume from a HARD export (must say it cannot) ===")
    RUN3 = ROOT / "runs/_probe/_resume_state_test3"
    shutil.rmtree(RUN3, ignore_errors=True)
    RUN3.mkdir(parents=True)
    shutil.copy2(ROOT / CORPUS, RUN3 / "corpus.txt")
    shutil.copy2(ROOT / SIDECAR, RUN3 / "perply_targets.npz")
    p3 = run_trainer(train_cmd(RUN3, 25, resume=str(hard)),
                     cwd=str(ROOT), env=env, timeout=5400)
    out3 = p3.stdout + p3.stderr
    print(f"  rc={p3.returncode}")
    check("hard-resume either succeeds or explains itself",
          ("carries no optimiser state" in out3) or (p3.returncode == 0))
    check("hard-resume does NOT claim to restore Adam moments",
          "restored Adam moments" not in out3)

    # ---------------- summary --------------------------------------------
    print(f"\n=== {len(PASS)} PASS, {len(FAIL)} FAIL ===")
    if FAIL:
        for f in FAIL:
            print(f"  FAILED: {f}")
    (ROOT / "runs/_probe/resume_state_test.json").write_text(json.dumps(
        {"pass": PASS, "fail": FAIL}, indent=2))
    return 1 if FAIL else 0


if __name__ == "__main__":
    # Reap the scratch run dirs at the END too. Each holds a corpus copy,
    # a sidecar copy and full checkpoints (~1.4 GB together). Cleaning only
    # at the start leaves that on disk after every successful run, which
    # is how the volume filled up during the V24 launch.
    try:
        code = main()
    finally:
        for _d in (RUN, ROOT / "runs/_probe/_resume_state_test2",
                   ROOT / "runs/_probe/_resume_state_test3"):
            try:
                shutil.rmtree(_d, ignore_errors=True)
            except Exception:
                pass
    raise SystemExit(code)
