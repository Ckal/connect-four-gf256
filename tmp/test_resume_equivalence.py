"""Resume must be trajectory-equivalent to an uninterrupted run.

This is the property that actually matters for an 18-hour run: if the run
is interrupted at step N and resumed, the model must continue on the SAME
trajectory -- same weights, same optimiser moments, same RNG stream.  If it
does not, every resume silently perturbs the run and the "final" artifact
depends on where the interruptions happened.

The test:
  1. Train N+10 steps in one go                       -> reference
  2. Train N steps, resume from final_float.pt, +10   -> resumed
  3. Compare the float parameters and the optimiser state.

Method note: the trainer samples rows from the corpus with its own seeded
RNG, so the two runs must share the seed AND the resume must restore the
Python/NumPy/Torch RNG state.  If the restore is broken, the sampled rows
diverge and the weights diverge even though the optimiser is identical.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-c4" / "Scripts" / "python.exe")

CORPUS = ROOT / "runs" / "_probe" / "perply_v23_full" / "corpus_320k_fixed.txt"
SIDECAR = ROOT / "runs" / "_probe" / "perply_v23_full" / "merged_repaired.npz"

SCRATCH = ROOT / "runs" / "_probe" / "_resume_equiv"

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def train(run_dir, steps, resume=None, seed=32):
    """Run the real trainer. Returns (exit_code, stdout_tail).

    Mirrors run_v24.sh: the corpus and the sidecar are copied INTO the run
    directory and the trainer loads them from there with `--load-corpus`.
    That is also what makes the resume self-contained -- the resumed run
    reads the same corpus the original run trained on.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "corpus.txt").exists():
        shutil.copy2(CORPUS, run_dir / "corpus.txt")
        shutil.copy2(SIDECAR, run_dir / "perply_targets.npz")
    cmd = [
        PY, "-u", "-m", "src.c4.train",
        "--run-dir", str(run_dir),
        "--load-corpus",
        "--perply-targets", str(run_dir / "perply_targets.npz"),
        "--steps", str(steps),
        "--batch", "16",
        "--ctx", "192",
        "--lr", "0.0006",
        "--seed", str(seed),
        "--transcript", "v2",
        "--move-loss", "7way",
        "--soft-w", "1.0",
        "--eval-every", "1000000",
        "--save-every", "1000000",
        "--max-legacy-frac", "0",
        "--corpus-source", "teacher",
        "--seed-games", "0",
        "--selfplay-games", "0",
        "--teacher-games", "0",
        "--selfplay-every", "1000000",
        "--hard-start", "0",
        "--hard-ramp", "1000",
        "--hard-w", "0.5",
        "--table-bias", "0.0",
        "--lags", "0,1,2,3,4,5,6,7,8,9,10,20",
        "--col-prior", "0.0",
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                          text=True, timeout=1800)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    print("=== phase 1: uninterrupted reference run (12 steps) ===")
    ref_dir = SCRATCH / "ref"
    rc, out = train(ref_dir, 12)
    check("reference run exits 0", rc == 0, out[-800:])
    if rc != 0:
        print(out[-3000:])
        return 1

    print("=== phase 2: split run (6 steps) then resume to 12 ===")
    part_dir = SCRATCH / "split"
    rc, out = train(part_dir, 6)
    check("first half exits 0", rc == 0, out[-800:])
    if rc != 0:
        print(out[-3000:])
        return 1
    ckpt = part_dir / "final_float.pt"
    check("first half wrote final_float.pt", ckpt.exists())

    rc, out = train(part_dir, 12, resume=ckpt)
    check("resumed run exits 0", rc == 0, out[-800:])
    if rc != 0:
        print(out[-3000:])
        return 1
    check("resume reported restoring optimiser state",
          "Adam" in out or "adam" in out.lower() or "optimis" in out.lower()
          or "optimiz" in out.lower())
    check("resume reported restoring RNG state",
          "RNG" in out or "rng" in out.lower())

    print("=== phase 3: compare the two final float checkpoints ===")
    import torch
    # weights_only=False is REQUIRED here, not a habit: the float checkpoint
    # stores `rng_state` with the Python `random.getstate()` tuple and the
    # NumPy `bit_generator.state` dict.  Those are plain Python objects that
    # `weights_only=True` refuses to unpickle, so the safe loader cannot
    # read the checkpoint this test is verifying.  The file is produced by
    # our own trainer in this same repository.
    a = torch.load(ref_dir / "final_float.pt", map_location="cpu",
                   weights_only=False)
    b = torch.load(part_dir / "final_float.pt", map_location="cpu",
                   weights_only=False)
    sa, sb = a["state"], b["state"]
    check("both checkpoints carry the same parameter keys",
          set(sa.keys()) == set(sb.keys()),
          f"{set(sa.keys()) ^ set(sb.keys())}")

    worst = 0.0
    worst_key = ""
    for k in sa:
        ta = sa[k].float() if torch.is_tensor(sa[k]) else torch.tensor(sa[k])
        tb = sb[k].float() if torch.is_tensor(sb[k]) else torch.tensor(sb[k])
        if ta.shape != tb.shape:
            check(f"param {k} shape matches", False, f"{ta.shape} vs {tb.shape}")
            continue
        d = float((ta - tb).abs().max())
        if d > worst:
            worst = d
            worst_key = k
    print(f"    max |weight difference| = {worst:.3e}  (worst: {worst_key})")
    check("final weights are numerically identical (tol 1e-6)",
          worst < 1e-6, f"max diff {worst:.3e}")

    print("=== phase 4: compare optimiser moments ===")
    check("reference carries opt_state", "opt_state" in a)
    check("resumed carries opt_state", "opt_state" in b)
    if "opt_state" in a and "opt_state" in b:
        # Adam moments are keyed by parameter index.
        ea = a["opt_state"].get("state", {})
        eb = b["opt_state"].get("state", {})
        check("both have Adam moments for the same parameters",
              set(ea.keys()) == set(eb.keys()),
              f"{set(ea.keys()) ^ set(eb.keys())}")
        worst_exp = 0.0
        for k in ea:
            for mk in ("exp_avg", "exp_avg_sq"):
                va = ea[k].get(mk)
                vb = eb[k].get(mk)
                if va is None or vb is None:
                    continue
                d = float((va.float() - vb.float()).abs().max())
                worst_exp = max(worst_exp, d)
        print(f"    max |Adam moment difference| = {worst_exp:.3e}")
        check("Adam moments are numerically identical (tol 1e-6)",
              worst_exp < 1e-6, f"max diff {worst_exp:.3e}")

    print()
    print(f"=== {PASS} PASS, {FAIL} FAIL ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    # Reap the scratch dir at the END too. It holds two full training runs
    # with corpus + sidecar + optimiser state (~1.2 GB). Cleaning only at
    # the start means a successful run leaves that on disk forever, which
    # is how the volume filled up during the V24 launch.
    try:
        code = main()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(code)
