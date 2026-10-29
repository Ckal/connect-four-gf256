"""Measure the SHIPPED per-ply label-agreement scorer on real checkpoints.

This is the independent cross-check of the promotion gate: the scorer lives
in `src/c4/train.py` and is used by the trainer's promotion key.  This script
loads checkpoints through the canonical loader and reports the scorer's
number, so the promotion gate's metric can be verified against the
independent `tmp/measure_label_agreement.py` measurement (91.29 % on the
corrected final checkpoint).

Usage:
    ./.venv-c4/Scripts/python.exe -u tmp/measure_promotion_scorer.py <run_dir> [...]
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.c4.train import PerPlyAgreement, _align_perply_targets  # noqa: E402
from src.c4.agent import hard_to_cpu  # noqa: E402


def load_corpus(run: pathlib.Path):
    raw = []
    with open(run / "corpus.txt", "rb") as fh:
        for ln in fh.read().splitlines():
            if ln:
                raw.append(ln)
    return raw


# The run dir to score a checkpoint FILE against, set by --corpus-run.
_CORPUS_RUN: pathlib.Path | None = None


def _owner_run_dir(ckpt_file: pathlib.Path,
                   explicit: pathlib.Path | None = None) -> pathlib.Path | None:
    """Find the run dir that owns a checkpoint file.

    `explicit` (the --corpus-run flag) wins outright. Otherwise:

    1. the checkpoint's own directory and its ancestors (a snapshot inside
       `runs/_probe/v24/checkpoints/` resolves to `runs/_probe/v24`);
    2. any run dir under `runs/_probe/` that has the corpus + sidecar --
       but ONLY if there is exactly ONE. If several qualify, this returns
       None so the caller fails loudly instead of silently scoring against
       an unrelated corpus.

    That last guard matters: picking "the first one alphabetically" scored
    a 320,000-row V24 checkpoint against a 500-row scratch corpus and
    reported a misleadingly high number. A different corpus is a
    different measurement, not a comparable one.

    A run dir qualifies when it contains `corpus.txt` AND
    `perply_targets.npz` -- the two files the scorer needs.
    """
    if explicit is not None:
        if (explicit / "corpus.txt").exists() and (explicit / "perply_targets.npz").exists():
            return explicit
        return None
    # 1. ancestors of the checkpoint file
    for cand in [ckpt_file.parent] + list(ckpt_file.parents):
        if (cand / "corpus.txt").exists() and (cand / "perply_targets.npz").exists():
            return cand
    # 2. any qualifying run dir under runs/_probe -- only if unambiguous
    root = pathlib.Path("runs/_probe")
    cands = []
    if root.is_dir():
        for cand in sorted(root.iterdir()):
            if not cand.is_dir():
                continue
            if (cand / "corpus.txt").exists() and (cand / "perply_targets.npz").exists():
                cands.append(cand)
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        print(f"  NOTE: {len(cands)} run dirs under runs/_probe could own this "
              f"checkpoint:")
        for c in cands[:12]:
            nrows = sum(1 for ln in (c / "corpus.txt").read_bytes().splitlines() if ln)
            print(f"          {c}  ({nrows} corpus rows)")
        print("         pass --corpus-run <dir> to choose; refusing to guess, "
              "because scoring against a different corpus is not a "
              "comparable measurement.")
    return None


def main() -> int:
    # `--corpus-run <dir>` names the run dir whose corpus+sidecar to score
    # against, when a checkpoint FILE is passed. Required when more than one
    # run dir could own the checkpoint, because scoring against a different
    # corpus is not a comparable measurement.
    global _CORPUS_RUN
    argv = sys.argv[1:]
    if "--corpus-run" in argv:
        i = argv.index("--corpus-run")
        if i + 1 >= len(argv):
            print("ERROR: --corpus-run needs a directory argument")
            return 2
        _CORPUS_RUN = pathlib.Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    runs = [pathlib.Path(a) for a in argv]
    if not runs:
        print("usage: measure_promotion_scorer.py [--corpus-run <dir>] "
              "<run_dir|checkpoint.pt> [...]")
        return 2

    failed = False
    for run_arg in runs:
        run = run_arg
        ckpt_override = None
        corpus = run / "corpus.txt"
        sidecar = run / "perply_targets.npz"
        print(f"\n=== {run} ===")
        if not run.is_dir():
            # A checkpoint FILE was passed instead of a run directory.
            #
            # This is a legitimate use: the trainer PRUNES old step
            # snapshots (--keep-last), so a checkpoint that was pinned to a
            # stable path outside the run dir can no longer be scored via
            # its run. Refusing it outright would make a pinned artefact
            # unverifiable.
            #
            # Score it against the run dir that owns the corpus + sidecar.
            # The corpus/sidecar are identical for every checkpoint of a run,
            # so the metric is comparable.
            if run.suffix == ".pt" and run.is_file():
                owner = _owner_run_dir(run, _CORPUS_RUN)
                if owner is None:
                    print(f"  ERROR: {run} is a checkpoint file but no owning "
                          f"run dir (with corpus.txt + perply_targets.npz) "
                          f"could be determined unambiguously.")
                    print("         pass --corpus-run <dir> to name it, or "
                          "pass the RUN DIR that produced it.")
                    failed = True
                    continue
                print(f"  (checkpoint file; scoring against owner run "
                      f"{owner})")
                corpus = owner / "corpus.txt"
                sidecar = owner / "perply_targets.npz"
                run = owner
                ckpt_override = run_arg
            else:
                # A checkpoint file was passed instead of a run directory. Say so
                # loudly and FAIL: a silent "SKIP" here looks like a passing
                # measurement of zero checkpoints, which is exactly the kind of
                # vacuous result this project forbids.
                print(f"  ERROR: {run} is not a directory.")
                print("         usage: measure_promotion_scorer.py <run_dir> [...]")
                print("         (a run dir must contain corpus.txt + perply_targets.npz)")
                failed = True
                continue
        if not corpus.exists() or not sidecar.exists():
            missing = [p.name for p in (corpus, sidecar) if not p.exists()]
            print(f"  ERROR: missing {missing} in {run}")
            failed = True
            continue
        raw_lines = load_corpus(run)

        class A:
            pass

        a = A()
        a.perply_targets = str(sidecar)
        a.transcript = "v2"
        a.max_legacy_frac = 0.0
        a.corpus_source = "teacher"
        pt, po, nply = _align_perply_targets(a, raw_lines, "v2")
        if pt is None:
            print("  SKIP: sidecar not aligned")
            continue
        print(f"  corpus rows: {len(raw_lines)}  sidecar: {tuple(pt.shape)}")

        ckpts = []
        # An explicit checkpoint FILE was passed: score ONLY that file.
        if ckpt_override is not None:
            ckpts.append(ckpt_override)
        else:
            # The PROMOTED artifact is `checkpoints/best_hard.pt`. It must be
            # measured explicitly: the `step*_hard.pt` glob does NOT match it
            # (fnmatch("best_hard.pt","step*_hard.pt") is False), so a
            # run-root-only lookup silently never measured the promoted model.
            for p in (run / "hard_model_final.pt",
                      run / "checkpoints" / "best_hard.pt",
                      run / "best_hard.pt"):
                if p.exists() and p not in ckpts:
                    ckpts.append(p)
            if (run / "checkpoints").exists():
                for p in sorted((run / "checkpoints").glob("step*_hard.pt")):
                    if p not in ckpts:
                        ckpts.append(p)

        for ck in ckpts:
            try:
                # Load the HARD payload straight from the file.  Do NOT use
                # `_load_agent` + `export_hard()`: GFMarkovLM_V6 has no
                # `import_hard`, so a V6 checkpoint re-loaded that way keeps
                # ZERO float parameters, and re-exporting from them yields an
                # all-zero model (constant column 0).  The saved payload IS
                # the shipped artefact, so read it directly.
                raw = torch.load(str(ck), map_location="cpu", weights_only=False)
                payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
                hard = hard_to_cpu(payload)
                # hard_forward is a method on GFMarkovLM; build one with the
                # payload's own architecture.  It reads ONLY the integer
                # tables in `hard`, so the float params are irrelevant.
                from src.gf256_markov_lm_v6 import GFMarkovLM as V6
                from src.gf256_markov_lm import GFMarkovLM as V7
                d = int(payload["emb"].shape[1])
                layers = len(payload["coef"])
                fan_in = int(payload["coef"][0].shape[1])
                lags = tuple(int(x) for x in payload["lags"])
                cls = V7 if d >= 64 else V6
                mdl = cls(d=d, layers=layers, fan_in=fan_in, lags=lags)
                mdl.eval()
            except Exception as exc:
                print(f"  {ck.name}: LOAD FAILED {exc!r}")
                continue
            # ctx must match the trainer's --ctx (192 for the acceptance
            # run): trainer_window reads at index ctx-1, so a different ctx
            # shifts the read position.
            scorer = PerPlyAgreement(raw_lines, pt, po, nply,
                                     torch.device("cpu"), n_rows=512,
                                     hard=hard, model=mdl, ctx=192)
            res = scorer.score(mdl, max_positions=4000)
            print(f"  {ck.name:28s} LABEL {res['perply_agreement_pct']:6.2f}%"
                  f"  n={res['perply_n']}  mass={res['perply_mass']:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
