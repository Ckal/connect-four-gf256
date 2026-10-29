"""REGRESSION: the per-ply sidecar must be indexed by CORPUS ROW.

This is the test that was missing when the per-ply branch read the sidecar
by BATCH POSITION instead of by the corpus row the window came from.

The bug was invisible from the outside: every target it read was a real,
well-formed teacher label -- of a different game. Nothing crashed, the loss
went down, `LEGAL` climbed, and every artifact looked healthy. The only
symptom was that the model agreed with the teacher on ~20 % of positions
while agreeing with the PLAYED move on ~56 %.

The test below is model-free and fast: it rebuilds ONE training batch
through the trainer's own code path and asserts that

  1. the target the trainer uses equals the target of the corpus row the
     window was actually sampled from, and
  2. no supervised position has an all-zero target (which makes
     `_weighted_move_loss` silently substitute a one-hot on the PLAYED
     move -- a fabricated label).

Run:  ./.venv-c4/Scripts/python.exe tmp/test_perply_row_index.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.fmt import PLY_LEN  # noqa: E402
from src.c4.move_loss import (build_move_targets,  # noqa: E402
                              header_legality_from_x, header_visibility)
from src.c4.train import (_header_index_from_starts,  # noqa: E402
                          _move_byte_mask, _perply_corpus_rows,
                          _perply_rows_for_batch, build_batch)

CORPUS = ROOT / "runs/_probe/perply_v23_full/corpus_320k_fixed.txt"
SIDECAR = ROOT / "runs/_probe/perply_v23_full/merged_repaired.npz"
NROWS = 20000
BATCH = 16
CTX = 192


def main() -> int:
    # --- unit check of the SHIPPED helper, independent of any data ---
    ok = True
    _cu = torch.tensor([7, 100, 3], dtype=torch.long)
    _Wu = 5
    _ru = _perply_corpus_rows(_cu, _Wu, torch.device("cpu")).tolist()
    _eu = [7] * _Wu + [100] * _Wu + [3] * _Wu
    if _ru != _eu:
        print(f"FAIL: _perply_corpus_rows gave {_ru}, expected {_eu}")
        ok = False
    else:
        print("unit: _perply_corpus_rows maps each window to its corpus row "
              "-- OK")

    if not CORPUS.exists() or not SIDECAR.exists():
        print("SKIP: production corpus/sidecar not present")
        return 0 if ok else 1

    # allow_pickle=True is safe here: the .npz is this repository's own
    # relabeller output (tools/relabel_perply.py), written locally and never
    # downloaded. It stores a small object array for the teacher column.
    z = np.load(SIDECAR, allow_pickle=True)
    T = z["target"]
    PO = torch.tensor(z["ply_of"].astype(np.int64), dtype=torch.long)
    NP = torch.tensor(z["n_plies"].astype(np.int64), dtype=torch.long)

    examples = []
    with open(CORPUS, "rb") as fh:
        for i, line in enumerate(fh):
            if i >= NROWS:
                break
            examples.append(line.rstrip(b"\r\n"))

    device = torch.device("cpu")
    rng = np.random.default_rng(32)
    n_sup = 0
    n_match = 0
    n_dead = 0
    n_distinct_rows = set()
    n_batch_index_reads = 0

    for _ in range(30):
        x, y, chosen, starts = build_batch(
            examples, BATCH, device, rng, CTX,
            is_teacher=[True] * len(examples),
            max_legacy_frac=0.0, snap_to_end=True)
        move_mask = _move_byte_mask(y, starts)
        keep, move_col = build_move_targets(y, move_mask)
        keep = keep.reshape(-1)
        move_col = move_col.reshape(-1)
        hdr_index = _header_index_from_starts(starts, y.shape[1], device=device)
        legal3 = header_legality_from_x(x, move_mask, hdr_index)
        visible = header_visibility(starts, y.shape[1], device=device)
        keep = keep & visible.reshape(-1)
        legal_flat = legal3.reshape(-1, 7)

        slot = _perply_rows_for_batch(examples, chosen, starts, NP, PO,
                                      y.shape[1] + 1, PLY_LEN)
        flat_slot = slot.reshape(-1)
        hit = keep.reshape(-1) & (flat_slot >= 0)
        if not bool(hit.any()):
            continue

        W = y.shape[1]
        # THE SHIPPED ROW INDEX
        corpus_row = _perply_corpus_rows(chosen, W, device)
        idx = torch.nonzero(hit).squeeze(1)
        rows = corpus_row[idx].cpu().numpy()
        s_idx = flat_slot[idx].cpu().numpy()
        tgt = T[rows, s_idx].astype(np.float64)
        leg = legal_flat[idx].numpy()

        n_sup += len(s_idx)
        n_match += int((tgt.sum(axis=1) > 1e-6).sum())
        n_dead += int((tgt.sum(axis=1) <= 1e-6).sum())
        n_distinct_rows.update(int(r) for r in rows)
        # A read that is really a batch index would land in 0..B-1.
        n_batch_index_reads += int((rows < BATCH).sum())

    print("=== per-ply sidecar row index ===")
    print(f"  supervised positions : {n_sup}")
    print(f"  targets with mass    : {n_match}")
    print(f"  all-zero targets     : {n_dead}")
    print(f"  distinct corpus rows : {len(n_distinct_rows)}")
    print(f"  reads inside 0..{BATCH - 1} (batch-index range): "
          f"{n_batch_index_reads}")

    ok = True
    if n_dead > 0:
        print("FAIL: all-zero targets would trigger the silent played-move "
              "fallback")
        ok = False
    if n_distinct_rows and len(n_distinct_rows) <= BATCH:
        print(f"FAIL: only {len(n_distinct_rows)} distinct corpus rows read "
              f"for {n_sup} supervised positions -- the sidecar is being "
              f"indexed by batch position")
        ok = False
    if n_sup and n_match / n_sup < 0.999:
        print(f"FAIL: only {100.0 * n_match / n_sup:.2f}% of supervised "
              f"positions have a live target")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
