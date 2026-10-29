"""Multi-label (per-ply) target sidecar for the trainer.

The V23 sidecar carries ONE (N,7) target per transcript, aligned to the
transcript's FINAL ply. The production path therefore supervises exactly
one move position per 193-byte window -- 16 real labels per batch
(6.89%).

This module defines a PER-PLY sidecar format so a single window can carry
up to 19 targets (one per ply inside it):

    perply_targets.npz
        target  : (N, P, 7) float32   -- target distribution per ply
        ply_of  : (N, P) int16        -- which ply each slot refers to
        n_plies : (N,) int16          -- how many slots are valid

Slot p of row i describes the position BEFORE ply p of transcript i, and
its target is the teacher's move for that position.

The loader returns, for a batch, a (B, W) grid of sidecar row ids per
TOKEN, exactly like `_soft_rows_for_batch` does for the single-label
sidecar -- but now with one row id per supervised ply instead of one per
window.
"""
from __future__ import annotations

import numpy as np
import torch

from .fmt import MOVE_OFFSET


def load_perply_targets(path: str):
    """Load a per-ply target sidecar.

    Returns (target (N,P,7) float32 tensor, ply_of (N,P) long tensor).
    """
    z = np.load(path)
    if "target" not in z.files:
        raise SystemExit(f"{path} has no 'target' array")
    T = torch.tensor(z["target"], dtype=torch.float32)
    if "ply_of" in z.files:
        P = torch.tensor(z["ply_of"].astype(np.int64), dtype=torch.long)
    else:
        P = torch.arange(T.shape[1]).view(1, -1).expand(T.shape[0], -1).contiguous()
    return T, P


def perply_rows_for_batch(examples, chosen, starts, ply_bank, ply_of,
                           width, ply_len):
    """(B, W) long: sidecar (row, slot) index per token, or -1.

    For window k of example `chosen[k]` starting at byte `st`:

      * the ply that OWNS position t (the position predicting stream byte
        st+t+1) is k_ply = (st + t + 1) // ply_len,
      * the labelled position of ply k_ply is the one BEFORE that ply's
        move byte, so its target belongs to position t if
        k_ply * ply_len + (ply_len - 2) is the byte being predicted,

    which is exactly the move byte of ply k_ply. So a target for ply p
    lands on the token that predicts ply p's move byte.

    The grid stores, per token, the flattened index row * P + slot of the
    target that applies there, or -1.
    """
    B, W = len(chosen), width - 1
    out = torch.full((B, W), -1, dtype=torch.long)
    for k in range(B):
        row = int(chosen[k])
        st = int(starts[k])
        nply = len(examples[row]) // ply_len
        P = ply_of[row].shape[0]
        for p in range(nply):
            if p >= P:
                break
            # The move byte of ply p sits at absolute byte
            # p*PLY_LEN + MOVE_OFFSET (9), and the token that PREDICTS it
            # is at t = abs_move - st - 1 (because y[t] predicts stream
            # byte st+t+1). That token is the position BEFORE ply p's
            # move -- exactly the position a teacher would label.
            abs_move = p * ply_len + MOVE_OFFSET
            if not (st <= abs_move < st + width):
                continue
            pos = abs_move - st - 1
            if not (0 <= pos < W):
                continue
            out[k, pos] = row * P + p
    return out


def gather_perply_dist(row_flat, ply_bank, keep, n_slots, device,
                       soft_w=1.0):
    """Build the (N, 7) target distribution for the kept positions.

    `row_flat` is the flattened (B*W,) grid from perply_rows_for_batch,
    `keep` the flattened loss mask. Returns a (N,7) tensor aligned with
    `keep` (zero where keep is False).
    """
    idx = row_flat.to(device)
    valid = keep & (idx >= 0)
    dist = torch.zeros(int(keep.numel()), 7, device=device)
    if not bool(valid.any()):
        return dist, valid
    sel = idx[valid]
    tgt = ply_bank.reshape(-1, 7)[sel]
    dist[valid] = tgt
    return dist, valid
