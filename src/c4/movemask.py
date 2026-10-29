"""Positional move-byte mask, and the guard that keeps the two formats honest.

THE PROBLEM
-----------
train.py's `_move_byte_mask` finds move bytes BY VALUE:

    (y >= COL_BYTE_OFFSET) & (y <= COL_BYTE_OFFSET + 6)      # 0x41..0x47

That works in V1 by luck, and BREAKS in V2:

  * V1 header columns are 0x30..0x36 (heights + '0'), so the only bytes in
    0x41..0x47 are the move byte and the `last` byte.
  * `last` is the PREVIOUS ply's move, already written, so it is trivially
    predictable. Counting it inflates accuracy with a free win per ply --
    tg_masksplit2.py measured exactly that (67.41% on copied-last vs
    73.77% on real moves, i.e. the blend was deflated, and both numbers were
    meaningless as a strength proxy).

V2 does not make this worse -- `last` still lives at 0x41..0x47 by design
(keeping the move range disjoint from the 0x81..0xFF columns is what makes
COL_SLICE safe for the argmax). It makes it IRRECOVERABLE by value, because
position is now the ONLY thing that distinguishes them.

THE FIX
-------
Move bytes are at a FIXED position in the byte stream: ply k's move byte is
at k * PLY_LEN + MOVE_OFFSET. `build_batch` already returns the byte offset
where each window began (`starts`), so the mask is pure arithmetic -- no
inspection of byte values at all, and it cannot be fooled by the `last` byte,
by padding, or by a full column's 0xFF.

This module is the ONLY place that should decide what a move byte is.
"""
from __future__ import annotations

import numpy as np

from . import fmt


def move_mask_for_window(starts, ctx: int, device=None) -> np.ndarray:
    """(B, ctx) bool: True where the TARGET byte y[t] is a move byte.

    `starts[k]` is the byte offset into the dense sequence where window k
    began, exactly as returned by `build_batch`. Because `y[t]` predicts
    sequence byte `start + t + 1`, y[t] is ply k's move byte iff

        (start + t + 1) % PLY_LEN == MOVE_OFFSET

    Padding bytes are never selected: pad sits past the sequence end, and
    pad == 0x0A is not congruent to MOVE_OFFSET mod PLY_LEN.
    """
    version = fmt.DEFAULT_VERSION
    mod = fmt.get(version)
    ply_len = mod.PLY_LEN
    move_off = mod.MOVE_OFFSET

    starts = np.asarray(starts, dtype=np.int64).reshape(-1, 1)   # (B,1)
    t = np.arange(ctx, dtype=np.int64).reshape(1, -1)             # (1,ctx)
    return ((starts + t + 1) % ply_len) == move_off


def move_mask_from_starts(starts, ctx: int, device=None):
    """Same mask as a torch tensor on `device`."""
    import torch

    m = move_mask_for_window(starts, ctx)
    t = torch.tensor(m, dtype=torch.bool, device=device)
    return t


def move_byte_positions(start: int, ctx: int, version: str | None = None):
    """Absolute sequence offsets of the move bytes inside one window.

    Used by the diagnostics that must slice tags by POSITION rather than by
    byte value (tg_tagcheck2.py, tg_opsuite.py).
    """
    mod = fmt.get(version)
    ply_len = mod.PLY_LEN
    off = mod.MOVE_OFFSET
    return [start + t + 1 for t in range(ctx)
            if (start + t + 1) % ply_len == off]


def validate_stream(seq: bytes, version: str | None = None) -> None:
    """Raise if a dense sequence is not cleanly framed for `version`.

    Cheap structural test used before training on a corpus file, so a V1 file
    can never be fed to a V2 run (which would read pure noise and produce a
    plausible-looking but meaningless loss).
    """
    mod = fmt.get(version)
    ply_len = mod.PLY_LEN
    if not seq:
        return
    if len(seq) % ply_len:
        raise ValueError(
            f"{len(seq)}B dense row is not a multiple of PLY_LEN={ply_len} "
            f"(wrong transcript version?)")
    head = seq[0]
    if version == "v1":
        # V1 column bytes are HGT_ZERO..HGT_TOP (0x30..0x36)
        if not (0x30 <= head <= 0x36):
            raise ValueError(
                f"v1 expected a height byte 0x30..0x36 first, got {head:#04x}")
    else:
        if not head & mod.COLUMN_FLAG:
            raise ValueError(
                f"v2 expected a column byte with 0x80 set, got {head:#04x} "
                f"(this looks like a v1 row)")


__all__ = ["move_mask_for_window", "move_mask_from_starts",
           "move_byte_positions", "validate_stream"]