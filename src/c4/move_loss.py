"""Masked 7-way move loss: the core of what V2 buys us.

The current loss is a 256-way softmax over ALL byte values, applied at every
move position. At a move byte only 7 outcomes are legal (COL_BYTE_OFFSET ..
COL_BYTE_OFFSET+6), so 249 of 256 logits -- and 249/256 of the gradient
signal -- are spent on values that can never be correct. The model must spend
capacity learning the alphabet instead of the game.

V2 makes the fix cheap: the column byte of the CURRENT header holds one bit
per row, so a column is legal iff its height < 6, and the legality mask is
read straight out of the header that is already in the window.

LEGALITY SOURCE, AND WHY NOT "just use 7 always"
------------------------------------------------
A full column is illegal. With a fixed 7-way softmax we would happily train
the model to output a full column and then have the UI drop it at serve time,
teaching a policy that is never actually played. The mask must therefore be
per-position, from the board -- not a constant.

This module returns, for a batch of windows:
  col_logits : (N, 7) logits restricted to the 7 legal columns
  target     : (N,)   which legal column the row's move byte names
  keep       : (N,)   positions that are real move bytes with a valid target
"""
from __future__ import annotations

import numpy as np
import torch


def move_loss(logits_flat: torch.Tensor,
              y_flat: torch.Tensor,
              keep: torch.Tensor,
              move_col: torch.Tensor,
              col_idx: torch.Tensor) -> torch.Tensor:
    """Masked 7-way cross-entropy over legal columns.

    logits_flat : (P, 256) raw logits for every position in the batch
    y_flat      : (P,)    target byte value
    keep        : (P,)    bool, real move bytes with a legal target
    move_col    : (P,)    long, 0..6 = which column this position's move byte
                           refers to (target - COL_BYTE_OFFSET)
    col_idx     : (P, 7)  long, column index of each of the 7 logits, i.e.
                           COL_BYTE_OFFSET + arange(7) broadcast; kept for
                           clarity and future per-column weighting.

    Returns a scalar mean NLL over the kept positions.
    """
    from .fmt import COL_BYTE_OFFSET

    if keep.sum() == 0:
        return logits_flat.sum() * 0.0
    sel_logits = logits_flat[keep]                       # (N, 256)
    sub = sel_logits[:, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]  # (N, 7)
    lp = torch.log_softmax(sub, dim=-1)
    tgt = move_col[keep]                                 # (N,) in 0..6
    return -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean()


def build_move_targets(y: torch.Tensor,
                       move_mask: torch.Tensor,
                       n_cols: int = 7):
    """Split a padded (B,T) target batch into the 7-way loss's inputs.

    Returns (keep, move_col) with the SAME flattened shape as y, where keep
    marks a position that is (a) a real move byte and (b) within 0..n_cols-1
    of COL_BYTE_OFFSET. Positions that are move bytes but somehow outside the
    range are dropped rather than crashed on: a corrupt row must not take the
    whole run down.
    """
    from .fmt import COL_BYTE_OFFSET

    flat = y.reshape(-1)
    flat_mask = move_mask.reshape(-1)
    vals = flat - COL_BYTE_OFFSET
    ok = flat_mask & (vals >= 0) & (vals < n_cols)
    return ok, torch.where(ok, vals, torch.zeros_like(vals))


def column_height(header: torch.Tensor) -> torch.Tensor:
    """(...,) v2 column bytes -> (...,) height 0..6.

    The v2 layout is `0x80 | (1 << h) | ownership_bits`, so the height is the
    INDEX OF THE HIGHEST SET BIT below the flag -- not a popcount. Ownership
    bits sit *below* the marker, so they never change which bit is highest:
    0x81 (h=0, empty) and 0x83 (h=1, P1 stone) both have low bits set, but
    0x81 -> 0 and 0x83 -> 1. Getting this wrong silently turns every EMPTY
    column into a height-1 column, which forges a stone that is not there.

    This mirrors `transcript_v2.decode_column`, which uses `bit_length() - 1`.
    """
    from .fmt import COLUMN_FLAG

    is_col = (header & COLUMN_FLAG) != 0
    v = header & ~COLUMN_FLAG & 0x7F            # marker + ownership only
    # highest set bit, via exact integer compares (no float log2)
    # ASCENDING: each true test overwrites, so the LARGEST matching h wins
    # last. Iterating descending would let the smallest h overwrite the
    # correct largest one (0xd5 -> height 1 instead of 6).
    height = torch.zeros_like(v)
    for h in range(1, 7):
        height = torch.where(v >= (1 << h), torch.full_like(height, h), height)
    # v == 0 means "flag set but no height marker" -- malformed, not height 0
    return torch.where(is_col & (v > 0), height, torch.full_like(height, -1))


def header_legality(header: torch.Tensor, max_height: int = 6) -> torch.Tensor:
    """(..., 7) header bytes -> (..., 7) bool, True where that column is legal.

    `header` holds the 7 column bytes of one board, in column order. A column
    is legal iff it really is a v2 column byte and its height < max_height.
    """
    h = column_height(header)
    return (h >= 0) & (h < max_height)


def header_legality_from_x(x: torch.Tensor,
                           move_mask: torch.Tensor,
                           header_index: torch.Tensor):
    """Gather each position's 7 header bytes, then read legality per column.

    x            : (B, T)  input bytes
    move_mask    : (B, T)  bool -- True at real move bytes
    header_index : (B, T)  long -- x-index of THIS position's 9-byte header,
                    i.e. the header of the board on which the move was played.

    The 7 header slots describe 7 DIFFERENT columns, so the mask must gather
    slot j to decide column j; broadcasting one bit over all 7 columns would
    be wrong.
    """
    b, t = move_mask.shape
    n = 7
    W = x.shape[1]
    hi = header_index.clamp(min=0, max=W - n)                 # (B, T)
    # Gather the 7 header bytes for every position at once: flatten x to
    # (B, T*W) so each (row, position) can index its own 7 bytes.
    flat_x = x.unsqueeze(1).expand(b, t, W).reshape(b, t * W)
    offs = torch.arange(n, device=x.device).view(1, 1, n)    # (1, 1, 7)
    idx = (hi.unsqueeze(-1) + offs).clamp(min=0, max=W - 1)    # (B, T, 7)
    hdr = torch.gather(flat_x, 1, idx.reshape(b, t * n)).reshape(b, t, n)
    return header_legality(hdr) & move_mask.unsqueeze(-1)


def move_loss(logits_flat: torch.Tensor,
              y_flat: torch.Tensor,
              keep: torch.Tensor,
              move_col: torch.Tensor,
              legal: torch.Tensor | None = None):
    """Masked 7-way cross-entropy over legal columns.

    logits_flat : (P, 256) raw logits, flattened batch
    y_flat      : (P,)    target byte value
    keep        : (P,)    bool -- real move byte naming one of the 7 columns
    move_col    : (P,)    long 0..6 -- which column the target refers to
    legal       : (P, 7)  bool -- which columns were legal (optional)

    A position whose target is ILLEGAL -- impossible with a correct corpus,
    but reachable from a mislabelled row -- is DROPPED rather than crashed on,
    and a fully-masked row falls back to an unmasked softmax so it cannot
    produce NaN.
    """
    from .fmt import COL_BYTE_OFFSET

    if keep.sum() == 0:
        return logits_flat.sum() * 0.0
    if legal is not None:
        keep = keep & legal[keep].any(dim=1)
        if keep.sum() == 0:
            return logits_flat.sum() * 0.0
    sel = logits_flat[keep]                              # (N, 256)
    sub = sel[:, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]    # (N, 7)
    if legal is not None:
        lg = legal[keep]
        dead = ~lg.any(dim=1)
        if dead.any():
            lg = lg.clone()
            lg[dead] = True
        sub = sub.masked_fill(~lg, float("-inf"))
    lp = torch.log_softmax(sub, dim=-1)
    tgt = move_col[keep]                                 # (N,)
    return -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).mean()


def blend_soft_targets(base: torch.Tensor, soft: torch.Tensor, weight: float):
    """Replace one-hot move targets with the corpus's soft target.

    The trainer normally derives its 7-way target from the move byte alone,
    which is one-hot. For imported positions that byte is an argmax over
    equally-optimal columns, so training on it re-injects exactly the
    arbitrary low-index bias the soft target exists to remove -- 41.6% of
    our positions have ties, and argmax picks the smallest column every time.

    `base` is (N, 7) one-hot; `soft` is (N, 7) distribution; `weight`
    interpolates so the one-hot term keeps a little mass on a legal move
    (never all of it, which would break the tie-resolution signal).
    """
    if soft is None or weight <= 0:
        return base
    return (1.0 - weight) * base + weight * soft


def build_move_targets(y: torch.Tensor, move_mask: torch.Tensor, n_cols: int = 7):
    """(keep, move_col) flattened, same shape as y flattened.

    `keep` marks a position that is a real move byte AND whose value names one
    of the n_cols columns. Header, turn, newline, padding and out-of-range
    move bytes are all excluded.
    """
    from .fmt import COL_BYTE_OFFSET

    flat = y.reshape(-1)
    flat_mask = move_mask.reshape(-1)
    vals = flat - COL_BYTE_OFFSET
    ok = flat_mask & (vals >= 0) & (vals < n_cols)
    return ok, torch.where(ok, vals, torch.zeros_like(vals))


def header_visibility(starts, width: int, ply_len: int = 10, device=None):
    """(B, T) bool: is this position's 7-byte header fully inside the window?

    A window may begin mid-ply. The header of the board a move was played on
    sits at absolute byte k*PLY_LEN, which lands BEFORE the window for the
    first ply or two of any window that starts off-grid. Legality cannot be
    read from a header that is not in the window, so those positions must be
    EXCLUDED from the masked loss rather than guessed -- masking them to
    "all 7 legal" would silently train the model to pick a full column.

    Loss of coverage is bounded and tiny: at ctx=192 a window spans ~19 plies
    and only the first ~1-2 are affected.
    """
    import torch as _t
    from .fmt import PLY_LEN

    ply_len = ply_len or PLY_LEN
    s = _t.as_tensor(starts, dtype=_t.long, device=device)
    t = _t.arange(width, device=s.device).view(1, -1)
    k = (s.view(-1, 1) + t + 1) // ply_len
    local = k * ply_len - s.view(-1, 1)            # x is window-local
    return (local >= 0) & (local + 7 <= width)


__all__ = ["column_height", "header_legality", "header_legality_from_x",
           "move_loss", "build_move_targets", "header_visibility"]
