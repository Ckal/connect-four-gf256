"""Generate threat-decision training positions with unambiguous hard labels.

Motivation
----------
Self-play corpus training learns from *trajectories*: "given this board, the
game that eventually was won used this move."  In a 40-ply game only ~1 move
actually decides the win; the other 39 are noise.  That is why V7 never learned
threat recognition (vs-heuristic stuck at W0 L32).

This module instead emits **single decision points** where the correct column
is forced:

  win_now     -- I have an immediate win; take it.
  block_now   -- the opponent has an immediate win and I cannot win; block it.
  make_threat -- no immediate win either way, but one move creates a double
                 threat (two winning squares the opponent cannot both cover),
                 so I keep the initiative.
  safe        -- nothing forcing exists; play the strongest move that does not
                 hand the opponent an immediate win.

Every label is a single byte 0x41..0x47, so a position costs 10 bytes on disk
and a million positions cost ~10 MB.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

from .board import Board, P1, P2, COLS, ROWS, WIN_LEN, is_terminal
from .opponents import (heuristic_policy, lookahead_policy, minimax_policy,
                        random_policy)


def _snap(b: Board) -> tuple:
    return (b.mask, b.p1, b.p2, b.turn, b.last_move, list(b.heights))


def _restore(b: Board, s: tuple) -> None:
    b.mask, b.p1, b.p2, b.turn, b.last_move, heights = s
    b.heights = list(heights)


def _with_move(b: Board, col: int, side: int) -> tuple:
    """Drop `col` for `side` in place and return the snapshot needed to undo it.

    NOTE: Board.is_win() derives the winner as `P2 if turn == P1 else P1`, so
    `turn` MUST be flipped to the opponent after the drop or is_win() reports on
    the wrong player. `_restore` puts everything back.
    """
    row = b.heights[col]
    bit = 1 << (row * COLS + col)
    s = _snap(b)
    if side == P1:
        b.p1 |= bit
    else:
        b.p2 |= bit
    b.mask |= bit
    b.heights[col] = row + 1
    b.last_move = col
    b.turn = P2 if side == P1 else P1
    return s


def _winning_cols(b: Board, player: int) -> list[int]:
    """Columns where dropping for `player` wins immediately.

    Reads the winning line off the bitboard directly instead of calling
    is_win(), so it does not depend on b.turn and avoids the snapshot dance.
    """
    me = b.p1 if player == P1 else b.p2
    opp = b.p2 if player == P1 else b.p1
    out = []
    for c in b.valid_actions():
        row = b.heights[c]
        bit = 1 << (row * COLS + c)
        nxt = me | bit
        if _win_from(nxt, opp, row, c):
            out.append(c)
    return out


def _win_from(me: int, opp: int, row: int, col: int) -> bool:
    """True if `me` has WIN_LEN in a line through (row, col) once opp's pieces
    are treated as blockers. Mirrors Board.is_win()'s direction set."""
    def mine(r: int, c: int) -> bool:
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        return bool(me >> (r * COLS + c) & 1)

    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run = 1
        for s in (1, -1):
            k = 1
            while mine(row + dr * k * s, col + dc * k * s):
                run += 1
                k += 1
        if run >= WIN_LEN:
            return True
    return False


def _creates_double_threat(b: Board, side: int) -> list[int]:
    """Columns whose play leaves us >=2 winning squares next turn while the
    opponent has no winning reply -- an unstoppable fork."""
    opp = P2 if side == P1 else P1
    out = []
    for c in b.valid_actions():
        s1 = _with_move(b, c, side)
        if b.is_win() == side:
            _restore(b, s1)
            continue
        opp_wins = _winning_cols(b, opp)
        ok = len(opp_wins) <= 1
        if ok:
            if opp_wins:
                # If the opponent's only winning square is answered, the fork
                # must still hold on the resulting board.
                s2 = _with_move(b, opp_wins[0], opp)
                ok = len(_winning_cols(b, side)) >= 2
                _restore(b, s2)
            else:
                ok = len(_winning_cols(b, side)) >= 2
        if ok:
            out.append(c)
        _restore(b, s1)
    return out


def classify(b: Board, side: int) -> tuple[str, int] | None:
    """Return (category, forced_column) or None if the position is unusable."""
    if is_terminal(b):
        return None

    mine = _winning_cols(b, side)
    if mine:
        return "win_now", min(mine)

    opp = P2 if side == P1 else P1
    opp_win = _winning_cols(b, opp)
    if opp_win:
        return "block_now", min(opp_win)

    threats = _creates_double_threat(b, side)
    if threats:
        return "make_threat", min(threats)

    # Nothing forcing: keep the ladder's own choice when it is available.
    safe = b.valid_actions()
    if not safe:
        return None
    try:
        pref = minimax_policy(b, [])
    except Exception:
        pref = None
    if pref in safe:
        return "safe", pref
    return "safe", min(safe, key=lambda c: (abs(c - 3), c))


def _walk(rng: random.Random) -> tuple[Board, int, int, str] | None:
    """Play a mixed game; return the first position with a forced label."""
    ladder = [random_policy, lookahead_policy, heuristic_policy]
    weights = [0.45, 0.30, 0.25]
    b = Board()
    for _ in range(rng.randint(6, 40)):
        if is_terminal(b):
            return None
        pol = rng.choices(ladder, weights=weights, k=1)[0]
        c = pol(b, [])
        if c not in b.valid_actions():
            c = rng.choice(b.valid_actions())
        b.apply_move(c)
        side = b.turn
        res = classify(b, side)
        if res is not None:
            return b, side, res[1], res[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0xC4C4_0001)
    args = ap.parse_args()

    from .transcript import COL_BYTE_OFFSET, encode_state

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    seen: set[bytes] = set()
    written = 0
    t0 = time.perf_counter()

    with out.open("wb") as f:
        while written < args.n:
            got = _walk(rng)
            if got is None:
                continue
            b, side, col, cat = got
            line = encode_state(b, b.last_move, side) + bytes([COL_BYTE_OFFSET + col])
            if line in seen:
                continue
            seen.add(line)
            f.write(line + b"\n")
            counts[cat] += 1
            written += 1
            if written % 25_000 == 0:
                el = time.perf_counter() - t0
                print(f"  {written}/{args.n}  {written / el:.0f} pos/s  "
                      f"{el:.0f}s", flush=True)

    dt = time.perf_counter() - t0
    size = out.stat().st_size
    print(f"wrote {written} positions -> {out}")
    print(f"size {size / 1e6:.2f} MB ({size / max(written, 1):.1f} B/pos)")
    for k, v in counts.most_common():
        print(f"  {k:12s} {v:8d}  {v / max(written, 1) * 100:5.1f}%")
    print(f"elapsed {dt:.1f}s ({written / max(dt, 1e-9):.0f} pos/s)")


if __name__ == "__main__":
    main()