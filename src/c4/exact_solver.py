"""Exact Connect Four endgame solver for positions with <= `max_empty` cells.

The user is right: "there is an algo for calculationg the last 7 steps."
A Connect Four position with 7 or fewer empty cells is fully solvable, and
below that it becomes very cheap. This module implements that properly, in
the style of Pascal Pons' *Solving methods for Connect 4*: bitboard negamax
with a transposition table, returning WIN / DRAW / LOSS rather than a score,
so "exact" is a well-defined, testable claim.

Deliberate properties
---------------------
* Win detection uses START MASKS (below), so no row or diagonal wrap can
  fabricate a four. Whole-mask shift scans are unsound here: stride 1 wraps
  between rows, and stride COLS-1 = 6 is coprime with 7, so an "up-right"
  diagonal leaves the board and re-enters another row. src/c4/board.py
  documents the same trap and sidesteps it by anchoring to the last move;
  the search here has no `last_move`, so it needs the masks instead.
* A move that completes a four resolves BEFORE recursing -- the opponent
  never moves after the game ends.
* Positions are keyed by (mine, theirs) with the seats swapped each ply, so
  a single table serves both colours.
* The node budget RAISES rather than degrading. Callers get None for
  UNPROVED. This module never presents an unproved position as a draw.

Bitboard layout is the project-wide one (src/c4/board.py): row-major,
7 columns x 6 rows, bit index = r * COLS + c, row 0 = the bottom row.

Public API
----------
    ExactSolver(max_empty=7, budget=...)    bounded exact solver
    .solve(b, side)                        -> WIN / DRAW / LOSS / None
    .best_move(b, side)                    -> column, or None if unproved
    .solve_batch([(b, side), ...])          shares one table (much faster)
    solve_board(b, side)                    one-shot convenience
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.c4.board import COLS, ROWS, Board  # noqa: E402

WIN = 1
DRAW = 0
LOSS = -1

NAME = {WIN: "WIN", DRAW: "DRAW", LOSS: "LOSS"}

# ---------------------------------------------------------------------------
# Canonical win detection -- START MASKS
#
# For an axis of stride s, a run of k starting at cell i is on the board iff
# bit i lies in the axis' start mask. Intersecting the accumulated AND with
# that mask means a surviving bit proves all k cells are set AND in bounds, so
# a wrap can never manufacture a run.
# ---------------------------------------------------------------------------
_FOUR_START: Dict[int, int] = {}
for _s in (1, COLS, COLS + 1, COLS - 1):
    _m = 0
    if _s == 1:                      # horizontal: any row, start col 0..3
        for _r in range(ROWS):
            _m |= 0b1111 << (_r * COLS)
    elif _s == COLS:                 # vertical: start row 0..2, any col
        for _r in range(3):
            for _c in range(COLS):
                _m |= 1 << (_r * COLS + _c)
    elif _s == COLS + 1:             # diag \: start row 0..2, col 0..3
        for _r in range(3):
            _m |= 0b1111 << (_r * COLS)
    else:                            # diag /: start row 0..2, col 3..6
        for _r in range(3):
            _m |= 0b1111 << (_r * COLS + 3)
    _FOUR_START[_s] = _m

# Valid starts for a run of 3 are a superset of those for a run of 4: if
# i, i+s, i+2s, i+3s are all on the board then i+s starts a valid 3-run. So
# S(k-1) = S(k) | (S(k) << s). The shift is by the AXIS STRIDE, never by 1.
THREE_START: Dict[int, int] = {s: m | (m << s) for s, m in _FOUR_START.items()}

AXES: Tuple[int, ...] = tuple(_FOUR_START.keys())


def has_four(mask: int) -> bool:
    """Exact: does `mask` contain a 4-in-a-row on any of the four axes?"""
    for s, start in _FOUR_START.items():
        if mask & (mask >> s) & (mask >> (2 * s)) & (mask >> (3 * s)) & start:
            return True
    return False


def has_three(mask: int) -> bool:
    """Exact: does `mask` contain a 3-in-a-row on any of the four axes?"""
    for s, start in THREE_START.items():
        if mask & (mask >> s) & (mask >> (2 * s)) & start:
            return True
    return False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def move_order(cols: Sequence[int]) -> List[int]:
    """Centre-first ordering -- the single biggest pruning win here."""
    prio = {3: 0, 2: 1, 4: 1, 1: 2, 5: 2, 0: 3, 6: 3}
    return sorted(cols, key=lambda c: prio.get(c, 9))


class ExactSolver:
    """Bounded exact solver. UNPROVED is None, never a guessed value."""

    def __init__(self, max_empty: int = 7, budget: int = 4_000_000):
        self.max_empty = max_empty
        self.budget = budget
        self.table: Dict[Tuple[int, int], int] = {}
        self.nodes = 0
        self.unproved = 0
        self.solved = 0
        self.table_hits = 0

    # -- internals ------------------------------------------------------
    def _negamax(self, mine: int, theirs: int,
                 heights: List[int], empties: int) -> int:
        """Result for `mine` (the side to move): WIN / DRAW / LOSS."""
        key = (mine, theirs)
        cached = self.table.get(key)
        if cached is not None:
            self.table_hits += 1
            return cached

        # `theirs` holding a four means the side to move already lost, because
        # the previous ply ended the game. Unreachable in a clean search (we
        # return on the winning move) but cheap insurance against a bad table.
        if has_four(theirs):
            self.table[key] = LOSS
            return LOSS

        self.nodes += 1
        if self.nodes > self.budget:
            raise TimeoutError("exact solve exceeded node budget")

        if empties == 0:
            self.table[key] = DRAW
            return DRAW

        best = LOSS
        for col in move_order([c for c in range(COLS) if heights[c] < ROWS]):
            h = heights[col]
            bit = 1 << (h * COLS + col)
            heights[col] = h + 1
            try:
                if has_four(mine | bit):
                    # we win immediately; the opponent never replies
                    val = WIN
                else:
                    val = -self._negamax(theirs, mine | bit,
                                          heights, empties - 1)
            finally:
                heights[col] = h
            if val == WIN:
                self.table[key] = WIN
                return WIN
            if val == DRAW:
                best = DRAW
        self.table[key] = best
        return best

    # -- public ---------------------------------------------------------
    def solve(self, b: Board, side: int = 1) -> Optional[int]:
        """Exact status for the side to move, or None if UNPROVED.

        `b` must be a non-terminal position; `side` is 1 or 2 and must match
        the side to move. None means "outside the bounded subspace" or "budget
        ran out" -- never a draw.
        """
        empties = COLS * ROWS - (b.p1.bit_count() + b.p2.bit_count())
        if empties > self.max_empty:
            self.unproved += 1
            return None
        mine, theirs = (b.p1, b.p2) if side == 1 else (b.p2, b.p1)
        heights = list(b.heights)
        try:
            out = self._negamax(mine, theirs, heights, empties)
        except TimeoutError:
            self.unproved += 1
            return None
        self.solved += 1
        return out

    def best_move(self, b: Board, side: int = 1) -> Optional[int]:
        """A column achieving the solved status, or None if UNPROVED.

        Must return a move that actually delivers the status `solve()` just
        reported. An earlier version kept a `fallback` set to the first legal
        column, so a DRAW position whose centre-first first move happens to
        lose returned that losing move -- validate_exact_solver.py layer C
        caught it as "claimed DRAW but opponent gets WIN".
        """
        status = self.solve(b, side)
        if status is None:
            return None
        mine, theirs = (b.p1, b.p2) if side == 1 else (b.p2, b.p1)
        empties = COLS * ROWS - (b.p1.bit_count() + b.p2.bit_count())
        heights = list(b.heights)
        first_legal: Optional[int] = None
        drawing: Optional[int] = None
        for col in move_order([c for c in range(COLS) if heights[c] < ROWS]):
            h = heights[col]
            bit = 1 << (h * COLS + col)
            heights[col] = h + 1
            try:
                if has_four(mine | bit):
                    return col            # wins on the spot
                val = -self._negamax(theirs, mine | bit, heights, empties - 1)
            except TimeoutError:
                continue
            finally:
                heights[col] = h
            if val == WIN:
                return col
            if first_legal is None:
                first_legal = col
            if val == DRAW and drawing is None:
                drawing = col
        # WIN: already returned above. DRAW: prefer a move that draws.
        # LOSS: every move loses, so any legal move is equally correct.
        return drawing if status == DRAW else first_legal

    def solve_batch(self, items: Sequence[Tuple[Board, int]]
                    ) -> List[Optional[int]]:
        """Solve many positions sharing one table."""
        return [self.solve(b, side) for b, side in items]

    def reset(self) -> None:
        self.table.clear()
        self.nodes = 0
        self.unproved = 0
        self.solved = 0
        self.table_hits = 0

    def stats(self) -> Dict[str, int]:
        return {
            "solved": self.solved,
            "unproved": self.unproved,
            "nodes": self.nodes,
            "table_size": len(self.table),
            "table_hits": self.table_hits,
        }


def solve_board(b: Board, side: int = 1, max_empty: int = 7,
                budget: int = 4_000_000) -> Optional[int]:
    """One-shot solve with a fresh solver (slow; prefer one reused solver)."""
    return ExactSolver(max_empty=max_empty, budget=budget).solve(b, side)


def clone(b: Board) -> Board:
    return copy.deepcopy(b)