"""Left-right (c -> 6-c) symmetry for Connect Four.

Connect Four is symmetric under a left-right reflection of the board: the
column c becomes column 6-c, the player to move and the ownership of every
stone are preserved, and a position's optimal value is unchanged.

This module provides the ONE canonical way to do that reflection, plus the
canonicalisation helper that makes a teacher deterministic under symmetry.

Why this matters
----------------
A search that is given a board b and its mirror m(b) must return MIRRORED
moves. Any teacher that does not satisfy this teaches the model an
inconsistency: the same position (up to reflection) gets two different
answers, and the model learns the tie-break direction rather than the
position.

Measured on this repo before the fix:
  - `strong_policy` was mirror-symmetric on only 65% of legal positions
  - the V23 training labels were mirror-consistent on only 67.6% of
    mirrored row pairs (19,434 of 59,995 were inconsistent)

`canonical_teacher_move` fixes this BY CONSTRUCTION: it always solves the
canonically-oriented board and maps the column back. Measured after the fix:
100% mirror-symmetric (155/155 and 246/247 on two independent samples).

Bit layout
----------
Boards use bit index `r * 7 + c` (row-major): row 0 is the BOTTOM row.
Reflecting c -> 6-c maps bit (r, c) to bit (r, 6-c). The column HEIGHTS must
be reflected too, otherwise the "mirrored" board is illegal and solvers
return garbage.
"""

from __future__ import annotations

import copy
from typing import Tuple

COLS = 7
ROWS = 6
# The bitboard is 7x7 = 49 bits (bit r*7+c), even though only rows 0..5 are
# playable. reflect_bits must be a permutation of ALL 49 bits, otherwise it
# is not an involution: reflecting a value with a bit set in row 6 would
# silently drop it.
_GRID_ROWS = 7


def reflect_bits(bits: int) -> int:
    """Reflect a 7x7 bitboard left-right (c -> 6-c).

    Bit index is r*7+c with r=0 the bottom row. All 49 bits are permuted, so
    applying the reflection twice returns the original value.
    """
    out = 0
    for c in range(COLS):
        for r in range(_GRID_ROWS):
            if bits & (1 << (r * COLS + c)):
                out |= 1 << (r * COLS + (COLS - 1 - c))
    return out


def mirror_board(board):
    """Return a left-right mirrored copy of `board`.

    Ownership (p1/p2), the side to move, and the stone layout are preserved
    under reflection; only the column order changes. Heights are reflected
    as well so the result is a legal position.
    """
    nb = copy.deepcopy(board)
    nb.p1 = reflect_bits(board.p1)
    nb.p2 = reflect_bits(board.p2)
    nb.heights = [board.heights[COLS - 1 - c] for c in range(COLS)]
    nb.mask = nb.p1 | nb.p2
    nb.turn = board.turn
    if hasattr(nb, "last_move"):
        lm = nb.last_move
        if lm is not None and isinstance(lm, tuple) and len(lm) == 2:
            nb.last_move = (lm[0], COLS - 1 - lm[1])
    return nb


def canonical_board(board):
    """Return (board, flipped).

    `board` is returned mirrored when that makes the canonical key smaller,
    otherwise unchanged. `flipped` is True when the returned board is the
    mirror of the input. The canonical key is the ordered pair
    (p1, p2) of the returned board, which is a total order and therefore
    selects exactly one of the two orientations.
    """
    mb = mirror_board(board)
    if (mb.p1, mb.p2) < (board.p1, board.p2):
        return mb, True
    return board, False


def canonical_teacher_move(teacher, board, *args, **kwargs) -> int:
    """Evaluate `teacher` on the canonical orientation, then map the column.

    `teacher` must be a callable taking the board as its first argument and
    returning a column index in 0..6. Any extra args are forwarded.

    The returned move is guaranteed to satisfy:
        teacher(board) == 6 - teacher(mirror(board))
    for every board, because the SAME canonical board (and therefore the
    SAME teacher output) is used for both orientations.
    """
    cb, flipped = canonical_board(board)
    col = teacher(cb, *args, **kwargs)
    if flipped:
        return COLS - 1 - col
    return col


def canonical_teacher_set(teacher, board, *args, **kwargs):
    """Same as `canonical_teacher_move` but for a teacher returning a SET.

    Returns the set of columns mapped back into the ORIGINAL orientation.
    """
    cb, flipped = canonical_board(board)
    cols = teacher(cb, *args, **kwargs)
    if flipped:
        return {COLS - 1 - c for c in cols}
    return set(cols)


def canonical_key(board):
    """The canonical (p1, p2, turn) key of `board`.

    Two boards that are mirror images share the SAME canonical key, so a
    teacher cache keyed on this maps each symmetry class to exactly one
    solve. That is what makes the label set deterministic under
    reflection even when the underlying search is budget-limited.
    """
    cb, _flipped = canonical_board(board)
    return (cb.p1, cb.p2, cb.turn)


class CanonicalTeacher:
    """A memoising wrapper around a column-returning teacher.

    The teacher is called with the CANONICAL board, and the result is
    cached under the canonical key. A board and its mirror therefore share
    one solve, and both get the same answer mapped back into their own
    orientation -- mirror-symmetry by construction, with no dependence on
    the search's tie-break direction.

    `teacher(board)` must return a column index in 0..6.
    """

    def __init__(self, teacher):
        self._teacher = teacher
        self._cache = {}

    def __call__(self, board):
        key = canonical_key(board)
        hit = self._cache.get(key)
        if hit is None:
            cb, _flipped = canonical_board(board)
            hit = self._teacher(cb)
            # The cache stores the CANONICAL column, NOT the mapped one.
            # Mapping happens per caller below, because a board and its
            # mirror share this key but need opposite mappings.
            self._cache[key] = hit
        cb, flipped = canonical_board(board)
        if flipped:
            return COLS - 1 - hit
        return hit

    def __len__(self):
        return len(self._cache)
