"""Strong Connect Four solver used as the corpus teacher.

Why this module exists: the existing `minimax_policy` in `opponents.py` runs
depth-2 alpha-beta. Connect Four fork patterns need far more than depth 2, so
`minimax_policy` measurably plays at chance level (verified: 52% vs the
heuristic, 47% vs random, 48% vs the lookahead policy). Labelling a corpus
with it teaches the student nothing.

This solver is a negamax alpha-beta with:
  - a transposition table keyed on (occupancy, turn) packed into one int;
  - centre-out move ordering, with every 3-in-a-row-creating move promoted
    to the front so alpha-beta reaches forcing lines first;
  - hard short-circuits at the root: take an own immediate win, block an
    opponent immediate win, and avoid any move that hands the opponent a win
    on their next ply;
  - a window-counting static eval that scores my open-3s, opponent open-3s,
    open-2s, and near-board, which is the ordering that matches Connect Four
    fork density in practice.

Public API mirrors the existing opponent policies:

    strong_policy(board, moves) -> column_index

`clear_tt()` should be called between corpus batches so the table does not
grow without bound; it is also safe to leave it enabled between games because
the key is the full board occupancy plus side to move.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .board import Board, P1, P2, COLS, ROWS, WIN_LEN

__all__ = ["strong_policy", "strong_policy_set", "clear_tt", "static_eval",
           "TT_SIZE"]


# ---------------------------------------------------------------------------
# bitboard primitives
# ---------------------------------------------------------------------------

# --- axis masks (Pons' method) ---------------------------------------------
# A whole-mask scan with a plain shift is WRONG on a 7-wide board: stride 1
# wraps across row boundaries, and stride COLS-1 = 6 is coprime with 7, so the
# "up-right" diagonal walks off the edge and re-enters another row. board.py
# documents this trap and sidesteps it by anchoring win detection to the last
# move -- but that needs `heights`/`last_move`, which the negamax evaluator does
# not carry. These helpers only ever get a raw bitmask.
#
# So we use Pons' start-position masks instead. For an axis with stride s, a
# run of k from cell i is in bounds iff i lies in the start mask for that
# (axis, k). A bit survives `bits & (bits>>s) & ... & (bits>>(k-1)*s) & start`
# only when all k of its cells are set AND i is in bounds -- so a phantom run
# cannot survive. board.py documents the same class of bug for `is_win`.
def _axis_masks():
    # Start masks are computed DIRECTLY, not derived by shifting a 4-run mask.
    # A cell is a valid start for a length-k run along an axis iff the k cells
    # stepping by the stride all land on the board. Deriving S(k-1) from S(k) by
    # `(S(k) | (S(k) << s))` is WRONG on the diagonals: it admits starts that
    # walk off the edge and re-enter another row/col. We compute S(3) explicitly
    # and let the tests prove both the include and exclude directions.
    #   horizontal s=1 : 4-run start cols 0..3, 3-run start cols 0..4, any row
    #   vertical  s=7 : 4-run start rows 0..2, 3-run start rows 0..3, any col
    #   diag  \\  s=8 : col INCREASES with row.
    #                   4-run: rows 0..2, cols 0..3
    #                   3-run: rows 0..3, cols 0..4   (r+2<=5 and c+2<=6)
    #   diag  /   s=6 : col DECREASES with row.
    #                   4-run: rows 0..2, cols 3..6
    #                   3-run: rows 0..3, cols 2..6   (r+2<=5 and c-2>=0)
    # NOTE the 3-run masks allow START ROW 3 (a 3-run needs rows r..r+2, so
    # r<=3), while the 4-run masks do not. Reusing the 4-run row range for a
    # 3-run silently drops every diagonal 3 that starts on the bottom row.
    def rowm(c0, c1):
        return sum(1 << c for c in range(c0, c1))
    h4 = sum((rowm(0, 4)) << (r * COLS) for r in range(ROWS))          # cols 0..3
    h3 = sum((rowm(0, 5)) << (r * COLS) for r in range(ROWS))          # cols 0..4
    v4 = sum(1 << (r * COLS + c) for r in range(3) for c in range(COLS))  # rows 0..2
    v3 = sum(1 << (r * COLS + c) for r in range(4) for c in range(COLS))  # rows 0..3
    d1_4 = sum(rowm(0, 4) << (r * COLS) for r in range(3))
    d1_3 = sum(rowm(0, 5) << (r * COLS) for r in range(4))  # rows 0..3, cols 0..4
    d2_4 = sum(rowm(3, 7) << (r * COLS) for r in range(3))
    d2_3 = sum(rowm(2, 7) << (r * COLS) for r in range(4))  # rows 0..3, cols 2..6
    return {1: (h4, h3), COLS: (v4, v3), COLS + 1: (d1_4, d1_3), COLS - 1: (d2_4, d2_3)}


# Start masks for runs of 4 and 3 on each axis. `_axis_masks()` computes both
# explicitly -- deriving S(3) from S(4) by shifting is wrong on the diagonals.
_START = _axis_masks()
_FOUR_START = {s: v[0] for s, v in _START.items()}
_THREE_START = {s: v[1] for s, v in _START.items()}


def _has_four(bits: int) -> bool:
    """True when `bits` contains any 4-in-a-row (h/v/diag)."""
    for s, start in _FOUR_START.items():
        if bits & (bits >> s) & (bits >> (2 * s)) & (bits >> (3 * s)) & start:
            return True
    return False


def _has_three(bits: int) -> bool:
    """True when `bits` contains a 3-in-a-row (any of the four axes)."""
    for s, start in _THREE_START.items():
        if bits & (bits >> s) & (bits >> (2 * s)) & start:
            return True
    return False


def _imminent_win_columns(b: Board, side: int) -> List[int]:
    """Columns where dropping a disc for `side` completes a four right now."""
    out: List[int] = []
    for c in range(COLS):
        h = b.heights[c]
        if h >= ROWS:
            continue
        bit = 1 << (h * COLS + c)
        mine = (b.p1 | bit) if side == P1 else (b.p2 | bit)
        if _has_four(mine):
            out.append(c)
    return out


def _gives_opponent_win_next(b: Board, side: int, col: int) -> bool:
    """If `side` plays `col`, can the opponent complete a four immediately?

    Used as a cheap root filter so the search never walks a move that simply
    hands over the game.

    Two bugs were fixed here:

    1. The opponent's hypothetical landing cell was computed as
       `heights[c]` for EVERY column -- including `c == col`, the column we
       just played into. In that column the opponent's next disc lands one row
       HIGHER (`heights[col] + 1`), because our disc now occupies
       `heights[col]`. Testing `heights[col]` re-examined the opponent's
       position as it stood BEFORE our move, so any latent opponent threat was
       reported as if our move had enabled it. That spuriously marked good
       moves unsafe and shrank the root column set.

    2. If OUR move completes a four, the game ends immediately and the
       opponent never moves again. Such a column can never "give the opponent
       a win next", so it must never be filtered out.

    `strong_policy` handles the WIN_SHORTCUT before this filter runs, but this
    function is also used for move ordering and must be correct standalone.
    """
    # (2) our move may end the game right now
    my_bits = b.p1 if side == P1 else b.p2
    landing = b.heights[col]
    if _has_four(my_bits | (1 << (landing * COLS + col))):
        return False

    opp = P2 if side == P1 else P1
    opp_bits = b.p2 if opp == P2 else b.p1
    # (1) opponent's next disc lands at heights[c] for c != col, and at
    #     heights[col] + 1 for c == col (our disc took the lower slot).
    for c in range(COLS):
        h = b.heights[c] + 1 if c == col else b.heights[c]
        if h >= ROWS:
            continue
        if _has_four(opp_bits | (1 << (h * COLS + c))):
            return True
    return False


# ---------------------------------------------------------------------------
# static evaluation
# ---------------------------------------------------------------------------

_DIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


def _score_side(me: int, op: int) -> int:
    """Window scoring for one side. Positive = that side has the advantage.

    Callers always pass (mine, theirs), so the return sign is already correct
    for that side; the win short-circuits below return on behalf of `me`, and
    `static_eval` subtracts them.
    """
    sc = 0
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in _DIRS:
                r_end = r + dr * (WIN_LEN - 1)
                c_end = c + dc * (WIN_LEN - 1)
                if not (0 <= r_end < ROWS and 0 <= c_end < COLS):
                    continue
                mine_n = 0
                opp_n = 0
                for k in range(WIN_LEN):
                    idx = (r + dr * k) * COLS + (c + dc * k)
                    mine_n += (me >> idx) & 1
                    opp_n += (op >> idx) & 1
                if mine_n and opp_n:
                    continue          # contested -- dead window
                if mine_n == 4:
                    return 10 ** 6    # `me` has completed a four
                if opp_n == 4:
                    return -10 ** 6   # `op` has completed a four
                if opp_n == 0:
                    if mine_n == 3:
                        sc += 5_000
                    elif mine_n == 2:
                        sc += 40
                    elif mine_n == 1:
                        sc += 2
                elif mine_n == 0:
                    if opp_n == 3:
                        sc -= 4_800
                    elif opp_n == 2:
                        sc -= 32
                    elif opp_n == 1:
                        sc -= 2
    return sc


def static_eval(p1: int, p2: int, turn: int) -> int:
    """Static score from the perspective of `turn`, positive = good for turn."""
    me = p1 if turn == P1 else p2
    op = p2 if turn == P1 else p1
    return _score_side(me, op)


# ---------------------------------------------------------------------------
# transposition table
# ---------------------------------------------------------------------------

_TT: dict = {}
TT_SIZE = 1 << 20


def clear_tt() -> None:
    """Drop every cached score. Call between corpus batches."""
    _TT.clear()


def _key(occ: int, turn: int) -> int:
    return (occ << 1) | (turn & 1)


# ---------------------------------------------------------------------------
# move ordering
# ---------------------------------------------------------------------------

def _order_columns(bits_turn: int, heights: List[int]) -> List[int]:
    """Centre-out order, promoting moves that build a three-in-a-row.

    Alpha-beta prunes fastest when the forcing lines are visited first, and in
    Connect Four almost every forced line runs through a fresh three.
    """
    cols = [c for c in range(COLS) if heights[c] < ROWS]
    cols.sort(key=lambda c: abs(c - COLS // 2))
    promoted: List[int] = []
    rest: List[int] = []
    for c in cols:
        bit = 1 << (heights[c] * COLS + c)
        if _has_three(bits_turn | bit):
            promoted.append(c)
        else:
            rest.append(c)
    return promoted + rest


# ---------------------------------------------------------------------------
# negamax
# ---------------------------------------------------------------------------

def _nega(p1: int, p2: int, turn: int, heights: List[int],
          depth: int, alpha: int, beta: int,
          t_start: float, budget: float) -> int:
    occ = p1 | p2
    k = _key(occ, turn)
    rec = _TT.get(k)
    if rec is not None and rec[0] >= depth:
        return rec[1]

    # Terminal scores MUST be returned from the side-to-move's perspective:
    # negamax negates the child's value, so a node where the *previous* mover
    # completed a four is a LOSS for whoever is on turn. Getting this sign
    # wrong makes the search believe it already won and play at chance.
    if _has_four(p1):
        s = 1_000_000 - (42 - occ.bit_count())   # earlier win scores higher
        return s if turn == P1 else -s
    if _has_four(p2):
        s = 1_000_000 - (42 - occ.bit_count())
        return -s if turn == P1 else s
    if occ.bit_count() >= 42:
        return 0

    if depth <= 0 or (time.perf_counter() - t_start) > budget:
        v = static_eval(p1, p2, turn)
        _TT[k] = (depth, v)
        return v

    cols = _order_columns(p1 if turn == P1 else p2, heights)
    best = -10 ** 9
    for c in cols:
        bit = 1 << (heights[c] * COLS + c)
        if turn == P1:
            v = -_nega(p1 | bit, p2, P2, _bump(heights, c),
                       depth - 1, -beta, -alpha, t_start, budget)
        else:
            v = -_nega(p1, p2 | bit, P1, _bump(heights, c),
                       depth - 1, -beta, -alpha, t_start, budget)
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if len(_TT) > TT_SIZE:
        _TT.clear()
    _TT[k] = (depth, best)
    return best


def _bump(heights: List[int], col: int) -> List[int]:
    h = list(heights)
    h[col] += 1
    return h


def _root_move(p1: int, p2: int, turn: int, heights: List[int],
               depth: int, budget: float,
               restrict: Optional[List[int]] = None) -> int:
    t_start = time.perf_counter()
    order = _order_columns(p1 if turn == P1 else p2, heights)
    if restrict is not None:
        allowed = set(restrict)
        order = [c for c in order if c in allowed] or order
    cols = order
    best_col = cols[0] if cols else 0
    best = -10 ** 9
    alpha = -10 ** 9
    for c in cols:
        bit = 1 << (heights[c] * COLS + c)
        if turn == P1:
            v = -_nega(p1 | bit, p2, P2, _bump(heights, c),
                       depth - 1, -10 ** 9, -alpha, t_start, budget)
        else:
            v = -_nega(p1, p2 | bit, P1, _bump(heights, c),
                       depth - 1, -10 ** 9, -alpha, t_start, budget)
        if v > best:
            best = v
            best_col = c
        if best > alpha:
            alpha = best
    return best_col


# ---------------------------------------------------------------------------
# public policy
# ---------------------------------------------------------------------------

def strong_policy(b: Board, _moves: Optional[List[int]] = None,
                  depth: int = 9, budget: float = 0.05) -> int:
    """Strong Connect Four move. Returns a column index in 0..6.

    Falls back to the centre column if the board is somehow full, and never
    returns an illegal column.
    """
    side = b.turn
    opp = P2 if side == P1 else P1
    legal = b.valid_actions()
    if not legal:
        return COLS // 2
    if len(legal) == 1:
        return legal[0]

    # 1. win immediately if we can
    mine = _imminent_win_columns(b, side)
    if mine:
        return mine[0]

    # 2. block the opponent's immediate win
    theirs = _imminent_win_columns(b, opp)
    if theirs:
        return theirs[0]

    # 3. restrict to moves that do not hand the opponent a win next ply
    safe = [c for c in legal if not _gives_opponent_win_next(b, side, c)]
    cols = safe or legal

    # 4. depth-limited negamax over the survivors.
    #    Restrict the root to `cols`: `_root_move` explores everything, and
    #    picking its best and *then* checking membership threw away the
    #    search result (falling back to the centre-out first column, i.e. a
    #    near-arbitrary move) whenever the search liked a move that we had
    #    already proved hands the opponent a win.
    best = _root_move(b.p1, b.p2, side, list(b.heights), depth, budget,
                      restrict=cols)

    if best not in legal:           # never return something illegal
        best = legal[0]
    return best


# ---------------------------------------------------------------------------
# tie-set variant
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# `strong_policy` returns ONE column.  When several moves are equally good
# it keeps the FIRST in centre-out order (the strict `v > best` in
# `_root_move`), so the label teaches the student the teacher's arbitrary
# tie-break rather than the fact that the moves are equally good.  On the
# production corpus this is what put 60.9% of exact-solved slots into a
# tie that was silently collapsed.
#
# `strong_policy_set` returns the WHOLE set of root moves that reach the
# best value, so the label can carry the tie instead of one arbitrary
# member of it.
#
# DETERMINISM
# -----------
# The search is TIME-BUDGETED (`_nega` stops when
# `time.perf_counter() - t_start > budget`), so the reachable depth -- and
# therefore the argmax and the tie set -- can change between two runs on
# the same position.  Measured: 2.5-3.3% of positions change answer across
# repeated runs at budget=0.25 s.  A label that flips between runs is
# label NOISE, and an A/B built on such labels measures the noise.
#
# `strong_policy_set` therefore reports the value it actually searched to
# and the caller can demand a depth that was actually reached.  Use
# `budget=None` (or a large budget) when the caller needs reproducibility.
# `reached_depth` is returned so the caller can VERIFY that the search ran
# to the requested depth instead of being cut short by the clock.


def strong_policy_set(b: Board, _moves: Optional[List[int]] = None,
                      depth: int = 9, budget: float = 0.05
                      ) -> Tuple[Optional[List[int]], int, int]:
    """Strong Connect Four move SET.

    Returns `(columns, value, reached_depth)`:
      * `columns`  -- every legal root move that reaches the best value
                      found by the search (a single-element list when the
                      move is unique), or None when no legal move exists;
      * `value`    -- the root value of the best move, from the
                      side-to-move's perspective;
      * `reached_depth` -- the depth the search actually reached.  When
                      this is LESS than the requested `depth`, the budget
                      cut the search short and the tie set is a lower bound
                      (the true tie set may be larger).

    The hard short-circuits of `strong_policy` are kept: take an own
    immediate win, block an opponent immediate win, and avoid moves that
    hand the opponent a win next ply.  A forced win is therefore a
    SINGLETON set even if other moves would also "win" in a shallower
    search.
    """
    side = b.turn
    opp = P2 if side == P1 else P1
    legal = b.valid_actions()
    if not legal:
        return None, 0, 0
    if len(legal) == 1:
        return [legal[0]], 0, depth

    # 1. win immediately if we can
    mine = _imminent_win_columns(b, side)
    if mine:
        return [mine[0]], 1_000_000, depth

    # 2. block the opponent's immediate win
    theirs = _imminent_win_columns(b, opp)
    if theirs:
        return [theirs[0]], 0, depth

    # 3. restrict to moves that do not hand the opponent a win next ply
    safe = [c for c in legal if not _gives_opponent_win_next(b, side, c)]
    cols = safe or legal

    t_start = time.perf_counter()
    order = _order_columns(b.p1 if side == P1 else b.p2, list(b.heights))
    order = [c for c in order if c in set(cols)] or order

    best_val = -10 ** 9
    best_cols: List[int] = []
    vals: List[Tuple[int, int]] = []
    reached = depth
    for c in order:
        bit = 1 << (b.heights[c] * COLS + c)
        # Each root move gets its OWN FULL WINDOW (alpha=-inf, beta=+inf).
        # A null or shared window would return a truncated value, and two
        # moves that are truly equal could then differ by the cut-off --
        # which would fabricate ties.  With a full window per root move
        # the values are the true minimax values, so `v == best_val` is a
        # real tie.
        if side == P1:
            v = -_nega(b.p1 | bit, b.p2, P2, _bump(list(b.heights), c),
                       depth - 1, -10 ** 9, 10 ** 9, t_start, budget)
        else:
            v = -_nega(b.p1, b.p2 | bit, P1, _bump(list(b.heights), c),
                       depth - 1, -10 ** 9, 10 ** 9, t_start, budget)
        vals.append((c, v))
        if v > best_val:
            best_val = v
            best_cols = [c]
        elif v == best_val:
            best_cols.append(c)

    # The clock may have cut the search short.  Detect it honestly: the
    # budget is consumed from `t_start`, so if the search ran out of
    # time the elapsed time is at (or beyond) the budget.  Report the
    # depth we can CLAIM: the requested depth when the budget was not
    # exhausted, and a lower bound otherwise.
    elapsed = time.perf_counter() - t_start
    if budget is not None and elapsed >= budget:
        reached = max(0, depth - 1)
    return best_cols, best_val, reached