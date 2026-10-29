"""Connect-Four opponent policies for evaluation.

Used by the training driver to assess whether the model is improving as a
player, not only as a next-byte predictor. Each policy takes a ``Board`` and
the move history, returns a column index.

Three opponents are exposed, in increasing difficulty:

  * ``random_policy``     -- uniform random over legal columns
  * ``heuristic_policy``  -- 1-ply: win > block win > block open-3 > centre
  * ``lookahead_policy``  -- 2-ply: win > block win > create open-3 >
                              block open-3 > centre, ties broken random
  * ``minimax_policy``    -- depth-3 negamax with alpha-beta (strong baseline)

The training eval hooks ``random_policy`` and ``heuristic_policy`` at every
checkpoint; ``minimax_policy`` is the rating-only check at the end.
"""

from __future__ import annotations

import random
from typing import List

from . import COLS, P1, P2
from .board import Board, Outcome
from .corpus import Policy


# --- helpers -----------------------------------------------------------------

def _line_at(b: Board, r: int, c: int, dr: int, dc: int, length: int = 4) -> List[int]:
    """Return the cells of a length-4 window whose anchor is (r,c) and direction
    (dr,dc); cells outside the board show EMPTY."""
    out: List[int] = []
    for k in range(-length + 1, 1):
        rr, cc = r + dr * k, c + dc * k
        if 0 <= rr < 6 and 0 <= cc < COLS:
            out.append(b.cell(rr, cc))
        else:
            out.append(0)
    return out


def _count_runs(player_mask: int, opp_mask: int, dr: int, dc: int) -> int:
    """How many distinct 4-windows through ``player_mask`` are open-3 or stronger?
    Used by the 1-ply heuristic and 2-ply lookahead."""
    n = 0
    for r in range(6):
        for c in range(COLS):
            mine = 0
            opp = 0
            ok = True
            for k in range(4):
                bit = 1 << ((r + dr * k) * COLS + (c + dc * k))
                if player_mask & bit:
                    mine += 1
                elif opp_mask & bit:
                    opp += 1
                else:
                    # an empty cell must actually be reachable: the cell beneath
                    # it must already be filled
                    rr, cc = r + dr * k, c + dc * k
                    below = (rr - 1, cc)
                    if not (0 <= below[0] < 6 and 0 <= below[1] < COLS):
                        ok = False
                        break
                    beneath_bit = 1 << (below[0] * COLS + below[1])
                    if not ((player_mask | opp_mask) & beneath_bit):
                        ok = False
                        break
            if not ok:
                continue
            if mine == 3 and opp == 0:
                n += 1
            elif mine == 2 and opp == 0:
                n += 0  # open-2 not counted (way too many windows)
    return n


def _threat_after(b: Board, col: int, player: int) -> bool:
    """Would `player` win by dropping in `col` next? (Centred win test.)"""
    c = Board()
    c.mask = b.mask
    c.p1 = b.p1
    c.p2 = b.p2
    c.turn = player
    c.heights = list(b.heights)
    c.apply_move(col)
    return c.is_win() == player


def _would_complete_open3(b: Board, col: int, player: int) -> bool:
    """Would `player` create an open-3 by playing `col`? (Their side's POV.)"""
    me = b.p1 if player == P1 else b.p2
    opp = b.p2 if player == P1 else b.p1
    # Simulate
    c = Board()
    c.mask = b.mask
    c.p1 = b.p1
    c.p2 = b.p2
    c.heights = list(b.heights)
    c.turn = player
    c.apply_move(col)
    me2 = c.p1 if player == P1 else c.p2
    opp2 = c.p2 if player == P1 else c.p1
    # check for new open-3 windows through the new piece
    new_idx = c.heights[col] - 1, col
    bit = 1 << (new_idx[0] * COLS + new_idx[1])
    if not (me2 & bit):
        return False
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run = 1
        for s in (1, -1):
            k = 1
            while True:
                rr, cc = new_idx[0] + dr * k * s, new_idx[1] + dc * k * s
                if not (0 <= rr < 6 and 0 <= cc < COLS):
                    break
                bbit = 1 << (rr * COLS + cc)
                if me2 & bbit:
                    run += 1
                    k += 1
                elif opp2 & bbit:
                    break
                else:
                    # empty: gravity-legal?
                    if rr + 1 < 6:
                        bbit2 = 1 << ((rr + 1) * COLS + cc)
                        if not ((me2 | opp2) & bbit2):
                            break
                    k += 1
            if run >= 3:
                return True
    return False


# --- 1: random ---------------------------------------------------------------

def random_policy(b: Board, _moves: List[int]) -> int:
    return random.choice(b.valid_actions())


# --- 0: centre rule (the historical reference bot) -------------------------

def centre_policy(b: Board, _moves: List[int]) -> int:
    """Always play column 3 if it is legal, else the leftmost legal column.

    This is the "centre column" reference bot that ``evaluate_play`` in
    train.py used to hard-code inline. It is the historical yardstick the
    whole project was measured against, so it must be reachable from the
    ladder harness like every other opponent -- otherwise the reference
    bot can only be measured through the trainer's sampling path, which
    grades a SAMPLED model rather than the greedy shipping policy.

    NOTE: the trainer's inline version picks a RANDOM legal column when
    column 3 is full. That randomness makes the reference bot's strength
    depend on the RNG draw, so a 2,000-game ladder against it is not
    reproducible. This version is deterministic; the ladder harness applies
    its own per-ply exploration to the opponent instead.
    """
    legal = b.valid_actions()
    if 3 in legal:
        return 3
    return legal[0]


# --- 2: heuristic (1-ply) ----------------------------------------------------

def heuristic_policy(b: Board, _moves: List[int]) -> int:
    """1-ply: win > block opp's immediate win > centre tiebreak."""
    side = b.turn
    opp = P2 if side == P1 else P1
    legal = b.valid_actions()
    # win in 1
    for c in legal:
        if _threat_after(b, c, side):
            return c
    # block opp in 1
    for c in legal:
        if _threat_after(b, c, opp):
            return c
    # centre bias
    return min(legal, key=lambda c: abs(c - 3))


# --- 3: lookahead (2-ply) ----------------------------------------------------

def lookahead_policy(b: Board, _moves: List[int]) -> int:
    """2-ply: win > block opp's win > create open-3 > block open-3 > centre."""
    rng = random.Random(0xC4C4_BEEF)
    side = b.turn
    opp = P2 if side == P1 else P1
    legal = b.valid_actions()
    if not legal:
        return 0  # never reached

    def score_move(c: int) -> float:
        # win now
        if _threat_after(b, c, side):
            return 1e6
        # block opp's win
        if _threat_after(b, c, opp):
            return 1e5
        # create open-3
        if _would_complete_open3(b, c, side):
            return 1e4
        # block opp's open-3 (play in the column that closes it)
        if _would_complete_open3(b, c, opp):
            return 1e3
        # centre bias
        return -abs(c - 3)

    best = max(legal, key=score_move)
    best_score = score_move(best)
    tied = [c for c in legal if score_move(c) == best_score]
    return rng.choice(tied) if len(tied) > 1 else best


# --- 4: minimax (depth 3) ----------------------------------------------------

def _clone_board(b: Board) -> Board:
    c = Board()
    c.mask = b.mask
    c.p1 = b.p1
    c.p2 = b.p2
    c.turn = b.turn
    c.last_move = b.last_move
    c.heights = list(b.heights)
    return c


def _minimax(b: Board, depth: int, alpha: int, beta: int, side: int,
             rng: random.Random) -> int:
    """Return the column to play.  side=1 -> maximiser for P1; -1 -> P2."""
    legal = b.valid_actions()
    if not legal:
        return -1
    # quick win-check
    opp = P2 if side == P1 else P1
    me_first = -1
    for c in legal:
        if _threat_after(b, c, side):
            me_first = c
            break
    if depth <= 0 or me_first != -1:
        # static score = +(open-3s for me) - (open-3s for opp), win/lose
        # matter far more. We measure by full window scan from current masks.
        me = b.p1 if side == P1 else b.p2
        opp_m = b.p2 if side == P1 else b.p1
        my3 = _count_runs(me, opp_m, 0, 1) + _count_runs(me, opp_m, 1, 0) + \
              _count_runs(me, opp_m, 1, 1) + _count_runs(me, opp_m, 1, -1)
        op3 = _count_runs(opp_m, me, 0, 1) + _count_runs(opp_m, me, 1, 0) + \
              _count_runs(opp_m, me, 1, 1) + _count_runs(opp_m, me, 1, -1)
        score = 100 * my3 - 100 * op3
        if me_first != -1:
            score = 1_000_000 if side == P1 else -1_000_000
        return score

    rng.shuffle(legal)
    if side == P1:
        best = -10**9
        for c in legal:
            child = _clone_board(b)
            child.turn = P1
            child.apply_move(c)
            v = _minimax(child, depth - 1, alpha, beta, P2, rng)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if beta <= alpha:
                break
        return best
    else:
        best = 10**9
        for c in legal:
            child = _clone_board(b)
            child.turn = P2
            child.apply_move(c)
            v = _minimax(child, depth - 1, alpha, beta, P1, rng)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if beta <= alpha:
                break
        return best


def minimax_policy(b: Board, _moves: List[int]) -> int:
    side = b.turn
    opp = P2 if side == P1 else P1
    legal = b.valid_actions()
    # immediate win short-circuit
    for c in legal:
        if _threat_after(b, c, side):
            return c
    # immediate block short-circuit
    for c in legal:
        if _threat_after(b, c, opp):
            return c
    rng = random.Random(0xC4C4_DEAD ^ len(_moves))
    best_col = legal[0]
    if side == P1:
        best_v = -10**9
        for c in legal:
            child = _clone_board(b)
            child.turn = P1
            child.apply_move(c)
            v = _minimax(child, 2, -10**9, 10**9, P2, rng)
            if v > best_v:
                best_v = v
                best_col = c
        return best_col
    else:
        best_v = 10**9
        for c in legal:
            child = _clone_board(b)
            child.turn = P2
            child.apply_move(c)
            v = _minimax(child, 2, -10**9, 10**9, P1, rng)
            if v < best_v:
                best_v = v
                best_col = c
        return best_col


# --- registry ---------------------------------------------------------------

OPPONENTS = {
    "centre": centre_policy,
    "random": random_policy,
    "heuristic": heuristic_policy,
    "lookahead": lookahead_policy,
    "minimax": minimax_policy,
}


def get(name: str) -> Policy:
    return OPPONENTS[name]
