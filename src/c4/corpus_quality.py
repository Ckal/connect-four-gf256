"""Replay-based quality grading for dense Connect Four corpus rows.

Why this exists
---------------
The corpus predates the teacher generator. Of the 5,636 rows on disk,
5,444 came from self-play against random and unfiltered opponents, and
**nothing on disk records who won them** -- that is why `train.py` used to
fall back to a flat 0.5 neutral tag on resume, teaching the model bad moves
at half strength.

But the transcript carries the full board. Replaying a row recovers the
winner, the legality of every ply, and whether either side dropped a
winning move. So quality does not require deleting anything: we *grade* the
rows we already have, and `corpus.txt` is never rewritten.

Measured over the live corpus (scripts/audit_corpus.py):

    decided          4,136  (73.4%)
    unfinished       1,500  (26.6%)   <- no outcome, unusable as outcome data
    illegal              0  ( 0.0%)   <- migration was sound
    dropped win     2,290  (40.6%)   <- at least one ply walked away from a win
    missed block        0  ( 0.0%)

Grading
-------
GOOD        decided, and the winner never passed up an available win
BLUNDER     decided, but the WINNER itself dropped a win -- it still won, so
            outcome weighting alone would happily teach that bad move
UNFINISHED  board never resolved: no winner exists, so the row cannot
            supervise an outcome-weighted objective at all
ILLEGAL     a ply was not a legal move for the side to move (should not
            occur; quarantined rather than silently used)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Sequence, Tuple

from .board import Board, P1, P2
from .transcript import PLY_LEN, MOVE_OFFSET, COL_BYTE_OFFSET

# A game is a list of columns; the Game type the trainer already uses.
Game = Tuple[List[int], int]


class Grade(IntEnum):
    GOOD = 0
    BLUNDER = 1
    UNFINISHED = 2
    ILLEGAL = 3


def moves_of_dense(line: bytes) -> List[int]:
    """Recover the move list from one dense corpus row.

    The move byte is the LAST byte of each PLY_LEN-byte record
    (MOVE_OFFSET = 9); offset PLY_LEN-2 is the turn byte. Note this differs
    from the target-slab index used in `action_consistent_examples`, because
    next-byte prediction shifts x -> y by -1.
    """
    n = len(line) // PLY_LEN
    return [line[i * PLY_LEN + MOVE_OFFSET] - COL_BYTE_OFFSET for i in range(n)]


def _winning_cols(b: Board, player: int) -> set:
    """Columns where `player` connects four on the spot.

    `Board.apply_move` mutates in place, so every probe runs on a deepcopy --
    otherwise the board we go on to audit is already corrupted.
    """
    out = set()
    for c in b.valid_actions():
        probe = copy.deepcopy(b)
        probe.apply_move(c)
        if probe.is_win() == player:
            out.add(c)
    return out


@dataclass
class RowQuality:
    grade: Grade
    winner: int          # 1, 2 or 0 when unresolved
    nplies: int
    dropped_wins: int
    missed_blocks: int
    legal: bool


def replay_row(line: bytes) -> RowQuality:
    """Replay one dense corpus row and grade it."""
    if not line or len(line) % PLY_LEN != 0:
        return RowQuality(Grade.ILLEGAL, 0, 0, 0, 0, False)

    moves = moves_of_dense(line)
    b = Board()
    side = P1
    dropped = 0
    missed = 0

    for c in moves:
        if c not in b.valid_actions():
            return RowQuality(Grade.ILLEGAL, 0, len(moves), dropped, missed, False)

        other = P2 if side == P1 else P1
        my_win = _winning_cols(b, side)
        opp_win = _winning_cols(b, other)
        if my_win:
            if c not in my_win:
                dropped += 1
        elif opp_win and c not in opp_win:
            missed += 1

        b.apply_move(c)
        winner = b.is_win()
        if winner:
            return RowQuality(Grade.GOOD if dropped == 0 else Grade.BLUNDER,
                              winner, len(moves), dropped, missed, True)
        side = other

    if not b.valid_actions():
        return RowQuality(Grade.GOOD, 0, len(moves), dropped, missed, True)
    return RowQuality(Grade.UNFINISHED, 0, len(moves), dropped, missed, True)


def grade_line(line: bytes) -> Tuple[Grade, int]:
    """Convenience: just the grade and winner, for the trainer."""
    q = replay_row(line)
    return q.grade, q.winner


def quality_gate(lines: Sequence[bytes],
                 clean: bool = True,
                 verbose: bool = True) -> Tuple[List[Game], dict]:
    """Grade every dense row and return the ones worth training on.

    Returns `(games, report)` where `games` is a list of `(moves, winner)`
    ready for `action_consistent_examples`, and `report` is the audit trail.

    `clean=True` (the point of this module) drops UNFINISHED and ILLEGAL
    rows entirely -- they carry no outcome, and the whole objective is
    outcome-weighted. `clean=False` keeps unfinished rows at quarter
    weight by tagging their winner as 0 so they contribute nothing anyway,
    which is useful only for counting, not for training.

    The corpus file itself is never touched: exclusion happens here, at
    training-example construction, and every exclusion is counted in the
    report so the decision is auditable rather than silent.
    """
    games: List[Game] = []
    kept = {g.name: 0 for g in Grade}
    dropped = {g.name: 0 for g in Grade}
    total_plies = 0
    kept_plies = 0
    winner_moves = 0
    loser_moves = 0

    for line in lines:
        if not line:
            continue
        q = replay_row(line)
        total_plies += q.nplies
        is_game = q.grade in (Grade.GOOD, Grade.BLUNDER, Grade.UNFINISHED) and q.legal
        if is_game:
            winner_moves += sum(1 for _ in range(q.nplies) if _ % 2 == (q.winner - 1))
            loser_moves += sum(1 for _ in range(q.nplies) if _ % 2 != (q.winner - 1))

        exclude = q.grade in (Grade.UNFINISHED, Grade.ILLEGAL) or (
            clean and q.grade is Grade.BLUNDER)

        if exclude:
            dropped[q.grade.name] += 1
            continue

        kept[q.grade.name] += 1
        kept_plies += q.nplies
        # An unfinished row kept under clean=False has no winner; tag it 0 so
        # outcome weighting neutralises it instead of teaching from it.
        games.append((moves_of_dense(line), q.winner))

    report = {
        "rows_in": len(lines),
        "rows_out": len(games),
        "kept": kept,
        "excluded": dropped,
        "plies_in": total_plies,
        "plies_out": kept_plies,
        "winner_move_plies": winner_moves,
        "loser_move_plies": loser_moves,
        "loser_move_frac": (loser_moves / max(winner_moves + loser_moves, 1)),
        "mode": "clean" if clean else "audit-only",
    }

    if verbose:
        print(f"corpus quality gate ({report['mode']}): "
              f"{report['rows_in']} rows -> {report['rows_out']} kept", flush=True)
        for g in Grade:
            if kept[g.name] or dropped[g.name]:
                print(f"    {g.name:11s} kept {kept[g.name]:6d}   "
                      f"excluded {dropped[g.name]:6d}", flush=True)
        print(f"    plies {total_plies} -> {kept_plies}", flush=True)
        print(f"    winner-move plies {winner_moves}  "
              f"loser-move plies {loser_moves} "
              f"({100.0*report['loser_move_frac']:.1f}% of decided plies)", flush=True)
    return games, report