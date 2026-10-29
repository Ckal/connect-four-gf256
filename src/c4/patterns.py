"""Tactical pattern detection for Connect Four.

Every function here reports what is PHYSICALLY ON THE BOARD. Nothing in
this module consults a model, a teacher, or an opinion -- that is the whole
point. The UI shows these patterns so the human can see WHY a move matters,
and the audit harness uses the exact same code to test whether the MODEL
sees them too. If the two disagree, the model is the problem, not the UI.

Pattern vocabulary (the four categories a Connect Four player cares about):

  COMPLETE FOUR  4 own pieces in a row            -> a win exists
  OPEN THREE     3 own + 1 EMPTY cell in the line  -> a threat; the 4th
                  cell decides the game
  STACKED THREE  3 own + 1 EMPTY cell that is BURIED under other pieces
                  -> a threat that needs support first
  BLOCKED THREE  3 own + 1 OPPONENT piece in the line -> dead, no threat

"3 of mine + 1 of theirs" is a BLOCKED THREE: the opponent's piece sits in
the gap, so the four can never be completed. It is not a threat, and a
model that treats it as one is worse than a model that ignores it.

An OPEN THREE is only an immediate win if its empty cell is the LOWEST OPEN
cell of its column -- otherwise the disc cannot land there yet (that is a
STACKED THREE) and the threat is not playable now.
"""

from __future__ import annotations

from typing import Any

from .board import Board, COLS, ROWS
from . import EMPTY, P1, P2

__all__ = [
    "windows",
    "complete_four",
    "open_threes",
    "blocked_threes",
    "immediate_wins",
    "pattern_report",
]


def windows() -> list[tuple[tuple[int, int], ...]]:
    """Every 4-cell line on a 7x6 board, as (row, col) tuples.

    69 windows: 24 horizontal, 21 vertical, 12 diagonal-up, 12 diagonal-down.

    The two diagonal families are enumerated over DIFFERENT column ranges.
    Up-right diagonals start at c in 0..3; down-right diagonals (which run
    left as they rise) start at c in 3..6. Using the same range for both
    silently generates windows with NEGATIVE columns, and `row * COLS + col`
    then aliases them onto unrelated real cells -- which produced wrong
    "immediate win" columns. Every window is bounds-checked at build time so
    that class of bug cannot come back unnoticed.
    """
    global _WINDOWS
    if _WINDOWS is None:
        w: list[tuple[tuple[int, int], ...]] = []

        def add(cells: tuple[tuple[int, int], ...]) -> None:
            for (r, c) in cells:
                if not (0 <= r < ROWS and 0 <= c < COLS):
                    raise AssertionError(
                        "window cell out of bounds: %r in %r" % ((r, c), cells))
            w.append(cells)

        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                add(((r, c), (r, c + 1), (r, c + 2), (r, c + 3)))
        # vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                add(((r, c), (r + 1, c), (r + 2, c), (r + 3, c)))
        # diagonal rising to the right: starts at c in 0..3
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                add(((r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)))
        # diagonal rising to the LEFT: starts at c in 3..6
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                add(((r, c), (r + 1, c - 1), (r + 2, c - 2), (r + 3, c - 3)))
        _WINDOWS = w
    return _WINDOWS


_WINDOWS: list[tuple[tuple[int, int], ...]] | None = None


def _player_mask(board: Board, player: int) -> int:
    return board.p1 if player == P1 else board.p2


def _classify(board: Board, player: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (complete, open, blocked) pattern lists for `player`."""
    own = _player_mask(board, player)
    opp = board.p2 if player == P1 else board.p1
    complete: list[dict] = []
    open_threes: list[dict] = []
    blocked: list[dict] = []

    for win in windows():
        n_own = n_opp = 0
        gaps: list[tuple[int, int]] = []
        for (r, c) in win:
            bit = 1 << (r * COLS + c)
            if own & bit:
                n_own += 1
            elif opp & bit:
                n_opp += 1
            else:
                gaps.append((r, c))
        if n_own == 4:
            complete.append({"cells": [list(x) for x in win]})
            continue
        if n_own == 3 and n_opp == 0 and len(gaps) == 1:
            r, c = gaps[0]
            playable = (board.heights[c] == r)
            open_threes.append({
                "cells": [list(x) for x in win],
                "gap": [r, c],
                "col": c,
                "playable_now": playable,
            })
            continue
        if n_own == 3 and n_opp == 1 and len(gaps) == 0:
            opp_cells = [[r, c] for (r, c) in win if not (own >> (r * COLS + c) & 1)]
            blocked.append({
                "cells": [list(x) for x in win],
                "opponent_cell": opp_cells[0] if opp_cells else None,
            })
    return complete, open_threes, blocked


def complete_four(board: Board, player: int) -> bool:
    """True if `player` already has four in a row on the board."""
    return bool(_classify(board, player)[0])


def open_threes(board: Board, player: int) -> list[dict]:
    """3 own + 1 empty. Each entry carries 'playable_now'."""
    return _classify(board, player)[1]


def blocked_threes(board: Board, player: int) -> list[dict]:
    """3 own + 1 opponent piece -- a dead line, no threat."""
    return _classify(board, player)[2]


def immediate_wins(board: Board, player: int) -> list[int]:
    """Columns where `player` completes a four RIGHT NOW.

    Derived from the open threes: the gap cell must be the lowest open cell
    of its column, otherwise the disc cannot land there and the four is not
    playable this turn.
    """
    out: list[int] = []
    for t in open_threes(board, player):
        if t["playable_now"] and t["col"] not in out:
            out.append(t["col"])
    return sorted(out)


def _fmt_cells(cells: list[list[int]]) -> str:
    """Human-readable line label, e.g. 'D2 E2 F2' style coordinate text."""
    return " ".join("%s%d" % (chr(ord("A") + c), r + 1) for (r, c) in cells)


def pattern_report(board: Board) -> dict[str, Any]:
    """Full pattern summary for both sides, ready for the UI or an audit."""
    out: dict[str, Any] = {}
    for player, key in ((P1, "p1"), (P2, "p2")):
        complete, open3, blocked = _classify(board, player)
        playable = [t for t in open3 if t["playable_now"]]
        stacked = [t for t in open3 if not t["playable_now"]]
        out[key] = {
            "player": player,
            "complete_four": bool(complete),
            "immediate_wins": immediate_wins(board, player),
            "open_threes": [
                {"cells": t["cells"], "gap": t["gap"], "col": t["col"],
                 "label": _fmt_cells(t["cells"])}
                for t in playable
            ],
            "stacked_threes": [
                {"cells": t["cells"], "gap": t["gap"], "col": t["col"],
                 "label": _fmt_cells(t["cells"])}
                for t in stacked
            ],
            "blocked_threes": [
                {"cells": t["cells"], "label": _fmt_cells(t["cells"]),
                 "opponent_cell": t["opponent_cell"]}
                for t in blocked
            ],
        }
    return out
