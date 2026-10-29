"""Connect-Four board representation and rules.

Storing the board as a 64-bit mask per player is fast (bitwise AND for win
detection) and is deliberately decoupled from the byte encoding so the model
learns the encoding, not the kernel.

A column's lowest open row is the only thing that matters; we keep a 6-bit
``heights[col]`` so apply_move is O(1) and winner detection is contiguous-bit
arithmetic.
"""

from __future__ import annotations

from . import ROWS, COLS, WIN_LEN, EMPTY, P1, P2


class _BoardBuilder:
    """Internal helper so transcript.decode can build a board from raw bits."""
    __slots__ = ("mask", "p1", "p2", "turn", "last_move", "heights")

    def __init__(self) -> None:
        self.mask = 0
        self.p1 = 0
        self.p2 = 0
        self.turn = P1
        self.last_move = None
        self.heights = [0] * COLS


def _set_cell(builder: _BoardBuilder, r: int, c: int, player: int) -> None:
    """Plausible-replay from a grid: fills cells in scan order. Doesn't need to
    match the order they were actually placed; the masks and heights are what
    matter for forward play."""
    idx = r * COLS + c
    bit = 1 << idx
    if player == P1:
        builder.p1 |= bit
    else:
        builder.p2 |= bit
    builder.mask |= bit
    if r + 1 > builder.heights[c]:
        builder.heights[c] = r + 1


class Outcome:
    ONGOING = 0
    WIN_P1 = 1
    WIN_P2 = 2
    DRAW = 3


class Board:
    """Two 64-bit masks; (row, col) -> bit (row*7+col)."""

    def __init__(self) -> None:
        # 64 bits per player; only low ROWS*COLS = 42 are meaningful.
        self.mask: int = 0  # bits set for all occupied squares
        self.p1: int = 0
        self.p2: int = 0
        self.turn: int = P1
        self.last_move: int | None = None
        # heights[col] = bit index of the lowest open row in column col, or ROWS if full
        self.heights: list[int] = [0] * COLS

    # -------- queries --------

    def is_full(self) -> bool:
        """True when every one of the ROWS*COLS cells is occupied.

        BUG (found via a live UI game): this used to check ONLY the single
        bit ROWS*COLS-1 -- the top cell of the last column. A game in which
        both players kept stacking column 6 (internal c6, the right-hand
        edge) filled that one cell after 6 plies and the board was declared
        a DRAW with 36 of 42 cells still empty. Every consumer of
        `is_full()` (duel.py, play.py, corpus generation, the exact
        solver's terminal test) inherited that early draw.

        The correct test is "no legal action remains": every column is
        full. `valid_actions()` already encodes exactly that, so use it
        rather than re-deriving a mask comparison.
        """
        return not self.valid_actions()

    def cell(self, row: int, col: int) -> int:
        idx = row * COLS + col
        bit = 1 << idx
        if not (self.mask & bit):
            return EMPTY
        return P1 if (self.p1 & bit) else P2

    def valid_actions(self) -> list[int]:
        """Columns where the top row is still empty."""
        return [c for c in range(COLS) if self.cell(ROWS - 1, c) == EMPTY]

    # -------- mutation --------

    def apply_move(self, col: int) -> "Board":
        """Returns the SAME board object, mutated. Caller decides; this keeps the
        play loop fast (no allocations in the inner loop)."""
        assert 0 <= col < COLS and self.cell(ROWS - 1, col) == EMPTY, (
            f"illegal col={col} turn={self.turn}"
        )
        row = self.heights[col]
        idx = row * COLS + col
        bit = 1 << idx
        if self.turn == P1:
            self.p1 |= bit
        else:
            self.p2 |= bit
        self.mask |= bit
        self.heights[col] = row + 1
        self.last_move = col
        self.turn = P2 if self.turn == P1 else P1
        return self

    # -------- terminal --------

    def is_win(self) -> int:
        """0 = no winner, else P1/P2.

        Win detection runs only along lines THROUGH THE LAST MOVE. A new line of
        four can only appear there, since every other cell is unchanged. This
        also removes the classic bitboard bug: a stride-1 shift wraps across a
        row boundary, so whole-board masks would report phantom horizontal wins.
        With last_move anchoring, the at-most-3 cells on each side of the new
        piece are read individually and never wrap.
        """
        if self.last_move is None:
            return 0
        player = P2 if self.turn == P1 else P1  # the side that just moved
        me = self.p1 if player == P1 else self.p2
        row = self.heights[self.last_move] - 1  # row the piece landed on
        col = self.last_move

        def mine(r: int, c: int) -> bool:
            if not (0 <= r < ROWS and 0 <= c < COLS):
                return False
            return bool(me >> (r * COLS + c) & 1)

        # (drow, dcol) for horizontal, vertical, and both diagonals
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            run = 1
            for s in (1, -1):
                k = 1
                while mine(row + dr * k * s, col + dc * k * s):
                    run += 1
                    k += 1
            if run >= WIN_LEN:
                return player
        return 0

    def outcome(self) -> int:
        w = self.is_win()
        if w:
            return Outcome.WIN_P1 if w == P1 else Outcome.WIN_P2
        if self.is_full():
            return Outcome.DRAW
        return Outcome.ONGOING


def is_terminal(board: "Board") -> bool:
    """True when the game is over: somebody won, or no legal action remains.

    THE canonical terminal predicate for the whole project. Every play
    loop, duel harness, corpus generator and trainer should call this (or
    `board.outcome()`), never a hand-rolled fullness test.

    Why this exists: `Board.is_full()` once checked a SINGLE mask bit
    instead of the whole board, so a game that stacked one edge column was
    declared a DRAW with 36 of 42 cells empty. The bug was invisible in
    the code and corrupted the corpus, the duels and the live UI.
    Centralising the predicate here makes the invariant explicit and gives
    new consumers one obvious correct call.
    `board.outcome() != Outcome.ONGOING` is equivalent; this spelling is
    preferred at call sites that only need a boolean.
    """
    return board.is_win() != 0 or not board.valid_actions()


def assert_terminal_consistent(board: "Board") -> None:
    """Assert the board's own terminal flags agree with the canonical rule.

    A cheap self-check for test harnesses and corpus builders: it catches
    the class of bug where `is_full()` and `outcome()` disagree, or where
    a full board still claims to be playable.
    """
    full = not board.valid_actions()
    assert board.is_full() == full, (
        f"is_full()={board.is_full()} but legal actions remain: "
        f"{board.valid_actions()}")
    if full:
        assert board.is_win() == 0 or board.outcome() in (
            Outcome.WIN_P1, Outcome.WIN_P2), (
            "full board with a win should report the win, not a draw")
    else:
        assert board.outcome() != Outcome.DRAW, (
            "outcome says DRAW but the board still has legal actions")


def play_sequence(moves: list[int]) -> tuple[Board, int]:
    """Re-play a list of column moves from empty. Returns (board, outcome)."""
    b = Board()
    for m in moves:
        b.apply_move(m)
        o = b.outcome()
        if o != Outcome.ONGOING:
            return b, o
    return b, Outcome.ONGOING if not b.is_full() else Outcome.DRAW


def board_from_cells(p1_cells: list[tuple[int, int]],
                     p2_cells: list[tuple[int, int]],
                     last: tuple[int, int] | None = None,
                     ) -> Board:
    """Build a position directly from (row, col) lists.

    Only valid when every cell is gravity-legal (all cells below it are filled
    by someone), which holds for anything produced by real play. ``last`` is the
    (row, col) of the most recent piece; win detection needs it, so if omitted
    the highest cell of ``p1_cells``+``p2_cells`` is used.

    Needed because a 4-cell diagonal cannot be reached by alternating play from
    an empty board: it requires 6+ tokens of scaffolding, and P2 gets only 4
    replies. Real games do reach diagonals, just not this early.
    """
    b = Board()
    tmp = _BoardBuilder()
    for cells, player in ((p1_cells, P1), (p2_cells, P2)):
        for r, c in cells:
            if not (0 <= r < ROWS and 0 <= c < COLS):
                raise ValueError(f"cell ({r},{c}) out of range")
            _set_cell(tmp, r, c, player)
    b.mask = tmp.mask
    b.p1 = tmp.p1
    b.p2 = tmp.p2
    b.heights = list(tmp.heights)
    if last is not None:
        r, c = last
        b.last_move = c
        b.heights[c] = r + 1
    else:
        filled = p1_cells + p2_cells
        if filled:
            _, c = max(filled, key=lambda rc: rc[0])
            b.last_move = c
    b.turn = P1
    return b


def set_last_and_turn(b: Board, last_rc: tuple[int, int], mover: int) -> Board:
    """Attach the last-move/turn metadata win detection needs (test helper)."""
    r, c = last_rc
    b.last_move = c
    b.heights[c] = max(b.heights[c], r + 1)
    b.turn = P2 if mover == P1 else P1
    return b
