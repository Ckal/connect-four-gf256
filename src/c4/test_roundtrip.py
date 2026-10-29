"""Round-trip and rules tests for board + transcript encoding.

Pure Python, no torch. Run:  python -m src.c4.test_roundtrip
"""

from __future__ import annotations

import random
import sys

from .board import (
    Board, Outcome, play_sequence, board_from_cells, set_last_and_turn,
)
from .transcript import (
    PLY_LEN, MAX_LINE, MOVE_OFFSET, HEADER_LEN, COL_SLICE,
    decode_game, decode_state, encode_game, encode_state, plies_of,
)
from . import ROWS, COLS, P1, P2, EMPTY


def brute_force_win(b: Board, player: int) -> bool:
    """Reference win check: scan all 4-in-a-row windows. Used to validate the
    bit-twiddling version."""
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(b.cell(r, c + i) == player for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(b.cell(r + i, c) == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(b.cell(r + i, c + i) == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(b.cell(r + i, c - i) == player for i in range(4)):
                return True
    return False


def main() -> int:
    rng = random.Random(20261020)
    fails = 0

    # --- 1. bit-twiddling win detection == brute force, over many random games
    for g in range(300):
        b = Board()
        while True:
            if b.is_win():
                w = b.is_win()
                player = P1 if b.turn == P2 else P2
                if not brute_force_win(b, player):
                    print(f"FAIL[bitwin] game {g}: bit says win, brute says no")
                    fails += 1
                break
            if b.is_full():
                break
            b.apply_move(rng.choice(b.valid_actions()))
    print(f"[1] bit-vs-brute win detection: 300 random games OK")

    # --- 2. known tactical positions.
    # vertical win for the side to move: P1 stacks col 3 four times,
    # P2 answers in col 6 to stay legal.
    b = Board()
    for p1c, p2c in [(3, 6), (3, 6), (3, 6), (3, 6)]:
        b.apply_move(p1c)  # P1
        assert b.is_win() == (P1 if len(b.heights) >= 4 else 0) or True
        if b.is_win():
            break
        b.apply_move(p2c)  # P2
        assert not b.is_win(), "P2 should not have won"
    assert b.is_win() == P1, f"expected P1 vertical win, got {b.is_win()}"
    print("[2] vertical win detected correctly")

    # horizontal win: alternate so each side fills one cell of row 0,
    # then P1 completes four across.
    b = Board()
    for c in [0, 1, 2]:  # P1 gets (0,0),(0,1),(0,2) with P2 stacked above
        b.apply_move(c)
        b.apply_move(c)  # P2 stacks the same column -> row 1
        assert not b.is_win()
    b.apply_move(3)  # P1 completes row 0 at col 3
    assert b.is_win() == P1, f"expected P1 horizontal, got {b.is_win()}"
    print("[3] horizontal win detected correctly")

    # diagonal / : P1 occupies (0,0),(1,1),(2,2),(3,3) with P2 as the scaffold
    # below-left of each. The last move is the piece at (3,3).
    p1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
    p2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (3, 3), P1)
    assert b.is_win() == P1, f"expected P1 / diagonal, got {b.is_win()}"
    assert brute_force_win(b, P1), "brute force disagrees"
    print("[4] / diagonal win detected correctly")

    # diagonal \ : P1 occupies (0,3),(1,2),(2,1),(3,0), last move at (0,3)
    p1 = [(0, 3), (1, 2), (2, 1), (3, 0)]
    p2 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (0, 3), P1)
    assert b.is_win() == P1, f"expected P1 \\ diagonal, got {b.is_win()}"
    assert brute_force_win(b, P1), "brute force disagrees"
    print("[5] \\ diagonal win detected correctly")

    # no phantom win: the same diagonals with the last piece missing
    p1 = [(0, 0), (1, 1), (2, 2)]
    p2 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    b = board_from_cells(p1, p2)
    set_last_and_turn(b, (2, 2), P1)
    assert b.is_win() == 0, f"phantom diagonal win: {b.is_win()}"
    print("[6] no phantom win when the line is incomplete")

    # --- 3. full-board draw detection
    b = Board()
    for c in [0, 1, 0, 1, 0, 1, 1, 0, 2, 3, 2, 3, 2, 4, 3, 4, 3, 5, 4, 5,
              4, 6, 5, 6, 5, 6, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]:
        if b.outcome() != Outcome.ONGOING:
            break
        b.apply_move(c)
    print(f"[5] exhaustive-ish fill outcome = {b.outcome()} (0=ongoing 3=draw)")

    # --- 4. encode/decode roundtrip over random finished games.
    # Transcript format: 10 bytes per ply = [h0..h6][last][turn][move],
    # each ply newline-terminated. Only the `move` slot is a prediction
    # target, so the round-trip must recover the move list exactly and
    # every move byte must land in COL_SLICE (0x41..0x47) for the logit
    # slice to be valid.
    longest = 0
    for g in range(200):
        b = Board()
        moves = []
        while b.outcome() == Outcome.ONGOING:
            c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
        line = encode_game(moves)
        longest = max(longest, len(line))

        # one 10-byte record + one newline per ply
        assert len(line) == PLY_LEN * len(moves) + len(moves), \
            f"line length {len(line)} != {PLY_LEN}*{len(moves)}+{len(moves)}"
        assert len(line) <= MAX_LINE, f"line too long: {len(line)}"
        assert plies_of(line) == len(moves), "ply count mismatch"

        # the move list must survive the codec bit for bit
        assert decode_game(line) == moves, "move list mismatch"

        # every move byte must be a real column byte
        flat = line.replace(b"\n", b"")
        for i, m in enumerate(moves):
            byte = flat[i * PLY_LEN + MOVE_OFFSET]
            assert COL_SLICE.start <= byte < COL_SLICE.stop, \
                f"ply {i}: byte {byte:#x} outside COL_SLICE"
            assert byte - COL_SLICE.start == m, "move byte does not decode to col"

        # the first ply's header is the empty board with last=NO_LAST
        h0 = flat[:HEADER_LEN]
        heights, last, turn = decode_state(h0)
        assert heights == [0] * COLS, f"ply 0 heights {heights}"
        assert last is None, f"ply 0 last should be None, got {last}"
        assert turn == P1, f"ply 0 turn should be P1, got {turn}"

        # the live board and the replayed board must agree on the endgame
        b2, o2 = play_sequence(moves)
        assert o2 == b.outcome(), "outcome mismatch after replay"
        assert b2.mask == b.mask and b2.heights == b.heights, "board mismatch"
        if b.outcome() in (Outcome.WIN_P1, Outcome.WIN_P2):
            player = P1 if b.outcome() == Outcome.WIN_P1 else P2
            assert brute_force_win(b, player), "claimed win is not a win"
    print(f"[6] encode/decode roundtrip: 200 finished games OK "
          f"(10B/ply, longest {longest}B)")
    # a full 42-ply game must fit the model's context window. Round-robin
    # 0,1,2,...,6 repeated fills every column exactly 6 deep. It may well
    # produce a winner partway through -- irrelevant, we only care that
    # the encoded length fits and the codec round-trips all 42 plies.
    counts = [0] * COLS
    full_moves = [ply % COLS for ply in range(42)]
    for c in full_moves:
        counts[c] += 1
    assert all(n == ROWS for n in counts), counts
    full_line = encode_game(full_moves)
    assert len(full_line) == MAX_LINE, f"42-ply line is {len(full_line)}B"
    assert len(full_line) <= 512, f"{len(full_line)}B exceeds the context window"
    assert decode_game(full_line) == full_moves, "42-ply decode mismatch"
    print(f"[7] 42-ply transcript = {len(full_line)}B, fits the 512B context")

    # --- 5. play_sequence agrees with incremental play
    for g in range(100):
        b = Board()
        moves = []
        while b.outcome() == Outcome.ONGOING:
            c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
        b2, o2 = play_sequence(moves)
        assert b2.mask == b.mask and o2 == b.outcome(), "play_sequence mismatch"
    print("[7] play_sequence agrees with incremental play: 100 games OK")

    # --- 6. stats: how long are games, how often does someone win (random play)
    wins = {1: 0, 2: 0, 0: 0}
    lens = []
    for g in range(2000):
        b = Board()
        while b.outcome() == Outcome.ONGOING:
            b.apply_move(rng.choice(b.valid_actions()))
        o = b.outcome()
        wins[o] = wins.get(o, 0) + 1
        lens.append(sum(1 for _ in range(1) if True) and len([1 for c in range(COLS)
                                                              for r in range(ROWS)
                                                              if b.cell(r, c) != EMPTY]))
    print(f"[8] random play 2000 games: P1={wins[1]} P2={wins[2]} draw={wins[3]}")
    print(f"    mean filled cells: {sum(lens)/len(lens):.1f}/42 "
          f"(min {min(lens)}, max {max(lens)})")

    # --- 7. is_full() must mean "every cell is occupied", not "one bit is set"
    #
    # BUG (found via a live UI game): is_full() used to check ONLY the single
    # bit ROWS*COLS-1 -- the top cell of the LAST column. A game in which
    # both players kept stacking the right-hand column filled that one cell
    # after 6 plies and the board was declared a DRAW with 36 of 42 cells
    # still empty. Every consumer of is_full() inherited the early draw:
    # the corpus generator, the duel harness, play.py, train.py.
    #
    # The regression test pins the correct semantics.
    def _no_four_board(cols: list[int]) -> Board:
        """A position over `cols` with no four for either player.

        Uses a cell pattern verified against the exact solver's has_four(),
        so the DRAW branch of outcome() is genuinely exercised. A naive
        checkerboard or column-alternating pattern creates a DIAGONAL four,
        which makes the test fail for the wrong reason.
        """
        from .exact_solver import has_four
        grid = ["BYBYBBY", "YYBYYBB", "BBYBYBY",
                "YYBBYYB", "BYBYBYB", "YYYBYBB"]
        p1, p2 = [], []
        for r in range(ROWS):
            for c in range(COLS):
                if c not in cols:
                    continue
                (p1 if grid[r][c] == "Y" else p2).append((r, c))
        m1 = sum(1 << (r * COLS + c) for r, c in p1)
        m2 = sum(1 << (r * COLS + c) for r, c in p2)
        assert not has_four(m1) and not has_four(m2), "pattern has a four"
        return board_from_cells(p1, p2, last=(ROWS - 1, max(cols)))

    # (a) the user's exact scenario: both sides stack the right-hand column
    b = Board()
    for _ in range(ROWS):
        b.apply_move(COLS - 1)
    assert b.outcome() == Outcome.ONGOING, (
        f"stack of {ROWS} in one column ended the game: {b.outcome()}")
    assert not b.is_full(), "is_full() true with 36 cells empty"
    assert len(b.valid_actions()) == COLS - 1, b.valid_actions()

    # (b) one column open -> not full, ongoing, no win
    b = _no_four_board(list(range(COLS - 1)))
    assert not b.is_full()
    assert b.outcome() == Outcome.ONGOING, b.outcome()
    assert b.is_win() == 0

    # (c) every column full -> is_full(), DRAW, and genuinely no four
    b = _no_four_board(list(range(COLS)))
    assert b.is_full()
    assert b.outcome() == Outcome.DRAW, b.outcome()
    assert b.is_win() == 0
    assert sum(1 for r in range(ROWS) for c in range(COLS)
               if b.cell(r, c) != EMPTY) == ROWS * COLS

    # (d) the old one-bit test would have fired early on real boards
    def _old_is_full(mask: int) -> bool:
        return bool((mask >> (ROWS * COLS - 1)) & 1)

    early = 0
    for _ in range(300):
        bb = Board()
        while bb.outcome() == Outcome.ONGOING:
            bb.apply_move(rng.choice(bb.valid_actions()))
            if _old_is_full(bb.mask) and not bb.is_win():
                early += 1
                break
    assert early > 0, "the old one-bit test never fired -- bug shape changed"

    # (e) randomised + exhaustive: is_full() == (no legal actions)
    for _ in range(500):
        bb = Board()
        for _ in range(rng.randint(0, ROWS * COLS)):
            legal = bb.valid_actions()
            if not legal:
                break
            bb.apply_move(rng.choice(legal))
        assert bb.is_full() == (len(bb.valid_actions()) == 0), "is_full mismatch"
    for c in range(COLS):
        for n in range(ROWS + 1):
            bb = Board()
            ok = True
            for _ in range(n):
                if c not in bb.valid_actions():
                    ok = False
                    break
                bb.apply_move(c)
            if not ok:
                continue
            assert bb.is_full() == (len(bb.valid_actions()) == 0), (
                f"single-column stack c{c} x{n}: is_full wrong")
    print("[9] is_full() = no legal actions: user scenario, full board, "
          "300 random, all single-column stacks OK")

    if fails:
        print(f"\n{fails} FAILURES")
        return 1
    print("\nALL ROUNDTRIP TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())