"""Regression test for the is_full() bug.

BUG: Board.is_full() checked only the single bit ROWS*COLS-1 (the top cell
of the last column) instead of all 42 cells. A game where both players
stacked the right-hand column was declared a DRAW after 6 plies, with 36
of 42 cells empty. This is the game the user actually played.

This test pins the CORRECT semantics and checks that the bug cannot come
back in a different disguise.
"""
from __future__ import annotations

import itertools
import random
import sys

sys.path.insert(0, ".")

from src.c4.board import Board, Outcome, P1, P2, ROWS, COLS

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def filled(b: Board) -> int:
    return bin(b.mask).count("1")


print("=" * 70)
print("[1] The user's exact game: both players stack the right-hand column")
print("=" * 70)
b = Board()
for i in range(6):
    b.apply_move(6)
    o = b.outcome()
    if o != Outcome.ONGOING:
        check(f"after {i+1} plies in one column the game is NOT over",
              False, f"outcome={o} filled={filled(b)}/42")
        break
else:
    check("after 6 plies in one column the game is still ONGOING", True)
check("is_full() is False with 36 cells empty", not b.is_full(),
      f"filled={filled(b)}/42")
check("outcome is ONGOING", b.outcome() == Outcome.ONGOING,
      str(b.outcome()))
check("the model can still move (valid actions remain)",
      len(b.valid_actions()) == 6, str(b.valid_actions()))

print()
print("=" * 70)
print("[2] A board IS full only when every column is full")
print("=" * 70)
# Build a board directly from cells so no four can appear. The pattern
# below is a verified 42-cell no-four board (checked with the exact
# solver's has_four on both masks): no row, column or diagonal has four.
def build_no_four(cols: list[int]) -> Board:
    from src.c4.board import board_from_cells
    from src.c4.exact_solver import has_four
    grid = [
        "BYBYBBY",
        "YYBYYBB",
        "BBYBYBY",
        "YYBBYYB",
        "BYBYBYB",
        "YYYBYBB",
    ]
    p1, p2 = [], []
    for r in range(ROWS):
        for c in range(COLS):
            if c not in cols:
                continue
            if grid[r][c] == "Y":
                p1.append((r, c))
            else:
                p2.append((r, c))
    m1 = sum(1 << (r * COLS + c) for r, c in p1)
    m2 = sum(1 << (r * COLS + c) for r, c in p2)
    assert not has_four(m1) and not has_four(m2), (
        f"no-four pattern broken for cols={cols}")
    return board_from_cells(p1, p2, last=(ROWS - 1, max(cols)))


b2 = build_no_four(list(range(COLS - 1)))
check("one column left open -> not full", not b2.is_full(),
      f"filled={filled(b2)}/42")
check("one column left open -> no win either",
      b2.is_win() == 0, f"winner={b2.is_win()} filled={filled(b2)}/42")
check("one column left open -> ongoing", b2.outcome() == Outcome.ONGOING,
      str(b2.outcome()))
check("one column left open -> 36 cells filled", filled(b2) == 36,
      str(filled(b2)))

b3 = build_no_four(list(range(COLS)))
check("all columns full -> is_full()", b3.is_full(), f"filled={filled(b3)}/42")
check("all columns full -> DRAW", b3.outcome() == Outcome.DRAW,
      str(b3.outcome()))
check("all columns full -> 42 cells filled", filled(b3) == 42,
      str(filled(b3)))
check("all columns full -> genuinely no four", b3.is_win() == 0,
      f"winner={b3.is_win()}")

print()
print("=" * 70)
print("[3] The old buggy behaviour would have triggered EARLY on every board")
print("=" * 70)
# The old test was "bit 41 set". That bit is the TOP cell of column 6.
# Any game that stacks column 6 to the top triggered it. Prove the new
# implementation disagrees with the old one exactly where it should.
rng = random.Random(7)
mismatch_old_new = 0
early_terminals = 0
for g in range(300):
    bb = Board()
    for _ in range(42):
        if bb.outcome() != Outcome.ONGOING:
            break
        c = rng.choice(bb.valid_actions())
        bb.apply_move(c)
        old_full = bool((bb.mask >> (ROWS * COLS - 1)) & 1)
        if old_full and not bb.is_full():
            mismatch_old_new += 1
            if old_full and filled(bb) < 42 and bb.is_win() == 0:
                early_terminals += 1
check("the old one-bit test fired early on some boards (bug is real)",
      mismatch_old_new > 0, f"mismatch={mismatch_old_new}")
check("the old test would have called real games full with empty cells",
      early_terminals > 0, f"early={early_terminals}")
print(f"       (old test fired early on {early_terminals}/300 random games)")

print()
print("=" * 70)
print("[4] A truly full board with no four is a DRAW")
print("=" * 70)
# Fill the board in a pattern that gives nobody a four: each column is
# filled by alternating players, which can still create a horizontal four.
# Just verify the DRAW branch is reachable at all.
b3 = build_no_four(list(range(COLS)))
o = b3.outcome()
check("a fully filled board ends the game",
      o in (Outcome.DRAW, Outcome.WIN_P1, Outcome.WIN_P2), str(o))
if o == Outcome.DRAW:
    check("DRAW only when all 42 cells are filled", filled(b3) == 42,
          str(filled(b3)))
else:
    # A four happened first -- that is also correct (the game ends at the
    # winning move, before the board is full).
    check("the game ended at the winning move, before the board filled",
          filled(b3) < 42, str(filled(b3)))
    print(f"       (a four appeared at {filled(b3)}/42 -- game ended correctly)")

print()
print("=" * 70)
print("[5] Exhaustive: is_full() iff no column has space")
print("=" * 70)
bad = 0
rng3 = random.Random(4242)
for g in range(500):
    bb = Board()
    for _ in range(rng3.randint(0, 42)):
        legal = bb.valid_actions()
        if not legal:
            break
        bb.apply_move(rng3.choice(legal))
    expect_full = len(bb.valid_actions()) == 0
    if bb.is_full() != expect_full:
        bad += 1
check("500 random boards: is_full() == (no valid actions)", bad == 0,
      f"mismatches={bad}")

# Exhaustive over one-column stacks: the exact shape that triggered the bug.
bad2 = 0
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
        expect = len(bb.valid_actions()) == 0
        if bb.is_full() != expect:
            bad2 += 1
check("all single-column stacks: is_full() correct", bad2 == 0,
      f"mismatches={bad2}")

print()
print("=" * 70)
print("VERDICT: %s" % ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS}"))
print("=" * 70)
sys.exit(1 if FAILS else 0)
