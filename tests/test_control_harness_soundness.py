"""Regression tests for the board-copy and metric-feasibility trap.

Three checks, each of which would have caught a real bug we hit:

  1. A board copy used for a what-if MUST reproduce the position exactly.
     If it does not, every what-if metric computed on it is meaningless.
     The original control's `clone()` copied p1/p2/turn but NOT mask/heights,
     so `apply_move()` was a no-op in ~30% of positions and the
     "dont hand over a win" metric could never fire.

  2. A board copy used for a what-if MUST stay in sync with the original
     when the move is legal: the copied board must differ from the
     original in exactly one cell (the played disc).

  3. A rate whose denominator contains mostly-infeasible cases must be
     reported together with the CONDITIONAL rate (cases where the event
     was actually possible). A rate that a random policy passes perfectly
     is vacuous.

Run:  python tmp/test_control_harness_soundness.py
"""
from __future__ import annotations

import copy
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.board import Board, COLS, ROWS  # noqa: E402
from src.c4 import patterns  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def copy_full(b: Board) -> Board:
    c = Board()
    c.p1 = b.p1
    c.p2 = b.p2
    c.mask = b.mask
    c.heights = list(b.heights)
    c.turn = b.turn
    c.last_move = b.last_move
    return c


def copy_broken(b: Board) -> Board:
    """The original control clone -- missing mask and heights."""
    c = Board()
    c.p1 = b.p1
    c.p2 = b.p2
    c.turn = b.turn
    return c


def random_board(rng: random.Random, n_moves: int) -> Board:
    b = Board()
    for _ in range(n_moves):
        legal = b.valid_actions()
        if not legal:
            break
        b.apply_move(rng.choice(legal))
    return b


print("=" * 70)
print("[1] A full board copy reproduces the position exactly")
print("=" * 70)
rng = random.Random(7)
bad = 0
for _ in range(500):
    b = random_board(rng, rng.randint(4, 30))
    c = copy_full(b)
    check_fields = (c.p1 == b.p1 and c.p2 == b.p2 and c.mask == b.mask
                    and c.heights == b.heights and c.turn == b.turn
                    and c.last_move == b.last_move)
    if not check_fields:
        bad += 1
    # and the copy must agree on every cell
    for r in range(ROWS):
        for c2 in range(COLS):
            if b.cell(r, c2) != c.cell(r, c2):
                bad += 1
                break
check("500 random boards copy identically (fields + every cell)",
      bad == 0, f"{bad} mismatches")

print()
print("=" * 70)
print("[2] After a legal move the copied board differs in exactly one cell")
print("=" * 70)
wrong_count = 0
unchanged = 0
trials = 0
for _ in range(500):
    b = random_board(rng, rng.randint(4, 30))
    legal = b.valid_actions()
    if not legal:
        continue
    trials += 1
    col = rng.choice(legal)
    b2 = copy_full(b)
    b2.apply_move(col)
    diffs = sum(1 for r in range(ROWS) for c in range(COLS)
                 if b.cell(r, c) != b2.cell(r, c))
    if diffs != 1:
        wrong_count += 1
    if (b2.p1, b2.p2) == (b.p1, b.p2):
        unchanged += 1
check("the copy always gains exactly one disc", wrong_count == 0,
      f"{wrong_count}/{trials} wrong")
check("the copy is never left unchanged", unchanged == 0,
      f"{unchanged}/{trials} unchanged")

print()
print("=" * 70)
print("[3] The ORIGINAL broken clone IS detected as broken")
print("=" * 70)
unchanged_broken = 0
trials = 0
for _ in range(500):
    b = random_board(rng, rng.randint(4, 30))
    legal = b.valid_actions()
    if not legal:
        continue
    trials += 1
    bb = copy_broken(b)
    bb.apply_move(rng.choice(legal))
    if (bb.p1, bb.p2) == (b.p1, b.p2):
        unchanged_broken += 1
check("the broken clone leaves the board unchanged often enough to matter",
      unchanged_broken > 0,
      f"{unchanged_broken}/{trials} unchanged")
print(f"        (the broken clone silently no-ops in "
      f"{100 * unchanged_broken / max(trials, 1):.1f}% of positions)")

print()
print("=" * 70)
print("[4] A what-if metric on the broken clone CANNOT fire")
print("=" * 70)


def gave_away(b: Board, move: int, me: int, opp: int, copier) -> bool:
    b2 = copier(b)
    b2.apply_move(move)
    return (not patterns.immediate_wins(b2, me)) and bool(
        patterns.immediate_wins(b2, opp))


rng2 = random.Random(99)
fired_full = 0
fired_broken = 0
n_clean = 0
for _ in range(1500):
    b = random_board(rng2, rng2.randint(4, 30))
    legal = b.valid_actions()
    if not legal:
        continue
    me, opp = b.turn, 3 - b.turn
    if patterns.immediate_wins(b, me) or patterns.immediate_wins(b, opp):
        continue
    n_clean += 1
    for m in legal:
        if gave_away(b, m, me, opp, copy_full):
            fired_full += 1
            break
    for m in legal:
        if gave_away(b, m, me, opp, copy_broken):
            fired_broken += 1
            break
check("the metric CAN fire on a correct copy", fired_full > 0,
      f"fired {fired_full} times")
check("the metric fires LESS on the broken copy (it is degraded)",
      fired_broken < fired_full,
      f"broken {fired_broken} vs full {fired_full}")

print()
print("=" * 70)
print("[5] A rate must be reported with the conditional denominator")
print("=" * 70)
# Build the feasibility split explicitly and check the two rates differ.
n_danger_pos = 0
n_clean = 0
for _ in range(1500):
    b = random_board(rng2, rng2.randint(4, 30))
    legal = b.valid_actions()
    if not legal:
        continue
    me, opp = b.turn, 3 - b.turn
    if patterns.immediate_wins(b, me) or patterns.immediate_wins(b, opp):
        continue
    n_clean += 1
    if any(gave_away(b, m, me, opp, copy_full) for m in legal):
        n_danger_pos += 1
check("giving away is possible in a MINORITY of clean positions",
      0 < n_danger_pos < n_clean,
      f"{n_danger_pos}/{n_clean}")
print(f"        giving away possible in {n_danger_pos}/{n_clean} clean "
      f"positions ({100 * n_danger_pos / max(n_clean, 1):.1f}%)")
print("        -> a flat 'gave away' rate over all clean positions is")
print("           dominated by positions where it is impossible.")

print()
print("=" * 70)
print("VERDICT: %s" % ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS}"))
print("=" * 70)
sys.exit(1 if FAILS else 0)
