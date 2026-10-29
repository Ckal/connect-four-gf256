"""Cross-check src/c4/patterns.py against the canonical tg_block_test win_cols.

patterns.immediate_wins is a NEW implementation (window enumeration + gap
analysis). tg_block_test.win_cols is the EXISTING, already-trusted
implementation (apply the move, ask is_win). They must agree on every
reachable position, or the new module is wrong.
"""
import sys, random, itertools, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.c4.board import Board, Outcome, P1, P2
from src.c4.patterns import (immediate_wins, open_threes, blocked_threes,
                             complete_four, windows, pattern_report)
from tg_block_test import win_cols

print("window count:", len(windows()), "(expect 69)")
assert len(windows()) == 69, "window enumeration is wrong"

random.seed(20261027)
mismatch = 0
checked = 0
for trial in range(4000):
    b = Board()
    nmoves = random.randint(0, 30)
    for _ in range(nmoves):
        legal = b.valid_actions()
        if not legal:
            break
        b.apply_move(random.choice(legal))
        if b.outcome() != Outcome.ONGOING:
            break
    for player in (P1, P2):
        a = immediate_wins(b, player)
        e = win_cols(b, player)
        checked += 1
        if a != e:
            mismatch += 1
            if mismatch <= 5:
                print("MISMATCH player=%d patterns=%s canonical=%s moves=%s" % (
                    player, a, e, getattr(b, "_moves", None)))
print("positions checked: %d" % checked)
print("mismatches vs canonical tg_block_test.win_cols: %d" % mismatch)

# --- the user's named patterns -----------------------------------------
# "3*orange 1*blue is good"  -> 3 of mine + 1 of theirs = BLOCKED (dead)
# "4*blue is perfect"          -> 4 in a row = COMPLETE FOUR (a win)
print()
print("=== the user's vocabulary, checked against the board ===")
b = Board()
for c in (0, 1, 2):          # P1 gets A1,B1,C1
    b.apply_move(c)
for c in (3, 4, 5):          # P2 gets D1,E1,F1
    b.apply_move(c)
b.apply_move(6)              # P1 D1
# P1 now has A1,B1,C1,D1 = FOUR horizontally
rep = pattern_report(b)
print("P1 four complete?", rep["p1"]["complete_four"])
print("P1 immediate wins:", rep["p1"]["immediate_wins"])
print("P2 blocked threes:", len(rep["p2"]["blocked_threes"]))

# 3 own + 1 opponent in the SAME window
b2 = Board()
b2.apply_move(0)   # P1 A1
b2.apply_move(3)   # P2 D1
b2.apply_move(1)   # P1 B1
b2.apply_move(4)   # P2 E1
b2.apply_move(2)   # P1 C1  -> P1 has A1,B1,C1 ; P2 has D1,E1
b2.apply_move(5)   # P2 F1  -> P2 D1,E1,F1
rep2 = pattern_report(b2)
print()
print("P2 has D1,E1,F1 (3 in a row, gap G1):")
print("  P2 open threes   :", [t["label"] for t in rep2["p2"]["open_threes"]])
print("  P2 immediate wins:", rep2["p2"]["immediate_wins"])
print("  (G1 must be playable -> heights[6]==0 -> True)")

# Now make it 3 orange + 1 blue in one window: P1 A1,B1,C1 + P2 D1 blocks
b3 = Board()
b3.apply_move(0)   # P1 A1
b3.apply_move(6)   # P2 G1
b3.apply_move(1)   # P1 B1
b3.apply_move(5)   # P2 F1
b3.apply_move(2)   # P1 C1
b3.apply_move(3)   # P2 D1  -> window A1..D1 has P1 x3 + P2 x1
rep3 = pattern_report(b3)
print()
print("Window A1-D1 contains P1 x3 (A1,B1,C1) and P2 x1 (D1):")
print("  P1 blocked threes:", [t["label"] for t in rep3["p1"]["blocked_threes"]])
print("  P1 open threes   :", [t["label"] for t in rep3["p1"]["open_threes"]])
print("  P1 immediate wins:", rep3["p1"]["immediate_wins"])
print()
print("VERDICT:", "PASS" if mismatch == 0 else "FAIL")
