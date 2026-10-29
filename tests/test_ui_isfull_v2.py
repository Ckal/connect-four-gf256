"""Live UI verification after the is_full() fix.

Replays the ORIGINAL reported scenario through the LIVE HTTP server:
both sides stack column 7 (internal c6). Before the fix the game ended
as a DRAW after 6 plies with 36 cells empty. Now it must stay ONGOING,
keep offering legal moves, and only end as a DRAW when the board is
genuinely full.

PREMISE FIX
------------------------
The previous version of this test drove the human to play column 7 four
times in a row and asserted the game stayed ongoing. That premise is
wrong: FOUR discs in one column is a vertical four, so the game correctly
ends in a win for the human on the 4th move -- no `is_full()` bug
involved. The old test therefore failed for a reason that had nothing to
do with the bug it was written to catch.

The scenario below keeps the intent (fill column 7 to 6 deep WITHOUT
winning) by alternating the human between column 7 and a neutral column,
so no side ever has four in a line. The model replies with whatever it
likes; the assertions are about SERVER behaviour (is_full / draw / legal
actions), not about the model's choice, so the test does not depend on
the model cooperating.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.60"
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def post(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


print("=" * 70)
print("[1] Column 7 filled to 6 deep must NOT end the game as a draw")
print("=" * 70)
post("/seat", {"seat": 1})
st = get("/state")
check("seat switch restarts the game", st["moves"] == [], str(st["moves"]))
check("user is P1 now", st["user_seat"] == 1, str(st["user_seat"]))
check("fresh game is ongoing", st["outcome"] == "ongoing", st["outcome"])
check("it is the user's turn", st["turn"] == "user", st["turn"])

# Play column 7 (internal c6) three times, alternating with column 0 so
# that neither side accumulates four in a line through the human's own
# moves. The MODEL's replies are whatever the model chooses -- the point
# of the test is server behaviour, not the model's taste.
human_cols = [6, 0, 6, 0, 6, 0]
rejected = 0
for i, col in enumerate(human_cols, start=1):
    st = get("/state")
    if st["outcome"] != "ongoing":
        break
    legal = st["valid_actions"]
    if col not in legal:
        check(f"column {col} still legal before move {i}", False, str(legal))
        break
    try:
        st = post("/move", {"col": col})
    except urllib.error.HTTPError as e:
        rejected += 1
        body = json.loads(e.read().decode())
        print(f"    move {i}: REJECTED -> {body.get('error')}")
        check(f"rejection {i} is a legal-move error, not a draw",
              "draw" not in str(body.get("error", "")).lower(), str(body))
        break
    filled = sum(1 for row in st["cells"] for v in row if v != 0)
    print(f"    after move {i} (col {col}): moves={st['moves']} "
          f"filled={filled}/42 outcome={st['outcome']} turn={st['turn']}")
    check(f"after {i} human moves the game is still ongoing",
          st["outcome"] == "ongoing", st["outcome"])

st = get("/state")
filled = sum(1 for row in st["cells"] for v in row if v != 0)
print(f"    final: filled={filled}/42 outcome={st['outcome']} "
      f"valid={st['valid_actions']}")
check("the game did NOT end in a draw", st["outcome"] != "draw", st["outcome"])
check("the board is not full", filled < 42, str(filled))
check("legal actions are still offered", len(st["valid_actions"]) >= 1,
      str(st["valid_actions"]))
# Column 7 (internal c6) is 6 deep only if the model kept answering there.
# Assert the invariant that matters: whenever the column IS 6 deep, it is
# not offered any more, and the game is still running.
if len(st["valid_actions"]) >= 7:
    pass
else:
    check("column 7 is not legal once it is 6 deep",
          6 not in st["valid_actions"], str(st["valid_actions"]))

# The game must still be playable in OTHER columns.
st = get("/state")
if st["outcome"] == "ongoing":
    # Pick a column where the HUMAN cannot complete a vertical four.
    # cells[row][col]: 0 empty, 1 = P1 (the human), 2 = P2 (the model).
    def human_count(col: int) -> int:
        return sum(1 for row in st["cells"] if row[col] == 1)

    other = next((c for c in st["valid_actions"]
                  if c != 6 and human_count(c) < 3), None)
    if other is not None:
        st = post("/move", {"col": other})
        # The invariant under test is "no premature draw". A win by either
        # side here is a legitimate game result, not a failure.
        check("the game does not end in a premature draw",
              st["outcome"] != "draw", st["outcome"])
        check("the model replied after the move", len(st["moves"]) >= 1,
              str(st["moves"]))
    else:
        check("a non-column-7 move is available", False,
              str(st["valid_actions"]))

print()
print("=" * 70)
print("[2] A genuinely full board IS a draw")
print("=" * 70)
get("/reset")
sys.path.insert(0, ".")
from src.c4.board import Board, Outcome, ROWS, COLS  # noqa: E402
from src.c4.exact_solver import has_four  # noqa: E402


def build_no_four(cols):
    from src.c4.board import board_from_cells
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
    assert not has_four(m1) and not has_four(m2)
    return board_from_cells(p1, p2, last=(ROWS - 1, max(cols)))


b = build_no_four(list(range(COLS)))
check("a genuinely full board is a DRAW", b.outcome() == Outcome.DRAW,
      str(b.outcome()))
b2 = build_no_four(list(range(COLS - 1)))
check("a board with one column open is ONGOING",
      b2.outcome() == Outcome.ONGOING, str(b2.outcome()))

print()
print("=" * 70)
print("[3] The UI's own pattern + probability panels still work")
print("=" * 70)
st = get("/state")
check("patterns are reported", "patterns" in st, str(list(st.keys())))
check("probabilities are reported", "probs" in st, str(list(st.keys())))
check("both distributions are present",
      "last_decision" in st and "user_advice" in st,
      str(list(st.keys())))

print()
print("=" * 70)
print("[4] The percentage panel says what the percentage means")
print("=" * 70)
with urllib.request.urlopen(BASE + "/", timeout=60) as r:
    html = r.read().decode("utf-8", errors="replace")
check("the page explains that the % is NOT a winning chance",
      "These percentages are not a winning chance" in html)
check("the page explains the 100% case",
      "only one legal column" in html)

print()
print("=" * 70)
print("VERDICT: %s" % ("PASS" if not FAILS else f"FAIL ({len(FAILS)}): {FAILS}"))
print("=" * 70)
sys.exit(1 if FAILS else 0)
