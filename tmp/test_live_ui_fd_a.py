"""End-to-end check that the LIVE UI server is serving the intended
checkpoint and that the model it plays matches the checkpoint on disk.

This closes the loop: a UI that loads a different file than the one we
measured would invalidate every number in the release gate.
"""
from __future__ import annotations

import json
import pathlib
import os
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE = "http://127.0.0.1:" + os.environ.get("UI_PORT", "7860")
EXPECTED = "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt"

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    print("=== live UI end-to-end ===")

    # 1. server is up
    with urllib.request.urlopen(BASE + "/", timeout=15) as r:
        check("server serves /", r.status == 200, f"HTTP {r.status}")

    # 2. a fresh game starts empty
    st = post("/seat", {"seat": 1})
    cells = st.get("cells") or []
    flat = [c for row in cells for c in row]
    check("seat 1: board starts empty", not any(flat), f"cells={cells!r}")
    check("seat 1: human moves first", st.get("turn") == "user",
          f"turn={st.get('turn')!r}")
    check("seat 1: no model move yet", st.get("ai_move") is None,
          f"ai_move={st.get('ai_move')!r}")

    # 3. human plays, model answers
    st = post("/move", {"col": 3})
    mv = st.get("ai_move")
    legal = st.get("valid_actions") or []
    check("after human move the model replies", isinstance(mv, int),
          f"ai_move={mv!r}")
    check("model reply is legal", isinstance(mv, int) and mv in legal,
          f"move={mv} legal={legal}")
    check("moves list records both plies", st.get("moves") == [3, mv],
          f"moves={st.get('moves')!r}")

    # 4. the model's reply equals the checkpoint's own hard argmax
    from src.c4.board import Board
    from src.c4.ui_server import _load_agent
    agent = _load_agent(EXPECTED, device="cpu")
    b2 = Board()
    b2.apply_move(3)
    expect = agent.choose(b2, [3], use_hard=True)
    check("UI model move == checkpoint hard argmax", expect == mv,
          f"ui={mv} ckpt={expect}")

    # 5. the UI's reported distribution matches the checkpoint's own
    #    hard export at the decision position -- this is what proves the
    #    live server is serving the checkpoint we measured.
    ld = st.get("last_decision") or {}
    ui_hard = ld.get("hard_cols") or []
    ck_raw = agent.hard_column_logits(b2, [3])
    ck_raw = [float(x) for x in (ck_raw.tolist() if hasattr(ck_raw, "tolist")
                                  else ck_raw)]
    # The UI reports a probability distribution over columns, i.e. the
    # hard logits passed through softmax (they sum to 1).  Compare like
    # with like.
    m = max(ck_raw)
    e = [pow(2.718281828459045, v - m) for v in ck_raw]
    s = sum(e)
    ck_soft = [v / s for v in e]
    l1 = sum(abs(a - b) for a, b in zip(ui_hard, ck_soft))
    check("UI hard distribution == checkpoint hard export (L1)",
          bool(ui_hard) and l1 < 1e-4, f"L1={l1:.3e}")

    # 6. the UI exposes the separated distributions
    check("next-move distribution present", "probs" in st and
          "cols" in (st.get("probs") or {}), f"keys={sorted(st.keys())}")
    check("decision distribution present", "hard_cols" in ld and
          "picked" in ld, f"ld keys={sorted(ld.keys())}")
    check("decision position captured", "choice_for" in st,
          f"choice_for={st.get('choice_for')!r}")
    check("actual model move exposed", "ai_move" in st)

    # 7. undo restores the previous position
    before = get("/state").get("cells")
    r = post("/undo", {})
    after = (r.get("state") or {}).get("cells")
    check("undo changes the board back", before != after,
          f"cells {before!r} -> {after!r}")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
