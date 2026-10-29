"""PARITY GATE: does the corrected probe agree with the LIVE UI?

The corrected probe (tmp/measure_tactics_v3.py) passes the MOVE HISTORY to
`hard_column_logits`. The live UI (src/c4/ui_server.py) does the same
(ui_server.py:207/279/298 all pass `list(self.moves)`).

UI FLOW (src/c4/ui_server.py user_move()):
  1. decision = _probs_locked(board, moves)      <- distribution AT the
                                                    decision position
  2. col = agent.choose(board, moves, use_hard=True)   <- the played move
  3. board.apply_move(col); moves.append(col)
  4. user_advice = _probs_locked(board, moves)   <- advice for the human

So after ONE user move, the snapshot contains:
  * moves            = history INCLUDING the model's reply
  * ai_move          = the column the model actually played
  * last_decision    = the distribution the model chose from, at the
                       pre-reply position
  * user_advice      = the distribution for the human's next move

The PRE-REPLY position is reconstructible: it is `moves` with the model's
reply removed, i.e. moves[:-1].

This test therefore compares, for every model ply:
  * probe argmax  vs  UI's ai_move
  * probe softmax vs  UI's last_decision.hard_cols
over the SAME move history.
"""
from __future__ import annotations

import http.cookiejar
import json
import pathlib
import random
import os
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.c4.board import Board  # noqa: E402

BASE = "http://127.0.0.1:" + os.environ.get("UI_PORT", "7860")


# The server is session-scoped: each visitor gets an opaque session id in a
# cookie, and each session owns its own board. A test that makes several
# requests WITHOUT carrying the cookie therefore sees a brand-new empty
# board on every request -- which looks exactly like a broken server.
# Every request here goes through one shared opener that keeps the cookie
# jar, so all requests hit the SAME session.
_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_JAR))


def get(path):
    with _OPENER.open(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=60) as r:
        return json.loads(r.read().decode())


def probe(agent, moves):
    """Hard-path logits for a move history -> (probs over 7, argmax)."""
    b = Board()
    for c in moves:
        b.apply_move(c)
    legal = b.valid_actions()
    logits = agent.hard_column_logits(b, moves)
    neg = torch.tensor(-1e9, dtype=logits.dtype)
    mask = torch.tensor([c in legal for c in range(7)], dtype=torch.bool)
    probs = torch.softmax(torch.where(mask, logits, neg), dim=-1).tolist()
    pick = max(legal, key=lambda c: float(logits[c]))
    return probs, pick


def main() -> int:
    ckpt = (sys.argv[1] if len(sys.argv) > 1 else
            "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt")
    from src.c4.ui_server import _load_agent
    agent = _load_agent(ckpt, device="cpu")

    rng = random.Random(2024)
    n_plies = 0
    n_col_match = 0
    n_dist = 0
    l1_sum = 0.0
    l1_max = 0.0
    fails = []

    for _game in range(10):
        get("/reset")
        post("/seat", {"seat": 1})
        for _ply in range(18):
            st = get("/state")
            if st.get("outcome") != "ongoing":
                break
            legal = st.get("valid_actions") or []
            if not legal:
                break
            post("/move", {"col": rng.choice(legal)})
            st2 = get("/state")
            ai = st2.get("ai_move")
            if ai is None:
                break
            moves = list(st2.get("moves") or [])
            if len(moves) < 2:
                continue
            pre = moves[:-1]          # position the model decided in
            probs, pick = probe(agent, pre)
            n_plies += 1
            if pick == ai:
                n_col_match += 1
            else:
                fails.append({"kind": "col_mismatch", "ui": ai,
                              "probe": pick, "pre": pre})
            dec = st2.get("last_decision") or {}
            uh = dec.get("hard_cols")
            if uh:
                n_dist += 1
                l1 = sum(abs((uh[c] or 0.0) - probs[c]) for c in range(7))
                l1_sum += l1
                l1_max = max(l1_max, l1)
                if l1 > 1e-4:
                    fails.append({"kind": "dist_mismatch", "l1": l1, "pre": pre,
                                  "ui": uh, "probe": [round(x, 6) for x in probs]})

    print("=== LIVE UI <-> CORRECTED PROBE PARITY ===")
    print(f"  model plies observed          : {n_plies}")
    print(f"  chosen column matches         : {n_col_match}/{n_plies}")
    print(f"  distribution comparisons      : {n_dist}")
    print(f"  mean L1 (UI vs probe)         : {l1_sum / max(n_dist, 1):.3e}")
    print(f"  max  L1 (UI vs probe)         : {l1_max:.3e}")
    if fails:
        print(f"  FAILURES: {len(fails)}")
        for f in fails[:8]:
            print("   ", f)
    else:
        print("  FAILURES: none")
    ok = (n_plies > 0 and n_col_match == n_plies and n_dist > 0
          and l1_max < 1e-4 and not fails)
    print(f"\n  PARITY_OK = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
