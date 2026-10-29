"""PARITY GATE: does the corrected probe agree with the LIVE UI?

The corrected probe (tmp/measure_tactics_v3.py) passes the MOVE HISTORY to
`hard_column_logits`. The live UI (src/c4/ui_server.py) does the same
(lines 207/279/298 pass `list(self.moves)`).

This test PROVES the probe and the shipped path agree, by driving real
games against the running UI over HTTP and comparing, at every model ply:
  * the UI's next-move distribution (probs.hard_cols)
  * the UI's actually-played column (ai_move)

against the probe's own hard-path argmax and softmax, computed from the
SAME move history.

Routes (verified against src/c4/ui_server.py):
  GET  /reset  /state  /probs
  POST /seat {"seat":1}   /move {"col":n}   /undo
"""
from __future__ import annotations

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


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


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

    for _game in range(8):
        get("/reset")
        # seat=1 => the HUMAN plays P1 and moves first, so the model is P2
        # and its replies are what we compare against.
        post("/seat", {"seat": 1})
        for _ply in range(16):
            st = get("/state")
            if st.get("outcome") != "ongoing":
                break
            legal = st.get("valid_actions") or []
            if not legal:
                break
            # The model only ever moves in response to a user move, so
            # drive it that way: play a user move, then read the reply.
            #
            # The previous version branched on st["turn"] == "ai".  With
            # seat=1 the snapshot always reports the turn for the HUMAN's
            # own move, so that branch was never taken and the test
            # compared nothing (0 plies, PARITY_OK = False).  It also
            # never incremented n_dist, so the distribution comparison
            # was dead code.  This is the structure of the authoritative
            # tmp/test_live_parity_v5.py.
            pre_moves = list(st.get("moves") or [])
            post("/move", {"col": rng.choice(legal)})
            st2 = get("/state")
            if st2.get("outcome") != "ongoing":
                break
            ai = st2.get("ai_move")
            if ai is None:
                continue
            mv = list(st2.get("moves") or [])
            if len(mv) < 2:
                continue
            n_plies += 1
            # The position the MODEL decided in is the one after the user's
            # move and before its own: mv[:-1].  The previous version
            # probed pre_moves (the position BEFORE the user's move), so it
            # compared the model's reply against a decision made in a
            # different position -- 74 spurious mismatches.
            b = Board()
            for c in mv[:-1]:
                b.apply_move(c)
            legalp = b.valid_actions()
            if not legalp:
                continue
            logits = agent.hard_column_logits(b, mv[:-1])
            pick = max(legalp, key=lambda c: float(logits[c]))
            if pick == ai:
                n_col_match += 1
            else:
                fails.append({"kind": "col_mismatch", "ui": ai,
                              "probe": pick, "moves": pre_moves})
            # distribution parity at the pre-move position
            neg = torch.tensor(-1e9, dtype=logits.dtype)
            mask = torch.tensor([c in legalp for c in range(7)], dtype=torch.bool)
            probs = torch.softmax(torch.where(mask, logits, neg),
                                  dim=-1).tolist()
            uh = (st2.get("last_decision") or {}).get("hard_cols")
            if uh:
                n_dist += 1
                l1 = sum(abs((uh[c] or 0.0) - probs[c]) for c in range(7))
                l1_sum += l1
                l1_max = max(l1, l1_max)
                if l1 > 1e-4:
                    fails.append({"kind": "dist_mismatch", "l1": l1,
                                  "moves": pre_moves})

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
    ok = (n_plies > 0 and n_col_match == n_plies and not fails)
    print(f"\n  PARITY_OK = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
