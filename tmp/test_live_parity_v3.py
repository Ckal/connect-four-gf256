"""PARITY GATE: does the corrected probe agree with the LIVE UI?

The corrected probe (tmp/measure_tactics_v3.py) passes the MOVE HISTORY to
`hard_column_logits`. The live UI (src/c4/ui_server.py) does the same
(lines 207/279/298 pass `list(self.moves)`).

This test PROVES the probe and the shipped path agree, by driving real
games against the running UI over HTTP and comparing:
  * the model's chosen column  (UI's /state "ai_move")
  * the UI's next-move distribution (UI's /state "probs" -> hard_cols)

against the probe's own hard-path argmax and softmax, computed from the
SAME move history.

If the L1 distance between the two distributions is ~0 and the chosen
column matches on every ply, the probe is measuring the shipped model.
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
    with _OPENER.open(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt"
    from src.c4.ui_server import _load_agent
    agent = _load_agent(ckpt, device="cpu")

    rng = random.Random(2024)
    n_plies = 0
    n_col_match = 0
    n_dist = 0
    l1_sum = 0.0
    l1_max = 0.0
    fails = []

    for game in range(6):
        # /reset is a GET route in src/c4/ui_server.py (do_GET handles
        # /reset; do_POST only accepts /seat, /undo and /move).  Posting
        # to it returns 404 "unknown POST" -- a stale-route bug in this
        # test, not a parity failure.
        get("/reset")
        # seat=1 => the HUMAN plays P1 and moves first, so the model is P2
        # and its replies are what we compare against.
        post("/seat", {"seat": 1})
        for _ in range(14):
            st = get("/state")
            if st.get("outcome") != "ongoing":
                break
            moves = st.get("moves") or []
            legal = st.get("valid_actions") or []
            if not legal:
                break
            # Always play a USER move, then read the UI's reply.  The
            # previous version branched on st["turn"] == "user", but with
            # seat=1 the human is P1 and the snapshot reports the turn for
            # the human's own move -- the branch was always taken, so the
            # model was never observed and the test silently compared
            # nothing (0 plies).  This is the same structure as the
            # authoritative tmp/test_live_parity_v5.py.
            post("/move", {"col": rng.choice(legal)})
            st2 = get("/state")
            if st2.get("outcome") != "ongoing":
                break
            ai = st2.get("ai_move")
            mv = list(st2.get("moves") or [])
            if ai is None or len(mv) < 2:
                continue
            pre = mv[:-1]          # the position the model decided in
            b = Board()
            for c in pre:
                b.apply_move(c)
            legal2 = b.valid_actions()
            if not legal2:
                continue
            logits = agent.hard_column_logits(b, pre)
            neg = torch.tensor(-1e9, dtype=logits.dtype)
            mask = torch.tensor([c in legal2 for c in range(7)], dtype=torch.bool)
            probs = torch.softmax(torch.where(mask, logits, neg), dim=-1).tolist()
            pick = max(legal2, key=lambda c: float(logits[c]))

            ui_hard = (st2.get("last_decision") or {}).get("hard_cols")
            n_plies += 1
            if ui_hard:
                n_dist += 1
                l1 = sum(abs((ui_hard[c] or 0.0) - probs[c]) for c in range(7))
                l1_sum += l1
                l1_max = max(l1, l1_max)
                if l1 > 1e-4:
                    fails.append({"kind": "dist_mismatch", "l1": l1,
                                  "moves": pre})
            if ai is not None:
                if ai == pick:
                    n_col_match += 1
                else:
                    fails.append({"kind": "col_mismatch", "ui": ai,
                                  "probe": pick, "moves": pre})

    print("=== LIVE UI <-> CORRECTED PROBE PARITY ===")
    print(f"  model plies observed          : {n_plies}")
    print(f"  distributions compared        : {n_dist}")
    print(f"  mean L1 (UI vs probe)         : {l1_sum / max(n_dist, 1):.3e}")
    print(f"  max  L1 (UI vs probe)         : {l1_max:.3e}")
    print(f"  chosen column matches         : {n_col_match}/{n_plies}")
    if fails:
        print(f"  FAILURES: {len(fails)}")
        for f in fails[:6]:
            print("   ", f)
    else:
        print("  FAILURES: none")
    ok = (n_dist > 0 and l1_max < 1e-4 and n_col_match == n_plies and not fails)
    print(f"\n  PARITY_OK = {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
