"""Validate src/c4/symmetry.py against the REAL agent path.

Checks:
  1. `decision_stream(moves)` is byte-identical to what the agent feeds
     the model (prompt_tokens).
  2. `mirrored_decision_stream(moves)` is byte-identical to
     prompt_tokens of the reflected move list.
  3. The mirrored stream's board equals the canonical mirror_board of the
     original board -- including per-cell ownership, not just heights.
  4. The penalty is ZERO for an exactly equivariant model and POSITIVE
     for a broken one.
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.agent import prompt_tokens, board_from_moves     # noqa: E402
from src.c4.board import Board                               # noqa: E402
from src.c4.mirror import mirror_board                       # noqa: E402
from src.c4.symmetry import (decision_stream, mirrored_decision_stream,
                             mirror_equivariance_loss,
                             verify_mirror_stream)           # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def main() -> int:
    print("=== 1-3: stream construction ===")
    cases = [
        [3],
        [0, 6],
        [0, 1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1, 0, 0, 6],
        [1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 3, 3],
    ]
    for mv in cases:
        ds = decision_stream(mv)
        pt = bytes(prompt_tokens(list(mv)).to(torch.uint8).tolist())
        check(f"decision_stream == prompt_tokens for {mv}", ds == pt,
              f"{len(ds)} vs {len(pt)}")

        mds = mirrored_decision_stream(mv)
        mpt = bytes(prompt_tokens([6 - c for c in mv]).to(torch.uint8).tolist())
        check(f"mirrored_decision_stream == prompt_tokens(reflected) for {mv}",
              mds == mpt)

        # board-level: reflected move list reproduces mirror_board
        b = board_from_moves(mv)
        mb = mirror_board(b)
        b2 = board_from_moves([6 - c for c in mv])
        check(f"reflected moves reproduce mirror_board for {mv}",
              b2.p1 == mb.p1 and b2.p2 == mb.p2 and b2.turn == mb.turn)

        # the module's own assertion must hold
        try:
            verify_mirror_stream(mv)
            check(f"verify_mirror_stream passes for {mv}", True)
        except Exception as e:
            check(f"verify_mirror_stream passes for {mv}", False, str(e))

    print("\n=== 4: the penalty ===")
    torch.manual_seed(0)
    legal = torch.ones(2, 7, dtype=torch.bool)

    # An exactly equivariant model: logits on the mirrored board are the
    # reflection of the logits on the original board.
    lo = torch.randn(2, 7)
    lm = lo.flip(-1)
    pen_ok = mirror_equivariance_loss(lo, lm, legal, legal)
    check("penalty == 0 for an equivariant model",
          float(pen_ok) < 1e-5, f"got {float(pen_ok):.3e}")

    # A broken model: the mirrored board gives a different answer.
    lb = torch.randn(2, 7)
    pen_bad = mirror_equivariance_loss(lo, lb, legal, legal)
    check("penalty > 0 for a non-equivariant model",
          float(pen_bad) > 1e-3, f"got {float(pen_bad):.3e}")

    # The penalty must be invariant to the order of the two positions.
    p1 = mirror_equivariance_loss(lo, lb, legal, legal)
    p2 = mirror_equivariance_loss(lb, lo, legal, legal)
    check("penalty is symmetric in the two orientations",
          abs(float(p1) - float(p2)) < 1e-5,
          f"{float(p1):.6f} vs {float(p2):.6f}")

    # Gradient flows.
    lo2 = lo.clone().requires_grad_(True)
    lm2 = lm.clone().requires_grad_(True)
    mirror_equivariance_loss(lo2, lm2, legal, legal).backward()
    check("gradient reaches the original logits",
          lo2.grad is not None and float(lo2.grad.abs().sum()) > 0)
    check("gradient reaches the mirrored logits",
          lm2.grad is not None and float(lm2.grad.abs().sum()) > 0)

    # Illegal columns must not contribute.
    leg_partial = legal.clone()
    leg_partial[:, 4:] = False
    lo3 = torch.randn(2, 7)
    lm3 = lo3.flip(-1)
    pen_masked = mirror_equivariance_loss(lo3, lm3, leg_partial, leg_partial)
    check("penalty == 0 for an equivariant model under a legal mask",
          float(pen_masked) < 1e-5, f"got {float(pen_masked):.3e}")

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
