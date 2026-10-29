"""Prove the BATCHED symmetry penalty == the UNBATCHED one, numerically.

The batched version exists only for speed.  If it changed the penalty
value, the A/B would be measuring a different objective, so the
equivalence must be exact, not approximate.

Checks:
  1. soft batched logits == soft per-position logits (max abs diff)
  2. hard batched logits == hard per-position logits (max abs diff)
  3. the resulting penalty value is identical
  4. the penalty's GRADIENT w.r.t. the model parameters agrees
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.c4 import train as T  # noqa: E402
from src.c4.agent import trainer_window  # noqa: E402
from src.c4.board import Board  # noqa: E402
from src.c4.symmetry import (decision_stream, mirrored_decision_stream,  # noqa: E402
                             mirror_equivariance_loss)


def build_model():
    from src.gf256_markov_lm_v6 import GFMarkovLM
    torch.manual_seed(1234)
    m = GFMarkovLM(d=32, layers=3, fan_in=3)
    return m


def main() -> int:
    dev = torch.device("cpu")
    ctx = 192
    model = build_model().to(dev)

    # --- a set of real, non-terminal move sequences of varied lengths ---
    seqs = []
    import random
    rng = random.Random(7)
    while len(seqs) < 6:
        b = Board()
        mv = []
        for _ in range(rng.randint(2, 14)):
            if b.is_win() != 0 or not b.valid_actions():
                break
            c = rng.choice(list(b.valid_actions()))
            mv.append(c)
            b.apply_move(c)
        if len(mv) >= 2 and b.is_win() == 0 and b.valid_actions():
            seqs.append(mv)

    os_ = [decision_stream(m, "v2") for m in seqs]
    ms_ = [mirrored_decision_stream(m, "v2") for m in seqs]

    # --- 1/2. logits ---
    lo_b = T._soft_logits_batch(model, os_, ctx, dev)
    lo_u = torch.stack([T._soft_logits(model, s, ctx, dev) for s in os_])
    d_soft = (lo_b - lo_u).abs().max().item()

    hard = T._hard_export(model)
    ho_b = T._hard_logits_batch(model, hard, os_, ctx)
    ho_u = torch.stack([T._hard_logits(model, hard, s, ctx) for s in os_])
    d_hard = (ho_b - ho_u).abs().max().item()

    lm_b = T._soft_logits_batch(model, ms_, ctx, dev)
    lm_u = torch.stack([T._soft_logits(model, s, ctx, dev) for s in ms_])
    d_soft_m = (lm_b - lm_u).abs().max().item()

    # --- 3. penalty value ---
    legal = []
    for m in seqs:
        b = Board()
        for c in m:
            b.apply_move(c)
        legal.append(torch.tensor([c in b.valid_actions() for c in range(7)]))
    la = torch.stack(legal)
    lam = la.flip(-1)

    pen_b = mirror_equivariance_loss(lo_b, lm_b, la, lam)
    pen_u = mirror_equivariance_loss(lo_u, lm_u, la, lam)
    d_pen = abs(float(pen_b) - float(pen_u))

    # --- 4. gradient ---
    model.zero_grad()
    pen_b.backward()
    g_b = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    pen_u.backward()
    g_u = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert len(g_b) == len(g_u) and g_b, "gradient lists differ"
    d_grad = max((a - b).abs().max().item() for a, b in zip(g_b, g_u))

    print("=== BATCHED vs UNBATCHED symmetry penalty ===")
    print(f"  soft logits   max|diff| : {d_soft:.3e}")
    print(f"  soft mirror   max|diff| : {d_soft_m:.3e}")
    print(f"  hard logits   max|diff| : {d_hard:.3e}")
    print(f"  penalty value |diff|    : {d_pen:.3e}")
    print(f"  gradient      max|diff| : {d_grad:.3e}")
    print(f"  penalty value           : {float(pen_b):.8f}")

    tol = 1e-5
    ok = (d_soft < tol and d_soft_m < tol and d_hard < tol
          and d_pen < tol and d_grad < tol)
    print(f"\n  {'PASS' if ok else 'FAIL'}: batched == unbatched (tol={tol})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
