"""Regression tests for the mirror-equivariance penalty.

These cover the BUG that the regulariser shipped with: the reflected
target was built from an already-masked log-softmax, so on any position
whose common legal support is NOT reflection-invariant, the KL term
became `p * 1e9` and the penalty was ~4e7 instead of a bounded O(1).

Assertions:
  1. exactly equivariant model            -> penalty == 0
  2. maximally non-equivariant model       -> penalty finite, O(1), > 0
  3. ASYMMETRIC legal support (the bug)    -> penalty finite, bounded
  4. penalty is invariant to swapping which side is "original"
  5. penalty is invariant to reflecting the input columns
  6. gradient flows (non-None, finite)
  7. real trained checkpoint               -> penalty finite and O(1)
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.c4.symmetry import mirror_equivariance_loss  # noqa: E402

N = 4
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def main() -> int:
    torch.manual_seed(0)

    # ---- 1. exactly equivariant -------------------------------------
    lo = torch.randn(N, 7)
    lm = lo.flip(-1).clone()          # p_mirror(c) = p_orig(6-c)
    legal = torch.zeros(N, 7, dtype=torch.bool)
    legal[:, [0, 2, 3, 4, 6]] = True
    p = mirror_equivariance_loss(lo, lm, legal, legal.flip(-1))
    check("equivariant -> 0", abs(float(p)) < 1e-6, f"got {float(p):.3e}")

    # ---- 2. non-equivariant, symmetric support ----------------------
    lo = torch.randn(N, 7)
    lm = torch.randn(N, 7)
    p = mirror_equivariance_loss(lo, lm, legal, legal.flip(-1))
    check("non-equivariant -> finite O(1)",
          torch.isfinite(p) and 0.0 < float(p) < 10.0, f"got {float(p):.3e}")

    # ---- 3. ASYMMETRIC common support (the bug trigger) -------------
    # Column 2 is legal on the original but column 4 (=6-2) is NOT legal
    # on the mirror, so `common` is not reflection-invariant.
    legal_a = torch.zeros(N, 7, dtype=torch.bool)
    legal_a[:, [0, 1, 2, 5, 6]] = True      # 2 legal, 4 illegal
    legal_b = legal_a.flip(-1).clone()      # 4 legal, 2 illegal
    lo = torch.randn(N, 7) * 3.0
    lm = torch.randn(N, 7) * 3.0
    p = mirror_equivariance_loss(lo, lm, legal_a, legal_b)
    ok = torch.isfinite(p) and float(p) < 10.0
    check("asymmetric support -> finite bounded", ok, f"got {float(p):.3e}")

    # The old code produced ~4e7 here.  Assert we are far below it.
    check("asymmetric support -> << 1e6", float(p) < 1e6, f"got {float(p):.3e}")

    # ---- 4. swap symmetry -------------------------------------------
    p2 = mirror_equivariance_loss(lm, lo, legal_b, legal_a)
    check("swap-symmetric", abs(float(p) - float(p2)) < 1e-5,
          f"{float(p):.6e} vs {float(p2):.6e}")

    # ---- 5. reflection invariance ------------------------------------
    # Reflecting BOTH logit rows and both legal rows must give the same
    # penalty: the violation measure is defined on reflection orbits.
    lo_r = lo.flip(-1).clone()
    lm_r = lm.flip(-1).clone()
    p3 = mirror_equivariance_loss(lo_r, lm_r, legal_a.flip(-1), legal_b.flip(-1))
    check("reflection-invariant", abs(float(p) - float(p3)) < 1e-5,
          f"{float(p):.6e} vs {float(p3):.6e}")

    # ---- 6. gradient ------------------------------------------------
    lo_g = lo.clone().requires_grad_(True)
    lm_g = lm.clone().requires_grad_(True)
    pg = mirror_equivariance_loss(lo_g, lm_g, legal_a, legal_b)
    pg.backward()
    check("gradient flows", lo_g.grad is not None and torch.isfinite(lo_g.grad).all()
          and float(lo_g.grad.abs().sum()) > 0,
          f"gradsum={None if lo_g.grad is None else float(lo_g.grad.abs().sum())}")

    # ---- 7. real trained checkpoint ---------------------------------
    from src.c4.board import Board
    from src.c4.symmetry import decision_stream, mirrored_decision_stream
    from src.c4 import train as T

    hold = ROOT / "runs/_probe/ab1500/corpus.txt"
    if hold.exists():
        # Load the FLOAT checkpoint: a hard export carries no float
        # weights, so evaluating the soft path on one yields an
        # untrained model (all-zero logits), which is not a valid
        # measurement of the penalty.
        import torch as _t
        from src.gf256_markov_lm_v6 import GFMarkovLM
        ck = _t.load(str(ROOT / "runs/_probe/mir_a_s6000/checkpoints/step6000_soft.pt"),
                     map_location="cpu", weights_only=False)
        meta = ck.get("meta", {})
        d = int(meta.get("d", 32))
        layers = int(meta.get("layers", 3))
        fan_in = int(meta.get("fan_in", 3))
        lags = tuple(int(x) for x in meta.get("lags", (0, 1, 2, 4, 8, 16, 32, 64)))
        model = GFMarkovLM(d=d, layers=layers, fan_in=fan_in, lags=lags)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        model.eval()
        print(f"  (float ckpt loaded: d={d} layers={layers} "
              f"missing={len(missing)} unexpected={len(unexpected)})")

        positions = []
        for ln in hold.read_bytes().split(b"\n"):
            if not ln:
                continue
            mv = [ln[k * 10 + 9] - 65 for k in range(len(ln) // 10)]
            if not mv:
                continue
            b = Board()
            ok = True
            for c in mv:
                if c not in b.valid_actions():
                    ok = False
                    break
                b.apply_move(c)
            if ok and b.is_win() == 0 and b.valid_actions():
                positions.append((b, mv))
            if len(positions) >= 80:
                break

        os_ = [decision_stream(m, "v2") for _, m in positions]
        ms_ = [mirrored_decision_stream(m, "v2") for _, m in positions]
        with torch.no_grad():
            lo = T._soft_logits_batch(model, os_, 192, torch.device("cpu"))
            lm = T._soft_logits_batch(model, ms_, 192, torch.device("cpu"))
        lg = []
        for b, _ in positions:
            lg.append(torch.tensor([c in b.valid_actions() for c in range(7)]))
        la = torch.stack(lg)
        lam = la.flip(-1)
        p = mirror_equivariance_loss(lo, lm, la, lam)
        agree = int((lo.argmax(-1) == (6 - lm.argmax(-1))).sum())
        print(f"  trained model: penalty={float(p):.6f}  "
              f"argmax mirror-agree={agree}/{len(positions)}")
        check("trained model -> finite bounded",
              torch.isfinite(p) and 0.0 <= float(p) < 10.0, f"got {float(p):.3e}")
        check("trained model -> non-degenerate (not all-zero logits)",
              float(lo.abs().sum()) > 1e-6, f"logit abs sum {float(lo.abs().sum()):.3e}")
    else:
        print("  (holdout absent, skipping checkpoint assertion)")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        return 1
    print("ALL SYMMETRY PENALTY ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
