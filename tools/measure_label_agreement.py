"""Measure direct sidecar-label agreement for ARBITRARY checkpoints.

Same measurement as tmp/audit_overfit_fit.py, but:
  * takes a list of checkpoint paths on the command line,
  * measures the HARD export via the canonical `_load_agent` path (the one
    the UI serves) AND the FLOAT weights via the same architecture, so the
    hard-vs-float gap is visible per checkpoint,
  * so the acceptance gate can be checked against the final export rather
    than a mid-run "best_hard.pt".

Usage:
  ./.venv-c4/Scripts/python.exe tmp/measure_label_agreement.py \
      <corpus.txt> <perply_targets.npz> <ckpt1> [ckpt2 ...]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.board import Board  # noqa: E402
from src.c4.fmt import PLY_LEN  # noqa: E402

COL_BYTE_OFFSET = 0x41
NEG = -1e9


def read_rows(path, n):
    rows = []
    with open(path, "rb") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            rows.append(line.rstrip(b"\r\n"))
    return rows


def hard_agent(ckpt):
    """Canonical serving path -- exactly what the UI serves."""
    from src.c4.ui_server import _load_agent
    return _load_agent(str(ckpt), device="cpu"), True


def float_agent(ckpt):
    """Same architecture, float weights, from a soft checkpoint."""
    from src.c4.agent import Agent
    from src.gf256_markov_lm_v6 import GFMarkovLM

    # weights_only=False: the file is produced by this repo's own training
    # run, so it is trusted input.
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    state = raw["state"]
    kw = dict(d=meta.get("d", 32), layers=meta.get("layers", 3),
              fan_in=meta.get("fan_in", 3), seed=meta.get("seed", 32),
              table_bias=meta.get("table_bias", 0.0))
    lags = meta.get("lags")
    if lags:
        kw["lags"] = tuple(lags)
    model = GFMarkovLM(**kw)
    model.load_state_dict(state)
    model.eval()
    version = meta.get("transcript", "v2")
    return Agent(model, torch.device("cpu"), greedy=True, version=version), False


def logits_for(agent, board, moves, hard):
    """Column logits via the agent's own API, legal-masked."""
    if hard:
        lg = agent.hard_column_logits(board, moves)
    else:
        lg = agent.column_logits(board, moves)
    return lg.detach().cpu().double().numpy()


def measure(agent, corpus, side, label, hard=True):
    z = np.load(side, allow_pickle=True)  # repo's own pipeline output
    tgt, ply_of = z["target"], z["ply_of"]
    rows = read_rows(corpus, len(tgt))

    hit = 0
    n = 0
    mass = 0.0
    src_stats = {}
    for i in range(len(rows)):
        rec = rows[i]
        nrec = len(rec) // PLY_LEN
        mv = [int(rec[q * PLY_LEN + 9]) - COL_BYTE_OFFSET for q in range(nrec)]
        for s in range(tgt.shape[1]):
            r = tgt[i, s]
            mx = r.max()
            if mx <= 1e-6:
                continue
            p = int(ply_of[i, s])
            if p < 1 or p >= nrec:
                continue
            b = Board()
            ok = True
            for m in mv[:p]:
                if m < 0 or m >= 7 or b.heights[m] >= 6:
                    ok = False
                    break
                b.apply_move(m)
            if not ok:
                continue
            legal = [c for c in range(7) if b.heights[c] < 6]
            if not legal:
                continue
            arr = logits_for(agent, b, mv[:p], hard).copy()
            for c in range(7):
                if c not in legal:
                    arr[c] = NEG
            e = np.exp(arr - arr.max())
            pr = e / e.sum()
            tied = tuple(int(c) for c in np.flatnonzero(r >= mx - 1e-6))
            n += 1
            mass += float(pr[list(tied)].sum())
            src = int(z["teacher"][i, s])
            st = src_stats.setdefault(src, [0, 0])
            st[0] += 1
            if int(np.argmax(pr)) in tied:
                hit += 1
                st[1] += 1
    return {
        "checkpoint": label,
        "n_positions": n,
        "pct_argmax_inside_label": round(100.0 * hit / n, 2) if n else None,
        "mean_prob_mass_on_label": round(mass / n, 4) if n else None,
        "by_source": {
            str(k): {"n": v[0], "pct_hit": round(100.0 * v[1] / v[0], 2)}
            for k, v in sorted(src_stats.items())
        },
    }


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    corpus = ROOT / sys.argv[1]
    side = ROOT / sys.argv[2]
    ckpts = [ROOT / a for a in sys.argv[3:]]

    out = []
    for ck in ckpts:
        if not ck.exists():
            print(f"!! missing {ck}")
            continue
        # HARD path: the canonical serving path.
        try:
            ag, is_hard = hard_agent(ck)
            res = measure(ag, corpus, side, f"{ck.name} [hard]", hard=is_hard)
        except Exception as exc:
            print(f"!! {ck.name} hard: {exc!r}")
            res = {"checkpoint": f"{ck.name} [hard]", "error": repr(exc)}
        out.append(res)
        print(json.dumps(res))
        # FLOAT path: same architecture, unquantized weights.
        try:
            ag, is_hard = float_agent(ck)
            res = measure(ag, corpus, side, f"{ck.name} [float]", hard=is_hard)
        except Exception as exc:
            print(f"!! {ck.name} float: {exc!r}")
            res = {"checkpoint": f"{ck.name} [float]", "error": repr(exc)}
        out.append(res)
        print(json.dumps(res))

    # OUTPUT PATH MUST BE PER-INVOCATION. A fixed path meant that two
    # concurrent battery chains (or two arms in one chain) overwrote each
    # other's JSON -- the last writer won and the other result vanished
    # with no error. Derive it from the checkpoint name(s) instead.
    stem = "-".join(p.stem for p in ckpts[:3])
    if len(ckpts) > 3:
        stem += f"-and-{len(ckpts)-3}more"
    op = ROOT / f"runs/_probe/label_agreement_{stem}.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {op}")

    good = [r for r in out if r.get("pct_argmax_inside_label") is not None]
    if good:
        best = max(good, key=lambda r: r["pct_argmax_inside_label"])
        print(f"\nBEST: {best['pct_argmax_inside_label']} %  ({best['checkpoint']})")
        print(f"ACCEPTANCE GATE (>90 %): "
              f"{'PASS' if best['pct_argmax_inside_label'] > 90 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
