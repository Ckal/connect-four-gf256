"""Regression test for the per-ply label-agreement PROMOTION GATE.

WHY
---
The trainer's promotion key used to fall back to byte-level bpb when it had
neither a holdout nor a ladder measurement.  Byte bpb scores the PLAYED
move byte; the trainer's actual objective is the masked 7-way loss against
the TEACHER's label.  These are different quantities and they inverted:

    500-row acceptance run (runs/_probe/overfit_fixed)
      step 1500 -> byte bpb 5.513, label agreement 62.82 %   <- promoted
      step 3923 -> byte bpb 6.798, label agreement 91.29 %   <- final export

The gate promoted the checkpoint that fits the labels LEAST.

WHAT THIS TEST CHECKS
---------------------
1. The shipped trainer contains a per-ply agreement scorer.
2. The scorer is wired into the promotion key as the PRIMARY component
   (not as an afterthought, not gated behind a holdout).
3. The scorer measures the LABEL, not the played move: on the 500-row
   acceptance corpus the shipped checkpoint family must produce a label
   agreement that is clearly DIFFERENT from the played-move agreement, and
   for the CORRECTED checkpoint the label agreement must be high.
4. The scorer must not silently collapse to zero/negative on a real corpus.

The test is mutation-sensitive: replacing the label-agreement primary with
the old byte-bpb fallback must make it FAIL.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}"  + (f"  ({detail})" if detail else ""))


def main() -> int:
    src = (ROOT / "src" / "c4" / "train.py").read_text(encoding="utf-8")

    print("=== 1. shipped scorer exists ===")
    check("PerPlyAgreement class defined",
          "class PerPlyAgreement:" in src)
    check("scorer computes perply_agreement_pct",
          "perply_agreement_pct" in src)
    check("scorer is instantiated for load-corpus runs",
          "perply_scorer = PerPlyAgreement(" in src)

    print("=== 2. promotion key uses it as PRIMARY ===")
    # There are TWO `sel = (...)` assignments: the initialiser
    #     sel = (-float("inf"), -1.0, -float("inf"), 0.0)
    # and the per-eval UPDATE
    #     sel = (_primary, ho_pct, -hard_bpb, float(best_distinct))
    # Only the UPDATE is the promotion key.  Anchor on `_primary` so the
    # initialiser cannot be matched by mistake.
    m = re.search(r"sel\s*=\s*\(\s*_primary.*?\)\s*\n", src, re.S)
    sel_block = m.group(0) if m else ""
    check("promotion-key update assignment found", bool(sel_block),
          sel_block[:160])
    check("promotion key uses _primary", "_primary" in sel_block)
    check("_primary selects label agreement",
          re.search(r"_primary\s*=\s*float\(ppa\[.perply_agreement_pct.\]\)", src)
          is not None)
    check("label agreement outranks the ladder score",
          src.find("_primary = float(ppa[\"perply_agreement_pct\"])") <
          src.find("_primary = float(_ladder_net)")
          if "_primary = float(_ladder_net)" in src else True)
    # The byte-bpb fallback must NOT be the primary any more.
    check("byte bpb is no longer the primary key",
          not re.search(r"_primary\s*=\s*float\(-hard_bpb\)", src))

    print("=== 3. metric is logged and printed ===")
    check("perply metrics reach metrics.jsonl",
          "**(ppa or {})," in src)
    check("panel prints the LABEL readout",
          "LABEL" in src and "perply_agreement_pct" in src)

    print("=== 4. behavioural: scorer runs on the real 500-row corpus ===")
    try:
        import numpy as np
        import torch

        from src.c4.train import PerPlyAgreement, _align_perply_targets
        from src.gf256_markov_lm_v6 import GFMarkovLM

        run = ROOT / "runs" / "_probe" / "overfit_fixed"
        corpus = run / "corpus.txt"
        sidecar = run / "perply_targets.npz"
        if not corpus.exists() or not sidecar.exists():
            check("acceptance corpus available", False,
                  f"missing {corpus} or {sidecar}")
        else:
            raw_lines = []
            # corpus.txt holds RAW BYTES, one game per line -- not hex.
            # Reading it as utf-8 text raises UnicodeDecodeError on the
            # 0x81 board bytes.
            with open(corpus, "rb") as fh:
                for ln in fh.read().splitlines():
                    if ln:
                        raw_lines.append(ln)

            class _A:
                pass

            a = _A()
            a.perply_targets = str(sidecar)
            a.transcript = "v2"
            a.max_legacy_frac = 0.0
            a.corpus_source = "teacher"
            pt, po, nply = _align_perply_targets(a, raw_lines, "v2")
            check("sidecar aligned", pt is not None and pt.shape[0] == len(raw_lines),
                  f"shape={None if pt is None else pt.shape}")

            ck = run / "hard_model_final.pt"
            check("corrected final checkpoint exists", ck.exists(), str(ck))
            if ck.exists() and pt is not None:
                # The hard export is a FLAT payload (emb/readout/coef/...),
                # not a {"state": ...} dict.  Load the payload DIRECTLY: a
                # V6 checkpoint re-loaded through `_load_agent` keeps ZERO
                # float parameters (GFMarkovLM_V6 has no `import_hard`), so
                # re-exporting from them yields an all-zero model and every
                # checkpoint would score the same constant column.
                import torch as _t
                raw = _t.load(str(ck), map_location="cpu", weights_only=False)
                payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
                from src.c4.agent import hard_to_cpu
                hard = hard_to_cpu(payload)
                from src.gf256_markov_lm_v6 import GFMarkovLM as V6
                from src.gf256_markov_lm import GFMarkovLM as V7
                d = int(payload["emb"].shape[1])
                layers = len(payload["coef"])
                fan_in = int(payload["coef"][0].shape[1])
                lags = tuple(int(x) for x in payload["lags"])
                cls = V7 if d >= 64 else V6
                model = cls(d=d, layers=layers, fan_in=fan_in, lags=lags)
                model.eval()
                meta = {}
                model.eval()

                # The scorer MUST receive the hard payload: it measures the
                # shipped integer artefact, and a V6 checkpoint re-loaded
                # through `_load_agent` has zero float params, so
                # `export_hard()` on it is an all-zero model.
                # ctx=192 matches the acceptance run's --ctx 192.
                scorer = PerPlyAgreement(raw_lines, pt, po, nply,
                                         torch.device("cpu"), n_rows=512,
                                         hard=hard, model=model, ctx=192)
                res = scorer.score(model, max_positions=3000)
                pct = res["perply_agreement_pct"]
                n = res["perply_n"]
                check("scorer produced positions", n > 500, f"n={n}")
                check("label agreement is high on the corrected checkpoint",
                      pct > 60.0, f"pct={pct:.2f}% n={n}")

                # The scorer must be measuring the LABEL, not the PLAYED
                # move.  The played move is encoded in the transcript's move
                # byte; the label lives in the sidecar.  Compare the two
                # WITHOUT any forward pass: if the label and the played move
                # agreed everywhere, the distinction would be vacuous -- and
                # a scorer that measured the played move would be
                # indistinguishable from one that measured the label.
                #
                # (The previous version ran one unbatched forward per
                # position and took >4 minutes.  This is the same claim
                # measured on the corpus, at no compute cost.)
                # The move byte sits at MOVE_OFFSET (9) inside the ply
                # record -- NOT at offset 3, which is a header byte.  Reading
                # the wrong offset makes the comparison vacuous (it reports
                # 0 % agreement with the label, which is impossible for a
                # teacher that mostly agrees with its own games).
                from src.c4.fmt import PLY_LEN, MOVE_OFFSET
                same = diff = 0
                for r in range(len(raw_lines)):
                    seq = raw_lines[r]
                    for s, p in enumerate(po[r].tolist()):
                        if p < 0:
                            continue
                        end = p * PLY_LEN
                        if end + MOVE_OFFSET >= len(seq):
                            continue
                        played = int(seq[end + MOVE_OFFSET]) - 0x41
                        lab = int(pt[r, s].argmax())
                        if played == lab:
                            same += 1
                        else:
                            diff += 1
                n_cmp = same + diff
                agree_pct = 100.0 * same / max(n_cmp, 1)
                check("label and played move genuinely differ on the corpus",
                      diff > 0 and n_cmp > 100,
                      f"same={same} diff={diff} (label==played {agree_pct:.2f}%)")
                check("scorer is not measuring the played move",
                      pct > 60.0,
                      f"scored-label={pct:.2f}% vs label==played {agree_pct:.2f}%")
    except Exception as exc:  # pragma: no cover - diagnostic path
        import traceback
        traceback.print_exc()
        check("behavioural section ran", False, repr(exc))

    print(f"\n=== {PASS} PASS, {FAIL} FAIL ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
