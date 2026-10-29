"""Load a SOFT (float) checkpoint into an Agent.

The soft checkpoint is {"state": state_dict, "meta": {...}}. The state dict
keys are `emb_logits` / `layers.N.*` / `readout.N` / `perm_mats` /
`readout_bias` -- NOT the `emb` key used by the V6-class HARD export
payload. The meta carries the exact arch: d, layers, fan_in, lags.

`weights_only=False` is required: the meta dict holds plain Python values
(str/int/list) and torch's safe unpickler cannot rebuild the whole dict.
This mirrors what src/c4/train.py and src/c4/ui_server.py already do for
their own soft checkpoints.
"""
import sys

sys.path.insert(0, ".")

import torch

from src.c4.agent import Agent
from src.gf256_markov_lm_v6 import GFMarkovLM as GFMarkovLM_V6
from src.gf256_markov_lm import GFMarkovLM as GFMarkovLM_V7


def load_soft_agent(path, device="cpu", greedy=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "state" in ck:
        state, meta = ck["state"], (ck.get("meta") or {})
    else:
        state, meta = ck, {}

    d = int(meta.get("d", state["emb_logits"].shape[1]))
    layers = int(meta.get("layers", 3))
    fan_in = int(meta.get("fan_in", 3))
    lags = tuple(int(x) for x in meta.get("lags", (0, 1, 2, 4, 8, 16, 32, 64)))

    cls = GFMarkovLM_V7 if d >= 64 else GFMarkovLM_V6
    kwargs: dict = dict(d=d, layers=layers, fan_in=fan_in, lags=lags)
    if cls is GFMarkovLM_V7:
        kwargs["n_coef"] = int(meta.get("n_coef", 16))

    # CRITICAL: `coef_set` is the GF coefficient LOOKUP TABLE
    # ([0, 1, c2, c3, ...]). It is drawn from a seeded RNG at construction
    # time, and `export_hard()` maps the per-channel argmax INDEX back to a
    # concrete GF coefficient through this table:
    #
    #     coef = coef_set[coef_logits.argmax(-1)]
    #
    # If the loader rebuilds the model with a different RNG draw, the
    # coef_set ORDER changes, so the SAME argmax index resolves to a
    # DIFFERENT GF coefficient -- and the re-export is silently wrong
    # (measured: coef payload delta 170, and the empty-board choice changed
    # from column 3 to column 0).
    #
    # The fix is to rebuild the model with the trainer's SEED so the RNG
    # reproduces the same coef_set, and -- for checkpoints written before
    # this was fixed -- to fall back to the seed recorded in meta.
    seed = meta.get("model_seed", meta.get("seed"))
    if seed is not None:
        kwargs["seed"] = int(seed)

    # `table_bias` scales the identity/one-hot init bonus on the table
    # logits. It does NOT change the RNG draw order, so coef_set is
    # identical for any value -- but the loader must still pass the
    # checkpoint's value so the reconstructed model matches the trained
    # one exactly (the saved state_dict overwrites the table logits
    # anyway, so this only matters for the constructor's bonus lines).
    if "table_bias" in meta:
        kwargs["table_bias"] = float(meta["table_bias"])

    m = cls(**kwargs)
    missing, unexpected = m.load_state_dict(state, strict=False)

    # If the checkpoint recorded the actual coef_set, assert we reproduced
    # it; if we did not, the export would be wrong, so fail loudly.
    want = meta.get("coef_set")
    if want is not None:
        got = list(getattr(m, "coef_set", []))
        if got != [int(x) for x in want]:
            raise SystemExit(
                f"coef_set mismatch rebuilding {path}: checkpoint has "
                f"{list(want)}, rebuilt {got}. The hard export would be "
                f"WRONG (coefficient indices resolve to different GF "
                f"constants). Refusing to load.")

    m.eval()

    dev = torch.device(device)
    m.to(dev)
    ag = Agent(m, dev, greedy=greedy, version=meta.get("transcript", "v2"))
    ag.model.eval()
    info = {"d": d, "layers": layers, "fan_in": fan_in,
            "missing": list(missing), "unexpected": list(unexpected)}
    return ag, info
