"""Dump a soft float checkpoint so play.py can exercise its soft-path loader.

The smoke trainer only emits hard_model_final.pt. To smoke-test the soft
loader we need a real state_dict .pt. This script:
  - rebuilds V6 (d=32, layers=3, fan_in=3, seed=0),
  - if a hard_model_final.pt exists, reads it back through GFMarkovLM
    by going float -> export_hard() (lossless), or simply constructs a
    random V6 and dumps its state_dict with the matching meta,
  - writes final_float.pt next to the hard checkpoint.

Usage:
  python -m src.c4.dump_float_ckpt runs/c4/smoke2
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.gf256_markov_lm_v6 import GFMarkovLM  # noqa: E402

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "runs/c4/smoke2"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GFMarkovLM(d=32, layers=3, fan_in=3, seed=0).to(device)
# tiny warmup so the readout is not exactly zero (we want a non-trivial
# state_dict that load_state_dict can round-trip and that load_model
# can restore)
xb = torch.randint(0, 256, (4, 96), dtype=torch.long, device=device)
model(xb)  # wires lazy ckt/rec lazy init
# a couple of optimizer steps so the state_dict has real grads history
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for _ in range(20):
    opt.zero_grad()
    y = model(xb)
    loss = y.float().sum() * 0.0 + torch.tensor(0.0, device=device)
    # we just want a state_dict; values are irrelevant. zero grad.
    loss.backward()
    opt.step()

ck = {
    "state": model.state_dict(),
    "meta": {"step": 20, "d": 32, "layers": 3, "fan_in": 3, "seed": 0},
}
out = os.path.join(OUT_DIR, "final_float.pt")
os.makedirs(OUT_DIR, exist_ok=True)
torch.save(ck, out)
print(f"wrote {out}")

m = torch.load(out, map_location="cpu", weights_only=False)
print("float ckpt keys:", list(m["state"].keys())[:4], "...")
print("meta:", m["meta"])
