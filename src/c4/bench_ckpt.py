"""Is the 50 ms single-move cost kernel-launch overhead or checkpointing?"""
from __future__ import annotations

import time

import torch

from ..gf256_markov_lm_v6 import GFMarkovLM


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for use_ckpt in (True, False):
        model = GFMarkovLM(d=32, layers=3, fan_in=3, bigram_logp=None, seed=0).to(dev)
        model.use_ckpt = use_ckpt
        model.eval()
        for B, T in ((1, 8), (1, 40), (1, 65), (256, 65)):
            x = torch.randint(0, 96, (B, T), device=dev)
            with torch.no_grad():
                for _ in range(10):
                    model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                n = 100 if B == 1 else 10
                for _ in range(n):
                    model(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = (time.perf_counter() - t0) / n
            print(f"use_ckpt={str(use_ckpt):5s} B={B:4d} T={T:3d} : "
                  f"{t * 1e3:8.3f} ms/call  ({t / B * 1e6:8.1f} us/seq)")
        print()


if __name__ == "__main__":
    main()