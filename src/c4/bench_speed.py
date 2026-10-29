"""Benchmark: soft vs hard forward pass, and end-to-end per-move cost.

Answers the question: "the hard model should be incredibly fast even when
the board is generated with Python -- how fast, really?"
"""
from __future__ import annotations

import time

import torch

from ..gf256_markov_lm_v6 import GFMarkovLM
from .board import Board, P1, is_terminal
from .transcript import encode_game


def _random_games(rng, n, max_plies=42):
    games = []
    for _ in range(n):
        b = Board()
        moves = []
        while True:
            c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
            if is_terminal(b):
                break
        games.append(encode_game(moves))
    return games


def main() -> None:
    import numpy as np

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GFMarkovLM(d=32, layers=3, fan_in=3, bigram_logp=None, seed=0).to(dev)
    model.eval()
    hard = {k: (v.cpu() if torch.is_tensor(v) else
                [t.cpu() for t in v] if isinstance(v, list) and torch.is_tensor(v[0])
                else v) for k, v in model.export_hard().items()}

    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    games = _random_games(rng, 512)
    t_py = time.perf_counter() - t0
    print(f"device={dev}  python game gen + encode : {t_py * 1e3:8.1f} ms "
          f"for 512 games ({t_py / 512 * 1e6:.1f} us/game)")

    T = 65
    B = 512
    x = torch.randint(0, 96, (B, T), device=dev)
    xc = x.cpu()

    with torch.no_grad():
        for _ in range(3):
            model(x)
            model.hard_forward(xc, hard)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(20):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_soft = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        for _ in range(5):
            model.hard_forward(xc, hard)
        t_hard = (time.perf_counter() - t0) / 5

    print(f"soft forward  B={B} T={T} : {t_soft * 1e3:8.3f} ms/batch "
          f"({t_soft / B * 1e6:8.2f} us/seq)")
    print(f"hard forward  B={B} T={T} : {t_hard * 1e3:8.3f} ms/batch "
          f"({t_hard / B * 1e6:8.2f} us/seq)   <- runs on CPU")

    B1 = 1
    x1 = torch.randint(0, 96, (B1, T), device=dev)
    x1c = x1.cpu()
    with torch.no_grad():
        for _ in range(5):
            model(x1)
            model.hard_forward(x1c, hard)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            model(x1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = (time.perf_counter() - t0) / 50
        t0 = time.perf_counter()
        for _ in range(50):
            model.hard_forward(x1c, hard)
        t2 = (time.perf_counter() - t0) / 50
    print()
    print(f"single-move soft (GPU)   : {t1 * 1e3:8.3f} ms")
    print(f"single-move hard (CPU)   : {t2 * 1e3:8.3f} ms")
    print(f"encode one game in python: {t_py / 512 * 1e3:8.3f} ms")
    print(f"=> end-to-end per move   : {(t1 + t_py / 512) * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()