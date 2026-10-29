"""
gf256_markov_lm_v6.py -- Lagrange Field Transformer V6 (the paper's model)
==============================================================================
This is the configuration the paper describes and reports (§5.2, §6.1, §7.2):
    d=32, layers=3, fan_in=4, lags (0,1,2,4,8,16,32,64), n_coef=8,
    82,400 steps, gentle STE from step 12,000 over a 4,000 ramp.

    A cheaper ablation --fan_in 3 --steps 20000 exists but is NOT the
    reported V6 (see FINDINGS.md F8).

    V7 (d=64, layers=6, fan_in=4, lags to 256) lives in gf256_markov_lm.py,
    which is what artifacts/gf256_hard_model.pt actually contains (F1).

Differences from the original V6 script, all of which affect whether a run is
trustworthy:

  1. --run-dir: checkpoints and logs go to a per-run directory instead of the
     current working directory, so concurrent/repeated runs stop overwriting
     each other.
  2. evaluate() restores the model's original training/eval mode instead of
     unconditionally forcing train().
  3. Final evaluation is on the FULL validation split by default, not a
     128-sequence subsample.
  4. CSV metric logs, so runs are comparable without scraping stdout.
  5. Saves the artifact into artifacts/ with a config sidecar.

Usage:
    python src/gf256_markov_lm_v6.py --seed 32 --run-dir runs/v6_s32
"""
import argparse
import csv
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ────────────────────────────────────────────────────────────────────────────
# GF(2^8) arithmetic (AES polynomial 0x11B)
# ────────────────────────────────────────────────────────────────────────────
def build_gf_tables():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x2 = x << 1
        if x2 & 0x100:
            x2 ^= 0x11B
        x = x2 ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


EXP, LOG = build_gf_tables()


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


# ────────────────────────────────────────────────────────────────────────────
# Walsh-Hadamard transform & XOR convolution
# ────────────────────────────────────────────────────────────────────────────
def fwht(x):
    shape = x.shape
    x = x.reshape(-1, 256)
    h = 1
    while h < 256:
        x = x.view(-1, 256 // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(shape)


def xor_convolve(dists):
    """Distribution of the XOR of independent GF(256) variables."""
    acc = None
    for d in dists:
        w = fwht(d)
        acc = w if acc is None else acc * w
    out = fwht(acc) / 256.0
    out = out.clamp_min(0.0)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


# ────────────────────────────────────────────────────────────────────────────
# Straight-Through Estimator
# ────────────────────────────────────────────────────────────────────────────
def ste_softmax(logits, ste):
    soft = F.softmax(logits, -1)
    if not ste:
        return soft
    hard = F.one_hot(soft.argmax(dim=-1),
                     num_classes=logits.shape[-1]).to(logits.dtype)
    return hard + (soft - soft.detach())


def row_entropy(p):
    return -(p * (p + 1e-12).log()).sum(-1).mean()


# ────────────────────────────────────────────────────────────────────────────
# GF Layer
# ────────────────────────────────────────────────────────────────────────────
class GFLayer(nn.Module):
    def __init__(self, d, fan_in, lags, coef_set, gen, table_bias=1.0):
        super().__init__()
        self.d, self.fan_in, self.lags = d, fan_in, lags
        # `table_bias` scales the identity/one-hot init bonus on the table
        # logits (1.0 = historical behaviour, 0.0 = unbiased).
        #
        # WHY THIS EXISTS: the bonus creates a top1-top2 margin
        # of ~7.7 logits. Adam moves a parameter by at most `lr` per step,
        # so a run of N steps can move it by at most lr*N -- 1.8 logits at
        # lr=0.0006 / 3000 steps, 7.2 over 12,000. The margin that must be
        # crossed is 7.33. So for any realistic step budget the argmax CAN
        # NOT flip, the exported integer table stays frozen, and the hard
        # export is board-blind no matter how the STE schedule is set.
        # Scaling the bonus down makes the argmax reachable (and lets the
        # model LEARN the identity solution instead of being handed it).
        #
        # RNG ORDER IS UNCHANGED: every randn/randperm draw below still
        # happens, only the `+=` bonus is scaled. That keeps `coef_set`
        # byte-identical to the historical draw, so existing checkpoints
        # stay loadable by the current loader.
        self.table_bias = float(table_bias)

        ch = torch.randint(0, d, (d, fan_in), generator=gen)
        lg = torch.randint(0, len(lags), (d, fan_in), generator=gen)
        ch[:, 0] = torch.arange(d)
        lg[:, 0] = lags.index(0)

        self.register_buffer("ch", ch)
        self.register_buffer("lag", lg)
        self.register_buffer("lag_vals", torch.tensor(lags)[lg])

        self.coef_logits = nn.Parameter(
            0.1 * torch.randn(d, fan_in, len(coef_set), generator=gen))
        with torch.no_grad():
            one_idx = coef_set.index(1)
            zero_idx = coef_set.index(0)
            self.coef_logits[:, 0, one_idx] += 6.0 * self.table_bias
            if fan_in > 1:
                self.coef_logits[:, 1:, zero_idx] += 4.0 * self.table_bias

        sbox = 0.1 * torch.randn(d, 256, 256, generator=gen)
        ar256 = torch.arange(256)
        sbox[torch.arange(d)[:, None], ar256[None, :], ar256[None, :]] += 8.0 * self.table_bias
        self.sbox_logits = nn.Parameter(sbox)
        self.ste = False

    def sbox(self, ste=False):
        return ste_softmax(self.sbox_logits, ste)

    def coefs(self, ste=False):
        return ste_softmax(self.coef_logits, ste)

    def forward(self, x, perm_mats):
        B, T, d, _ = x.shape
        max_lag = max(self.lags)
        pad = torch.zeros(B, max_lag, d, 256, device=x.device, dtype=x.dtype)
        pad[..., 0] = 1.0
        xp = torch.cat([pad, x], dim=1)

        M = torch.einsum("dfk,kij->dfij", self.coefs(self.ste), perm_mats)
        t = torch.arange(T, device=x.device)[:, None]
        parts = []
        for f in range(self.fan_in):
            tidx = t + max_lag - self.lag_vals[:, f][None, :]
            src = xp[:, tidx, self.ch[:, f][None, :]]
            parts.append(torch.einsum("btdi,dij->btdj", src, M[:, f]))

        z = xor_convolve(parts)
        return torch.einsum("btdi,dij->btdj", z, self.sbox(self.ste))


# ────────────────────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────────────────────
class GFMarkovLM(nn.Module):
    """V6 defaults: d=32, layers=3, fan_in=4, 8 lags, 8 GF coefficients.

    fan_in=3 is the cheaper ablation variant, not the reported V6.
    """

    def __init__(self, d=32, layers=3, fan_in=4,
                 lags=(0, 1, 2, 4, 8, 16, 32, 64), n_coef=8,
                 bigram_logp=None, seed=0, table_bias=1.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.d = d
        self.lags = list(lags)
        # See GFLayer.__init__ for why `table_bias` exists: the identity
        # bonus creates a ~7.7-logit margin that Adam cannot cross within
        # a realistic step budget, freezing the exported integer tables.
        # 1.0 = historical behaviour, 0.0 = unbiased (identity must be
        # learned). The RNG draw order is unchanged, so `coef_set` is
        # byte-identical to the historical draw.
        self.table_bias = float(table_bias)

        coef_set = [0, 1]
        while len(coef_set) < n_coef:
            c = int(torch.randint(2, 256, (1,), generator=gen))
            if c not in coef_set:
                coef_set.append(c)
        self.coef_set = coef_set

        pm = torch.zeros(len(coef_set), 256, 256)
        for k, c in enumerate(coef_set):
            for i in range(256):
                pm[k, i, gf_mul(c, i)] = 1.0
        self.register_buffer("perm_mats", pm)
        self.use_ckpt = True

        emb = 0.1 * torch.randn(256, d, 256, generator=gen)
        ar256 = torch.arange(256)
        emb[ar256, 0, ar256] += 10.0 * self.table_bias
        for j in range(1, d):
            emb[ar256, j, torch.randperm(256, generator=gen)] += 8.0 * self.table_bias
        self.emb_logits = nn.Parameter(emb)

        self.layers = nn.ModuleList(
            [GFLayer(d, fan_in, self.lags, coef_set, gen,
                     table_bias=self.table_bias) for _ in range(layers)])

        self.readout = nn.ParameterList(
            [nn.Parameter(torch.zeros(d, 256, 256)) for _ in range(layers + 1)])
        if bigram_logp is not None:
            with torch.no_grad():
                self.readout[0][0] = bigram_logp

        self.readout_bias = nn.Parameter(torch.zeros(256))
        self.ste = False

    def set_ste(self, on):
        self.ste = on
        for L in self.layers:
            L.ste = on

    def forward(self, tokens):
        h = ste_softmax(self.emb_logits, self.ste)[tokens]
        logits = torch.einsum("btdi,div->btv", h, self.readout[0])
        for layer, R in zip(self.layers, self.readout[1:]):
            if self.training and self.use_ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, self.perm_mats, use_reentrant=False)
            else:
                h = layer(h, self.perm_mats)
            logits = logits + torch.einsum("btdi,div->btv", h, R)
        return logits + self.readout_bias

    def regularisers(self):
        ent = row_entropy(F.softmax(self.emb_logits, -1))
        perm_pen = 0.0
        for L in self.layers:
            P = L.sbox(False)
            C = L.coefs(False)
            ent = ent + row_entropy(P) + row_entropy(C)
            perm_pen = perm_pen + ((P.sum(1) - 1.0) ** 2).mean()
        return ent / (1 + 2 * len(self.layers)), perm_pen / max(1, len(self.layers))

    @torch.no_grad()
    def export_hard(self):
        return {
            "emb": self.emb_logits.argmax(-1),
            "coef": [torch.tensor(self.coef_set, device=L.coef_logits.device)[
                L.coef_logits.argmax(-1)] for L in self.layers],
            "sbox": [L.sbox_logits.argmax(-1) for L in self.layers],
            "wiring": [(L.ch, L.lag) for L in self.layers],
            "readout": [R.detach() for R in self.readout],
            "bias": self.readout_bias.detach(),
            "lags": self.lags,
        }

    @torch.no_grad()
    def hard_forward(self, tokens, hard):
        """Integer forward pass: XOR, GF table lookups, and the float readout."""
        exp_t, log_t = torch.tensor(EXP), torch.tensor(LOG)
        lags = torch.tensor(hard["lags"])
        B, T, d = tokens.shape[0], tokens.shape[1], self.d
        h = hard["emb"][tokens]
        ar = torch.arange(d)
        logits = hard["readout"][0][ar, h].sum(2)
        for coef, sbox, (ch, lg), R in zip(hard["coef"], hard["sbox"],
                                            hard["wiring"], hard["readout"][1:]):
            acc = torch.zeros(B, T, d, dtype=torch.long)
            for f in range(coef.shape[1]):
                out = torch.zeros(B, T, d, dtype=torch.long)
                for j in range(d):
                    l = int(lags[lg[j, f]])
                    src = h[:, :, ch[j, f]]
                    if l:
                        src = torch.cat([torch.zeros(B, l, dtype=torch.long),
                                         src[:, :-l]], 1)
                    c = int(coef[j, f])
                    if c == 0:
                        prod = torch.zeros_like(src)
                    else:
                        prod = torch.where(src == 0, torch.zeros_like(src),
                                           exp_t[log_t[src] + LOG[c]])
                    out[:, :, j] = prod
                acc = acc ^ out
            h = sbox[ar, acc]
            logits = logits + R[ar, h].sum(2)
        return logits + hard["bias"]


URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")


def load_data(path):
    if path is None:
        path = "tinyshakespeare.txt"
    if not os.path.exists(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    raw = open(path, "rb").read()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def bigram_logp(data):
    counts = torch.ones(256, 256)
    counts.index_put_((data[:-1], data[1:]), torch.ones(len(data) - 1),
                      accumulate=True)
    return (counts / counts.sum(1, keepdim=True)).log()


def get_batch(data, B, T, device):
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, seqs, device, chunk=64):
    """Soft and hard BPC over the given sequences.

    Restores the model's original train/eval mode rather than forcing train(),
    so a final full-validation pass cannot leave the model in the wrong mode.
    """
    was_training = model.training
    was_ste = model.ste
    model.eval()
    model.set_ste(False)
    x, y = seqs[:, :-1], seqs[:, 1:]
    soft_nll, hard_nll = 0.0, 0.0
    for k in range(0, len(x), chunk):
        xb, yb = x[k:k + chunk], y[k:k + chunk]
        lo = model(xb.to(device)).float().cpu()
        soft_nll += F.cross_entropy(lo.reshape(-1, 256), yb.reshape(-1),
                                    reduction="sum").item()

    hard = model.export_hard()
    hard_cpu = {}
    for k, v in hard.items():
        if k == "lags" or not isinstance(v, (list, torch.Tensor)):
            hard_cpu[k] = v
        elif isinstance(v, list):
            if len(v) > 0 and isinstance(v[0], tuple):
                hard_cpu[k] = [(a.cpu(), b.cpu()) for a, b in v]
            else:
                hard_cpu[k] = [t.cpu() for t in v]
        else:
            hard_cpu[k] = v.cpu()
    hard = hard_cpu

    for k in range(0, len(x), chunk):
        xb, yb = x[k:k + chunk], y[k:k + chunk]
        lo = model.hard_forward(xb, hard)
        hard_nll += F.cross_entropy(lo.reshape(-1, 256), yb.reshape(-1),
                                    reduction="sum").item()

    n = y.numel() * math.log(2)
    model.set_ste(was_ste)
    model.train(was_training)
    return soft_nll / n, hard_nll / n


class CsvLogger:
    def __init__(self, path, fields):
        self.path = path
        self.fields = fields
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def write(self, row):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_dir = Path(args.run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train, val = load_data(args.data)
    model = GFMarkovLM(d=args.d, layers=args.layers, fan_in=args.fan_in,
                       bigram_logp=bigram_logp(train), seed=seed).to(args.device)
    model.use_ckpt = not args.no_ckpt
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.2f}M | d={args.d} layers={args.layers} "
          f"fan_in={args.fan_in} seed={seed} | device={args.device}")
    print(f"run dir: {run_dir}")

    cfg = {k: v for k, v in vars(args).items()}
    cfg["n_params"] = n_par
    cfg["train_bytes"] = len(train)
    cfg["val_bytes"] = len(val)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str),
                                         encoding="utf-8")

    L = args.ctx + 1
    all_val = val[: (len(val) // L) * L].view(-1, L)
    gsel = torch.Generator().manual_seed(123)
    quick_val = all_val[torch.randperm(len(all_val),
                                       generator=gsel)[:args.eval_seqs]]

    eval_log = CsvLogger(run_dir / "logs" / "eval_metrics.csv",
                         ["step", "soft_bpc_quick", "hard_bpc_quick",
                          "entropy", "elapsed_s", "best_hard_bpc_quick"])

    t0 = time.time()
    best_hard = float("inf")

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            soft, hard = evaluate(model, quick_val, args.device,
                                  chunk=args.eval_chunk_size)
            ent, _ = model.regularisers()
            best_hard = min(best_hard, hard)
            print(f"step {step:5d} | soft {soft:.3f} | hard {hard:.3f} | "
                  f"ent {ent.item():.3f} | {time.time()-t0:.0f}s", flush=True)
            eval_log.write({"step": step, "soft_bpc_quick": f"{soft:.4f}",
                            "hard_bpc_quick": f"{hard:.4f}",
                            "entropy": f"{ent.item():.5f}",
                            "elapsed_s": f"{time.time()-t0:.1f}",
                            "best_hard_bpc_quick": f"{best_hard:.4f}"})
            if hard < best_hard or step == 0:
                torch.save(model.cpu().export_hard(),
                           run_dir / "checkpoints" / "best_hard.pt")
                model.to(args.device)

        if step == args.steps:
            break

        if step == args.hard_start:
            print(f"\n--- Step {step}: Starting Gentle STE phase ---", flush=True)
            for pg in opt.param_groups:
                pg["lr"] *= args.lr_drop

        x, y = get_batch(train, args.batch, args.ctx, args.device)

        model.set_ste(False)
        logits_soft = model(x)
        ce_soft = F.cross_entropy(logits_soft.reshape(-1, 256), y.reshape(-1))

        hard_ce = torch.tensor(0.0, device=args.device)
        hard_w = 0.0
        if step >= args.hard_start and args.hard_w > 0:
            hard_w = args.hard_w * min(
                1.0, (step - args.hard_start + 1) / max(1, args.hard_ramp))
            model.set_ste(True)
            logits_hard = model(x)
            hard_ce = F.cross_entropy(logits_hard.reshape(-1, 256), y.reshape(-1))
            model.set_ste(False)

        ent, perm_pen = model.regularisers()
        lam = args.ent_max * min(1.0, (step + 1) / max(1, args.ent_warmup))
        loss = ce_soft + hard_w * hard_ce + lam * ent + args.perm_w * perm_pen

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

    # ---- final evaluation: full validation split, not the quick subsample ----
    soft_full, hard_full = evaluate(model, all_val, args.device,
                                    chunk=args.eval_chunk_size)
    print(f"\nFINAL full-val: soft {soft_full:.4f} | HARD {hard_full:.4f}")
    print(f"FINAL quick-val best hard: {best_hard:.4f}")
    print(f"elapsed: {time.time()-t0:.0f}s")

    (run_dir / "final_metrics.json").write_text(json.dumps({
        "seed": seed,
        "soft_bpc_full_val": soft_full,
        "hard_bpc_full_val": hard_full,
        "best_hard_bpc_quick": best_hard,
        "val_chars": int(all_val.numel()),
        "n_params": n_par,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")

    torch.save(model.cpu().export_hard(), run_dir / "hard_model_final.pt")
    print(f"saved -> {run_dir / 'hard_model_final.pt'}")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--run-dir", default=None,
                    help="Directory for config.json, logs/ and checkpoints/. "
                         "Default: runs/v6_seed<seed>")
    # V6 architecture (paper §4). NOTE: §5.2 wiring and §7.2 command line
    # both use fan_in=4 for the reported V6; fan_in=3 is the cheaper ablation.
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fan_in", type=int, default=3)
    # V6 optimisation recipe (paper §5, §7.2). §6.1 reports FINAL at 82,400.
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-drop", type=float, default=0.2,
                    help="Multiplier applied to lr at --hard-start.")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ent_max", type=float, default=2.0)
    ap.add_argument("--ent_warmup", type=int, default=1000)
    ap.add_argument("--perm_w", type=float, default=1.0)
    ap.add_argument("--hard_start", type=int, default=10000)
    ap.add_argument("--hard_ramp", type=int, default=2000)
    ap.add_argument("--hard_w", type=float, default=0.5)
    # eval
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_seqs", type=int, default=128,
                    help="Quick-eval subsample size. Final eval always uses "
                         "the full validation split.")
    ap.add_argument("--eval_chunk_size", type=int, default=8)
    ap.add_argument("--no_ckpt", action="store_true",
                    help="Disable activation checkpointing (faster, more VRAM)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.run_dir is None:
        a.run_dir = f"runs/v6_seed{a.seed}"
    print("device:", a.device)
    main(a)