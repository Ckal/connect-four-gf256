"""
  -  Lagrange Field Transformer, prototype v7 (Scale-Up)
=================================================================
V7 changes:
  * Scaled to d=64, layers=6, fan_in=4 (~120M parameters)
  * Expanded lags to 256 bytes for deep paragraph-level memory
  * Adjusted training schedule for larger capacity (200k steps)
  * Added easy local file loading for "Expert" domain training
"""
import argparse, math, os, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import random
import numpy as np

def build_gf_tables():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x2 = x << 1
        if x2 & 0x100: x2 ^= 0x11B
        x = x2 ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log

EXP, LOG = build_gf_tables()

def gf_mul(a, b):
    if a == 0 or b == 0: return 0
    return EXP[LOG[a] + LOG[b]]

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
    acc = None
    for d in dists:
        w = fwht(d)
        acc = w if acc is None else acc * w
    out = fwht(acc) / 256.0
    out = out.clamp_min(0.0)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)

def ste_softmax(logits, ste):
    soft = F.softmax(logits, -1)
    if not ste: return soft
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)
    return hard + (soft - soft.detach())

def row_entropy(p):
    return -(p * (p + 1e-12).log()).sum(-1).mean()

class GFLayer(nn.Module):
    def __init__(self, d, fan_in, lags, coef_set, gen, table_bias=1.0):
        super().__init__()
        self.d, self.fan_in, self.lags = d, fan_in, lags
        # See src/gf256_markov_lm_v6_1.py GFLayer.__init__ for why this
        # exists: the identity/one-hot init bonus creates a ~7.7-logit
        # margin that Adam cannot cross within a realistic step budget,
        # which freezes the exported integer tables and leaves the hard
        # export board-blind. 1.0 = historical, 0.0 = unbiased.
        # The RNG draw order is unchanged, so coef_set is unaffected.
        self.table_bias = float(table_bias)
        K = len(coef_set)

        ch = torch.randint(0, d, (d, fan_in), generator=gen)
        lg = torch.randint(0, len(lags), (d, fan_in), generator=gen)
        ch[:, 0] = torch.arange(d)
        lg[:, 0] = lags.index(0)

        self.register_buffer("ch", ch)
        self.register_buffer("lag", lg)
        self.register_buffer("lag_vals", torch.tensor(lags)[lg])

        self.coef_logits = nn.Parameter(0.1 * torch.randn(d, fan_in, K, generator=gen))
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

    def sbox(self, ste=False): return ste_softmax(self.sbox_logits, ste)
    def coefs(self, ste=False): return ste_softmax(self.coef_logits, ste)

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

class GFMarkovLM(nn.Module):
    def __init__(self, d=64, layers=6, fan_in=4, lags=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
                 n_coef=16, bigram_logp=None, seed=0, table_bias=1.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.d = d
        self.lags = list(lags)
        # See src/gf256_markov_lm_v6_1.py GFLayer.__init__ for why this
        # exists: the identity/one-hot init bonus creates a ~7.7-logit
        # margin that Adam cannot cross within a realistic step budget,
        # which freezes the exported integer tables and leaves the hard
        # export board-blind. 1.0 = historical, 0.0 = unbiased.
        # The RNG draw order is unchanged, so coef_set is unaffected.
        self.table_bias = float(table_bias)

        coef_set = [0, 1]
        while len(coef_set) < n_coef:
            c = int(torch.randint(2, 256, (1,), generator=gen))
            if c not in coef_set: coef_set.append(c)
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

        self.layers = nn.ModuleList([
            GFLayer(d, fan_in, self.lags, coef_set, gen,
                    table_bias=self.table_bias) for _ in range(layers)
        ])

        self.readout = nn.ParameterList([
            nn.Parameter(torch.zeros(d, 256, 256)) for _ in range(layers + 1)
        ])
        if bigram_logp is not None:
            with torch.no_grad():
                self.readout[0][0] = bigram_logp

        self.readout_bias = nn.Parameter(torch.zeros(256))
        self.ste = False

    def set_ste(self, on):
        self.ste = on
        for L in self.layers: L.ste = on

    def forward(self, tokens):
        h = ste_softmax(self.emb_logits, self.ste)[tokens]
        logits = torch.einsum("btdi,div->btv", h, self.readout[0])
        for layer, R in zip(self.layers, self.readout[1:]):
            if self.training and self.use_ckpt:
                h = torch.utils.checkpoint.checkpoint(layer, h, self.perm_mats, use_reentrant=False)
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
    def export_hard(self, meta: dict | None = None):
        """Export the integer tables. `meta` is stored verbatim so a resumed
        run can recover the step counter and keep its schedules aligned."""
        out = {
            "emb": self.emb_logits.argmax(-1),
            "coef": [torch.tensor(self.coef_set, device=L.coef_logits.device)[L.coef_logits.argmax(-1)] for L in self.layers],
            "sbox": [L.sbox_logits.argmax(-1) for L in self.layers],
            "wiring": [(L.ch, L.lag) for L in self.layers],
            "readout": [R.detach() for R in self.readout],
            "bias": self.readout_bias.detach(),
            "lags": self.lags,
        }
        if meta:
            out["meta"] = meta
        return out

    def import_hard(self, hard: dict) -> None:
        """Inverse of export_hard -- re-derive soft logits from the integer
        tables so the model resumes at exactly the accuracy point the
        checkpoint was exported at.

        Every integer table in the payload is the argmax of one soft-logit
        tensor, so the inverse is a one-hot re-encode at +10.0 (well past the
        +8.0 init bias used by the trainer, so the STE path is unambiguous).
        ``coef`` needs care: its values are GF(256) *elements*, and the argmax
        slot must point at the coef_set position whose *value* equals that
        element, so we invert coef_set once and index through it.
        """
        if list(hard["lags"]) != list(self.lags):
            raise ValueError(
                f"import_hard: lag mismatch. checkpoint lags={hard['lags']} "
                f"model lags={self.lags}"
            )
        device = self.emb_logits.device
        d = self.d
        K = len(self.coef_set)
        coef_index_of = {int(self.coef_set[i]): i for i in range(K)}

        # --- emb_logits: (256, d, 256) indexed [token, j, glyph].
        emb_arg = hard["emb"].to(device).long()
        if emb_arg.shape != (256, d):
            raise ValueError(f"emb shape {tuple(emb_arg.shape)} != (256,{d})")
        new_emb = torch.zeros_like(self.emb_logits)
        ar256 = torch.arange(256, device=device)
        ar_d = torch.arange(d, device=device)
        new_emb[ar256.unsqueeze(1), ar_d.unsqueeze(0), emb_arg] = 10.0
        self.emb_logits.data.copy_(new_emb)

        # --- coef_logits: (d, fan_in, K), argmax slot = coef_set index.
        for i, (L, idx) in enumerate(zip(self.layers, hard["coef"])):
            idx_t = idx.to(device).long()
            fan_in = idx_t.shape[1]
            if L.coef_logits.shape[1] != fan_in:
                raise ValueError(
                    f"import_hard: layer {i} fan_in {L.coef_logits.shape[1]} "
                    f"!= payload {fan_in}"
                )
            pos = torch.empty_like(idx_t)
            flat_pos, flat_val = pos.view(-1), idx_t.view(-1).tolist()
            for k, v in enumerate(flat_val):
                if int(v) not in coef_index_of:
                    raise ValueError(
                        f"import_hard: layer {i} coef value {v} not in "
                        f"model coef_set {list(self.coef_set)}"
                    )
                flat_pos[k] = coef_index_of[int(v)]
            new_cl = torch.zeros(d, fan_in, K, device=device)
            new_cl.scatter_(2, pos.unsqueeze(2), 10.0)
            L.coef_logits.data.copy_(new_cl)

        # --- sbox_logits: (d, 256, 256) -- a full table per dimension, where
            # argmax over the last axis for input glyph x yields the output glyph.
            # The payload stores that table directly as (d, 256).
            for i, (L, sbox_idx) in enumerate(zip(self.layers, hard["sbox"])):
                sbox_t = sbox_idx.to(device).long()
                if sbox_t.shape != (d, 256):
                    raise ValueError(
                        f"import_hard: layer {i} sbox shape {tuple(sbox_t.shape)} "
                        f"!= ({d}, 256)"
                    )
                new_sl = torch.zeros_like(L.sbox_logits)
                new_sl[ar_d.unsqueeze(1), ar256.unsqueeze(0), sbox_t] = 10.0
                L.sbox_logits.data.copy_(new_sl)

        # --- wiring (ch, lag): registered buffers on each layer.
        for i, (L, wiring) in enumerate(zip(self.layers, hard["wiring"])):
            ch_t, lg_t = wiring
            if ch_t.shape != L.ch.shape or lg_t.shape != L.lag.shape:
                raise ValueError(
                    f"import_hard: layer {i} wiring shape mismatch "
                    f"ch={tuple(ch_t.shape)} vs {tuple(L.ch.shape)}, "
                    f"lag={tuple(lg_t.shape)} vs {tuple(L.lag.shape)}"
                )
            L.ch = ch_t.to(device).long()
            L.lag = lg_t.to(device).long()
            L.lag_vals = torch.tensor(
                self.lags, device=device, dtype=torch.long
            )[L.lag]

        # --- readout / bias: exact copies.
        if len(hard["readout"]) != len(self.readout):
            raise ValueError(
                f"import_hard: readout count {len(hard['readout'])} != "
                f"{len(self.readout)}"
            )
        for i, (R, R_new) in enumerate(zip(self.readout, hard["readout"])):
            R_new = R_new.to(R.device)
            if R.shape != R_new.shape:
                raise ValueError(
                    f"import_hard: readout[{i}] shape "
                    f"{tuple(R.shape)} != {tuple(R_new.shape)}"
                )
            R.data.copy_(R_new)
        bias_new = hard["bias"].to(self.readout_bias.device)
        if bias_new.shape != self.readout_bias.shape:
            raise ValueError(
                f"import_hard: bias shape {tuple(bias_new.shape)} != "
                f"{tuple(self.readout_bias.shape)}"
            )
        self.readout_bias.data.copy_(bias_new)
    @torch.no_grad()
    def hard_forward(self, tokens, hard):
        exp_t, log_t = torch.tensor(EXP), torch.tensor(LOG)
        lags = torch.tensor(hard["lags"])
        B, T, d = tokens.shape[0], tokens.shape[1], self.d
        h = hard["emb"][tokens]
        ar = torch.arange(d)
        logits = hard["readout"][0][ar, h].sum(2)
        for coef, sbox, (ch, lg), R in zip(hard["coef"], hard["sbox"], hard["wiring"], hard["readout"][1:]):
            acc = torch.zeros(B, T, d, dtype=torch.long)
            for f in range(coef.shape[1]):
                out = torch.zeros(B, T, d, dtype=torch.long)
                for j in range(d):
                    l = int(lags[lg[j, f]])
                    # V7 can have lags up to 256, and callers may hand us a T
                    # shorter than that (short prompts, ragged eval batches). Then
                    # `src[:, :-l]` is (B, 0) and the cat below inflates the row
                    # to width l, which no longer matches `out[:, :, j]` of width
                    # T. Clamp so the leading pad never exceeds the actual T.
                    if l > T:
                        l = T - 1 if T > 1 else 0
                    src = h[:, :, ch[j, f]]
                    if l: src = torch.cat([torch.zeros(B, l, dtype=torch.long), src[:, :-l]], 1)
                    c = int(coef[j, f])
                    if c == 0: prod = torch.zeros_like(src)
                    else: prod = torch.where(src == 0, torch.zeros_like(src), exp_t[log_t[src] + LOG[c]])
                    out[:, :, j] = prod
                acc = acc ^ out
            h = sbox[ar, acc]
            logits = logits + R[ar, h].sum(2)
        return logits + hard["bias"]

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_data(path):
    if path == "tinyshakespeare.txt" and not os.path.exists(path):
        print("downloading Tiny Shakespeare ...")
        urllib.request.urlretrieve(URL, path)
    elif not os.path.exists(path):
        raise FileNotFoundError(f"Corpus not found at {path}. Please provide a valid .txt file.")
        
    raw = open(path, "rb").read()
    data = torch.tensor(list(raw), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]

def bigram_logp(data):
    counts = torch.ones(256, 256)
    counts.index_put_((data[:-1], data[1:]), torch.ones(len(data) - 1), accumulate=True)
    return (counts / counts.sum(1, keepdim=True)).log()

def get_batch(data, B, T, device):
    ix = torch.randint(0, len(data) - T - 1, (B,))
    x = torch.stack([data[i:i + T] for i in ix])
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def evaluate(model, seqs, device, chunk=64):
    was_ste = model.ste
    model.eval()
    model.set_ste(False)
    x, y = seqs[:, :-1], seqs[:, 1:]
    soft_nll, hard_nll = 0.0, 0.0
    for k in range(0, len(x), chunk):
        xb, yb = x[k:k + chunk], y[k:k + chunk]
        lo = model(xb.to(device)).float().cpu()
        soft_nll += F.cross_entropy(lo.reshape(-1, 256), yb.reshape(-1), reduction="sum").item()
    
    hard = model.export_hard()
    
    # FIXED: Safely move tensors to CPU, ignoring standard Python lists like "lags"
    hard_cpu = {}
    for k, v in hard.items():
        if k == "lags" or not isinstance(v, (list, torch.Tensor)):
            hard_cpu[k] = v  # Keep python lists of ints as-is
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
        hard_nll += F.cross_entropy(lo.reshape(-1, 256), yb.reshape(-1), reduction="sum").item()
    
    n = y.numel() * math.log(2)
    model.set_ste(was_ste)
    model.train()
    return soft_nll / n, hard_nll / n

def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train, val = load_data(args.data)
    model = GFMarkovLM(d=args.d, layers=args.layers, fan_in=args.fan_in, bigram_logp=bigram_logp(train), seed=seed).to(args.device)
    model.use_ckpt = not args.no_ckpt
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.2f}M | d={args.d} layers={args.layers} seed={seed}")

    L = args.ctx + 1
    all_val = val[: (len(val) // L) * L].view(-1, L)
    gsel = torch.Generator().manual_seed(123)
    quick_val = all_val[torch.randperm(len(all_val), generator=gsel)[:args.eval_seqs]]

    t0 = time.time()

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            soft, hard = evaluate(model, quick_val, args.device, chunk=args.eval_chunk_size)
            ent, _ = model.regularisers()
            print(f"step {step:5d} | soft BPC {soft:.3f} | hard BPC {hard:.3f} | entropy {ent.item():.3f} | {time.time()-t0:.0f}s")

        if step == args.steps: break
            
        if step == args.hard_start:
            print(f"\n--- Step {step}: Starting Gentle STE phase (aligning readouts) ---")
            for pg in opt.param_groups: pg['lr'] *= 0.2

        x, y = get_batch(train, args.batch, args.ctx, args.device)

        model.set_ste(False)
        logits_soft = model(x)
        ce_soft = F.cross_entropy(logits_soft.reshape(-1, 256), y.reshape(-1))

        hard_ce = torch.tensor(0.0, device=args.device)
        hard_w = 0.0
        if step >= args.hard_start and args.hard_w > 0:
            hard_w = args.hard_w * min(1.0, (step - args.hard_start + 1) / max(1, args.hard_ramp))
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

    soft, hard = evaluate(model, all_val, args.device, chunk=args.eval_chunk_size)
    print(f"\nFINAL on full val: soft {soft:.3f} | HARD {hard:.3f}")
    torch.save(model.cpu().export_hard(), "gf256_hard_model_v200MB.pt")
    print("saved hardened integer program -> gf256_hard_model_v200MB.pt")

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="tinyshakespeare.txt", help="Path to your expert corpus .txt file")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--fan_in", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=256)
    
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    
    ap.add_argument("--ent_max", type=float, default=2.0)
    ap.add_argument("--ent_warmup", type=int, default=2000) 
    ap.add_argument("--perm_w", type=float, default=1.0)
    
    ap.add_argument("--hard_start", type=int, default=1500)
    ap.add_argument("--hard_ramp", type=int, default=1000) #10000
    ap.add_argument("--hard_w", type=float, default=0.5)
    
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--eval_seqs", type=int, default=128)
    ap.add_argument("--eval_chunk_size", type=int, default=8)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap

if __name__ == "__main__":
    print("cuda" if torch.cuda.is_available() else "cpu")
    main(build_parser().parse_args())

###
# PS <user>\> python gf256_markov_lm.py --seed 64 --steps 12000 --ctx 256 --eval_every 400 --eval_seqs 64 --eval_chunk_size 4 --batch 4 --no_ckpt
#params 58.75M | d=64 layers=6 seed=64
#step     0 | soft BPC 3.596 | hard BPC 3.595 | entropy 0.798 | 2s
#step   400 | soft BPC 2.982 | hard BPC 6.352 | entropy 0.302 | 137s
#step   800 | soft BPC 2.949 | hard BPC 5.944 | entropy 0.176 | 276s
#step  1200 | soft BPC 2.922 | hard BPC 5.635 | entropy 0.123 | 414s#

#--- Step 1500: Starting Gentle STE phase (aligning readouts) ---
#step  1600 | soft BPC 2.705 | hard BPC 4.632 | entropy 0.099 | 579s
#step  2000 | soft BPC 2.659 | hard BPC 3.696 | entropy 0.095 | 839s
#step  2400 | soft BPC 2.656 | hard BPC 3.656 | entropy 0.093 | 1101s
#step  2800 | soft BPC 2.649 | hard BPC 3.436 | entropy 0.093 | 1362s
#step  3200 | soft BPC 2.635 | hard BPC 3.473 | entropy 0.093 | 1623s
#step  3600 | soft BPC 2.636 | hard BPC 3.487 | entropy 0.093 | 1883s
#step  4000 | soft BPC 2.637 | hard BPC 3.503 | entropy 0.093 | 2144s
#step  4400 | soft BPC 2.629 | hard BPC 3.395 | entropy 0.094 | 2404s
#step  4800 | soft BPC 2.613 | hard BPC 3.426 | entropy 0.094 | 2665s
#step  5200 | soft BPC 2.612 | hard BPC 3.414 | entropy 0.094 | 2928s
#step  5600 | soft BPC 2.605 | hard BPC 3.257 | entropy 0.094 | 3196s
#step  6000 | soft BPC 2.607 | hard BPC 3.420 | entropy 0.094 | 3463s
#step  6400 | soft BPC 2.606 | hard BPC 3.255 | entropy 0.094 | 3726s
#step  6800 | soft BPC 2.598 | hard BPC 3.207 | entropy 0.094 | 3989s
#step  7200 | soft BPC 2.591 | hard BPC 3.145 | entropy 0.094 | 4248s
#step  7600 | soft BPC 2.602 | hard BPC 3.096 | entropy 0.093 | 4507s
#step  8000 | soft BPC 2.590 | hard BPC 3.089 | entropy 0.093 | 4767s
###


###
# PS <user>> python gf256_markov_lm.py --seed 64 --steps 12000 --ctx 256 --eval_every 400 --eval_seqs 64 --eval_chunk_size 4 --batch 4 --no_ckpt --data smollm_subset.txt
# cuda
# params 58.75M | d=64 layers=6 seed=64
# step     0 | soft BPC 3.639 | hard BPC 3.639 | entropy 0.798 | 2s
# step   400 | soft BPC 3.045 | hard BPC 7.070 | entropy 0.324 | 135s
# step   800 | soft BPC 2.938 | hard BPC 5.737 | entropy 0.194 | 270s
# step  1200 | soft BPC 2.829 | hard BPC 5.602 | entropy 0.132 | 404s

# --- Step 1500: Starting Gentle STE phase (aligning readouts) ---
# step  1600 | soft BPC 2.616 | hard BPC 4.670 | entropy 0.105 | 568s
# step  2000 | soft BPC 2.598 | hard BPC 4.040 | entropy 0.101 | 827s
# step  2400 | soft BPC 2.621 | hard BPC 3.444 | entropy 0.100 | 1086s
# step  2800 | soft BPC 2.650 | hard BPC 3.313 | entropy 0.102 | 1344s
# step  3200 | soft BPC 2.668 | hard BPC 3.315 | entropy 0.103 | 1603s
# step  3600 | soft BPC 2.683 | hard BPC 3.349 | entropy 0.105 | 1861s
# step  4000 | soft BPC 2.705 | hard BPC 3.253 | entropy 0.107 | 2120s
# step  4400 | soft BPC 2.714 | hard BPC 3.414 | entropy 0.108 | 2379s
# step  4800 | soft BPC 2.719 | hard BPC 3.251 | entropy 0.110 | 2638s
# step  5200 | soft BPC 2.723 | hard BPC 3.357 | entropy 0.111 | 2898s
# step  5600 | soft BPC 2.723 | hard BPC 3.092 | entropy 0.112 | 3158s
# step  6000 | soft BPC 2.718 | hard BPC 3.145 | entropy 0.114 | 3419s
# step  6400 | soft BPC 2.710 | hard BPC 3.191 | entropy 0.115 | 3679s
# step  6800 | soft BPC 2.703 | hard BPC 3.068 | entropy 0.117 | 3940s
# step  7200 | soft BPC 2.698 | hard BPC 2.988 | entropy 0.118 | 4200s
# step  7600 | soft BPC 2.688 | hard BPC 2.954 | entropy 0.120 | 4460s
# step  8000 | soft BPC 2.689 | hard BPC 3.155 | entropy 0.122 | 4720s
# step  8400 | soft BPC 2.678 | hard BPC 3.054 | entropy 0.124 | 4979s
# step  8800 | soft BPC 2.671 | hard BPC 2.997 | entropy 0.125 | 5239s
# step  9200 | soft BPC 2.665 | hard BPC 3.016 | entropy 0.127 | 5499s
# step  9600 | soft BPC 2.666 | hard BPC 3.000 | entropy 0.128 | 5758s
# step 10000 | soft BPC 2.662 | hard BPC 2.918 | entropy 0.129 | 6018s
# step 10400 | soft BPC 2.658 | hard BPC 3.023 | entropy 0.130 | 6278s
# step 10800 | soft BPC 2.654 | hard BPC 3.091 | entropy 0.131 | 6538s
# step 11200 | soft BPC 2.654 | hard BPC 3.129 | entropy 0.132 | 6797s
# step 11600 | soft BPC 2.656 | hard BPC 3.077 | entropy 0.133 | 7056s
# step 12000 | soft BPC 2.658 | hard BPC 3.121 | entropy 0.134 | 7315s

# FINAL on full val: soft 2.708 | HARD 3.211
# saved hardened integer program -> gf256_hard_model_v2.pt