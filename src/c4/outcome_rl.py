"""
outcome_rl.py -- REINFORCE (outcome-based policy gradient) for the GF(256)
Markov LM on Connect Four.

Why this file exists
--------------------
`train.py` optimises next-byte cross-entropy against a fixed corpus. That
converges to "predict what the corpus said", which is imitation of the
heuristic that generated it -- it cannot exceed that teacher. The objective
here is different: the only loss is the *game result*.

    reward(ply) = +1 if the player who made that ply won
                = -1 if that player lost
                =  0 on a draw
    loss        = -E[ reward * log pi_θ(chosen_move | prompt) ]

No KL anchor to a reference policy, no warmup. The policy that samples a move
is the same policy the gradient is taken on (same temperature), so the
estimator is unbiased.

Self-play opponents, mixed per gradient step:
  * 'self'   -- both sides are the current policy (both sides get gradient)
  * 'heur'   -- model vs heuristic_policy (only the model's plies are trained)
  * 'pool'   -- model vs a frozen snapshot of itself from k steps ago
                (the step-0 warm-start model is always in the pool, which
                 makes "am I beating my past self?" a stable benchmark)

Usage
-----
    python -m src.c4.outcome_rl \
        --init runs/c4/v7mixed/final_float.pt \
        --run-dir runs/c4/rl1 \
        --steps 400 --games 48 --lr 1e-4

All prompts are LEFT-padded to the batch max and the policy is read at
position -1, which is the true last prompt byte. This matches the server's
inference path (`Agent.hard_column_logits`, which left-pads to HARD_MIN_LEN).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .agent import PAD_BYTE, prompt_tokens, hard_to_cpu
from .board import Board, Outcome
from .opponents import heuristic_policy
from .transcript import COL_SLICE
from ..gf256_markov_lm_v6 import GFMarkovLM

# Architecture of the warm-start checkpoint (runs/c4/v7mixed/final_float.pt).
ARCH = dict(d=32, layers=3, fan_in=3, seed=0)

# Token budget for gradient micro-batches (prompt positions per forward).
GRAD_TOKENS = 4096
# Token budget for generation forwards (no autograd, so much cheaper).
GEN_TOKENS = 8192


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------

def build_model(state: dict, device, step: int = 0) -> GFMarkovLM:
    """Rebuild the model and load `state`. Uses `ARCH` (seed=0) which must
    match the checkpoint's meta -- coef_set/perm_mats are seed-derived."""
    m = GFMarkovLM(**ARCH).to(device)
    m.load_state_dict(state)
    m.meta = {"step": step}
    return m


def tok(prompt) -> list[int]:
    """prompt_tokens() returns a tensor; records hold plain int lists."""
    return [int(t) for t in prompt]


def col_logits(model: GFMarkovLM, prompts, device, grad: bool):
    """(B, 7) column logits, each row at its own true last position.

    This byte-level Markov has NO padding mask, so padding bytes are real
    inputs to the state update -- a padded row does NOT equal the solo
    forward. (`Agent.batch_column_logits` happens to be safe only because its
    callers feed equal-length prompts.) So we group the batch by exact
    prompt length and forward each group unpadded. Connect Four prompts
    have quantized lengths, so there are only ~40 distinct groups and each
    one is a genuine full-width batch.

    Rows are returned in the caller's original order.
    """
    if not prompts:
        return torch.zeros((0, 7), device=device)

    # group indices by exact length, keeping insertion order for determinism
    groups: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)

    out = torch.zeros((len(prompts), 7), device=device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    for length in sorted(groups):
        idxs = groups[length]
        # every row in this group has the same length -> no padding needed
        rows = [list(prompts[i]) for i in idxs]
        assert len({len(r) for r in rows}) == 1, "length group must be uniform"
        x = torch.tensor(rows, dtype=torch.long, device=device)   # (g, T)
        with ctx:
            logits = model(x)                                    # (g, T, 256)
        out[idxs] = logits[:, -1, COL_SLICE].float()             # (g, 7)
    return out


def chunk_by_tokens(prompts, budget: int):
    """Split indices into chunks bounded by (chunk_size * longest_prompt).

    `prompts` is a list of prompt sequences; the return value is a list of
    index lists into `prompts`, in the original order.
    """
    out, cur, cur_max = [], [], 0
    for i in range(len(prompts)):
        m = max(cur_max, len(prompts[i]))
        if cur and m * (len(cur) + 1) > budget:
            out.append(cur)
            cur, cur_max = [i], len(prompts[i])
        else:
            cur.append(i)
            cur_max = m
    if cur:
        out.append(cur)
    return out


def masked_logprob(logits: torch.Tensor, legals, temp: float) -> torch.Tensor:
    """log softmax over legal columns only.

    `logits` is (B, 7). `legals` is a list of B lists of legal column ints
    (or a single list of ints when B == 1).
    """
    if legals and isinstance(legals[0], int):
        legals = [legals]                          # a bare list of ints (B == 1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for r, cols in enumerate(legals):
        idx = torch.tensor(cols, dtype=torch.long, device=logits.device)
        mask[r, idx] = True
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.log_softmax(logits / max(temp, 1e-6), dim=-1)


def sample_col(lp_row, rng: random.Random) -> int:
    """Inverse-CDF sample from a log-prob row."""
    cdf, acc = [], 0.0
    for v in lp_row:
        acc += math.exp(v)
        cdf.append(acc)
    u = rng.random() * acc
    for c, v in enumerate(cdf):
        if u <= v:
            return c
    return len(cdf) - 1


# ---------------------------------------------------------------------------
# one self-play batch
# ---------------------------------------------------------------------------
# A record is (prompt_list, legal_cols, col_played, side_to_move).

def _policy_choose(model, prompt, legal, temp, device, rng):
    lg = col_logits(model, [prompt], device, grad=False)
    lp = masked_logprob(lg[0], legal, temp)
    lp_cpu = lp.detach().cpu()
    return sample_col(lp_cpu, rng), lp_cpu


def play_batch(model, device, temp, rng, modes, pool, pool_choices,
               greedy=False) -> tuple[list[dict], float]:
    """Play `len(modes)` games to termination, recording every policy ply.

    `modes[i]` is 'self' | 'heur' | 'pool'. For 'self', both sides are the
    current policy and both sides' plies are recorded. For 'heur' / 'pool',
    the model plays the side given by `model_is_p1[i]` and only its plies
    are recorded.
    """
    games = []
    for i, mode in enumerate(modes):
        model_is_p1 = (i % 2 == 0)
        if mode == "self":
            model_is_p1 = None          # both sides are the policy
        games.append({
            "board": Board(),
            "moves": [],
            "recs": [],
            "mode": mode,
            "model_is_p1": model_is_p1,
            "active": True,
            "outcome": None,
        })

    ent_sum, ent_n = 0.0, 0

    while any(g["active"] for g in games):
        # Which games need a model / opponent decision right now?
        model_side, opp_side = [], []
        for i, g in enumerate(games):
            if not g["active"]:
                continue
            b: Board = g["board"]
            # CANONICAL terminal test: a game is over when someone won or
            # no legal action remains. A bare is_full() here would inherit
            # the historical one-bit bug (false draw with 36 empty cells).
            if b.is_win() != 0 or not b.valid_actions():
                g["outcome"] = Outcome.DRAW if b.is_win() == 0 else b.outcome()
                g["active"] = False
                continue
            if g["mode"] == "self":
                model_side.append(i)
            elif (b.turn == 1) == g["model_is_p1"]:
                model_side.append(i)
            else:
                opp_side.append(i)

        # --- current policy moves (recorded, differentiable later) --------
        if model_side:
            prompts = [tok(prompt_tokens(games[i]["moves"])) for i in model_side]
            legals = [games[i]["board"].valid_actions() for i in model_side]
            for chunk in chunk_by_tokens(prompts, GEN_TOKENS):
                sel = [model_side[c] for c in chunk]
                lg = col_logits(model, [prompts[c] for c in chunk], device, grad=False)
                lps = masked_logprob(lg, [legals[c] for c in chunk], temp)
                ent_sum += float(-(lps.exp() * lps.clamp_min(-1e9)).sum(-1).mean())
                ent_n += len(chunk)
                for k, gi in enumerate(sel):
                    g = games[gi]
                    b: Board = g["board"]
                    chunk_prompts = [prompts[c] for c in chunk]
                    chunk_legals  = [legals[c]  for c in chunk]
                    if greedy:
                        col = int(lps[k].argmax())
                    else:
                        col = sample_col(lps[k].cpu(), rng)
                    g["recs"].append((chunk_prompts[k], chunk_legals[k], col, b.turn))
                    g["moves"].append(col)
                    b.apply_move(col)
                    o = b.outcome()
                    if o != Outcome.ONGOING:
                        g["outcome"] = o
                        g["active"] = False

        # --- opponent moves (no gradient) --------------------------------
        for gi in opp_side:
            g = games[gi]
            b: Board = g["board"]
            if g["mode"] == "heur":
                col = heuristic_policy(b, g["moves"])
            else:
                snap = pool[pool_choices[gi]]
                col, _ = _policy_choose(snap, tok(prompt_tokens(g["moves"])),
                                        b.valid_actions(), temp, device, rng)
            g["moves"].append(col)
            b.apply_move(col)
            o = b.outcome()
            if o != Outcome.ONGOING:
                g["outcome"] = o
                g["active"] = False

    return games, ent_sum / max(ent_n, 1)


def assign_rewards(games) -> tuple[list, float]:
    """reward = outcome from the perspective of the mover at each recorded ply."""
    recs, all_r = [], []
    for g in games:
        o = g["outcome"]
        if o is None:
            o = Outcome.DRAW
        for (prompt, legal, col, side) in g["recs"]:
            if o == Outcome.WIN_P1:
                r = 1.0 if side == 1 else -1.0
            elif o == Outcome.WIN_P2:
                r = -1.0 if side == 1 else 1.0
            else:
                r = 0.0
            recs.append((prompt, legal, col, r))
            all_r.append(r)
    return recs, (sum(all_r) / max(len(all_r), 1))


# ---------------------------------------------------------------------------
# policy gradient
# ---------------------------------------------------------------------------

def policy_gradient(model, recs, temp, device, baseline, ent_w):
    """Sum over plies of -R * log pi(a|s), micro-batched by token budget.

    `baseline` is subtracted from every reward; it is a scalar constant, so
    it does not change the gradient direction, only the variance.
    """
    prompts = [r[0] for r in recs]
    chunks = chunk_by_tokens(prompts, GRAD_TOKENS)
    total_n = 0
    loss_sum = torch.zeros((), device=device)
    ent_sum = torch.zeros((), device=device)
    for ch in chunks:
        prompts = [recs[i][0] for i in ch]
        legals = [recs[i][1] for i in ch]
        cols = torch.tensor([recs[i][2] for i in ch], dtype=torch.long, device=device)
        rews = torch.tensor([recs[i][3] for i in ch], dtype=torch.float32, device=device)

        lg = col_logits(model, prompts, device, grad=True)          # (B,7)
        lp = masked_logprob(lg, legals, temp)                        # (B,7)
        logp = lp.gather(1, cols.unsqueeze(1)).squeeze(1)            # (B,)
        p = lp.exp()
        ent = -(p * lp.clamp_min(-1e9)).sum(-1)                      # (B,)
        b = torch.as_tensor(baseline, dtype=torch.float32, device=device)
        loss = -((rews - b) * logp).sum()
        loss = loss + ent_w * ent.sum()
        loss_sum = loss_sum + loss.detach()
        ent_sum = ent_sum + ent.detach().sum()
        total_n += len(ch)
        loss.backward()                              # grads ACCUMULATE across chunks
    return float(loss_sum) / max(total_n, 1), float(ent_sum) / max(total_n, 1)


# ---------------------------------------------------------------------------
# evaluation -- the honest metric: W/L/D as P1 vs the 1-ply heuristic
# ---------------------------------------------------------------------------

def evaluate(model, device, heuristic_games: int, seed: int) -> dict:
    rng = random.Random(seed)
    model.eval()
    modes = ["heur"] * heuristic_games
    games, _ = play_batch(model, device, 1.0, rng, modes, None, None, greedy=True)
    w = l = d = 0
    for g in games:
        o = g["outcome"]
        if o == Outcome.DRAW:
            d += 1
        elif (o == Outcome.WIN_P1) == g["model_is_p1"]:
            w += 1
        else:
            l += 1
    model.train()
    return {"w": w, "l": l, "d": d,
            "score": (w - l) / max(heuristic_games, 1)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="float checkpoint to warm-start from")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--games", type=int, default=48, help="games per gradient step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--ent-w", type=float, default=0.005,
                    help="entropy bonus coefficient (collapse guard, not a KL anchor)")
    ap.add_argument("--baseline", default="running",
                    choices=["running", "none", "batch"])
    ap.add_argument("--mix-self", type=float, default=0.5)
    ap.add_argument("--mix-heur", type=float, default=0.5)
    ap.add_argument("--pool-size", type=int, default=4)
    ap.add_argument("--pool-every", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-games", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0xC4C4)
    ap.add_argument("--max-ply", type=int, default=60)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # weights_only=True is safe here: the payload is tensors + a plain dict of
    # str/int/float (no pickled classes), and it blocks arbitrary unpickling.
    ck = torch.load(args.init, map_location="cpu", weights_only=True)
    state = ck["state"] if "state" in ck else ck
    model = build_model(state, device)
    model.train()
    print(f"[rl] warm-start {args.init} on {device}", flush=True)

    pool: list[GFMarkovLM] = []          # frozen snapshots, all on device
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_base, base_n = 0.0, 0
    best_score, hist = -1e9, []
    t0 = time.time()
    print(f"[rl] step 0 baseline eval ...", flush=True)
    ev = evaluate(model, device, args.eval_games, args.seed)
    print(f"[rl] step 0 (warm-start): W{ev['w']} L{ev['l']} D{ev['d']} "
          f"score={ev['score']:+.3f}", flush=True)
    hist.append({"step": 0, **ev, "elapsed_s": 0.0})

    metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    for step in range(1, args.steps + 1):
        # --- opponent mix ---------------------------------------------
        modes, pool_choices = [], []
        for i in range(args.games):
            u = rng.random()
            if u < args.mix_self:
                modes.append("self")
                pool_choices.append(None)
            elif u < args.mix_self + args.mix_heur or not pool:
                modes.append("heur")
                pool_choices.append(None)
            else:
                modes.append("pool")
                pool_choices.append(rng.randrange(len(pool)))
        games, gen_ent = play_batch(model, device, args.temp, rng, modes, pool,
                                    pool_choices, greedy=False)
        recs, mean_r = assign_rewards(games)

        # --- baseline --------------------------------------------------
        if args.baseline == "running":
            run_base = 0.9 * run_base + 0.1 * mean_r if base_n else mean_r
            base_n += 1
            b = run_base
        elif args.baseline == "batch":
            b = mean_r
        else:
            b = 0.0

        # --- gradient --------------------------------------------------
        loss, ent = policy_gradient(model, recs, args.temp, device, b, args.ent_w)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        # --- snapshot pool ---------------------------------------------
        if pool and step % args.pool_every == 0 and len(pool) < args.pool_size:
            pool.pop(0)
        if step % args.pool_every == 0:
            snap = build_model({k: v.detach().clone() for k, v in
                                model.state_dict().items()}, device)
            snap.eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            pool.append(snap)

        row = {"step": step, "loss": loss, "mean_reward": mean_r,
               "baseline": b, "gen_entropy": gen_ent, "pg_entropy": ent,
               "plies": len(recs), "elapsed_s": round(time.time() - t0, 1)}

        # --- periodic eval ---------------------------------------------
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, device, args.eval_games, args.seed)
            row.update(ev)
            hist.append({"step": step, **ev, "elapsed_s": row["elapsed_s"]})
            if ev["score"] > best_score:
                best_score = ev["score"]
                torch.save({"state": model.state_dict(),
                            "meta": {**ARCH, "step": step, "eval": ev}},
                           run_dir / "best.pt")
            print(f"[rl] step {step:4d} loss={loss:+.4f} R={mean_r:+.3f} "
                  f"ent={ent:.3f} | W{ev['w']} L{ev['l']} D{ev['d']} "
                  f"score={ev['score']:+.3f} best={best_score:+.3f} "
                  f"({row['elapsed_s']:.0f}s)", flush=True)
        else:
            print(f"[rl] step {step:4d} loss={loss:+.4f} R={mean_r:+.3f} "
                  f"ent={ent:.3f} plies={len(recs)} ({row['elapsed_s']:.0f}s)",
                  flush=True)
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()

        torch.save({"state": model.state_dict(), "meta": {**ARCH, "step": step}},
                   run_dir / "last.pt")

    # --- final artefacts ------------------------------------------------
    torch.save({"state": model.state_dict(), "meta": {**ARCH, "step": args.steps}},
               run_dir / "final_float.pt")
    hard = hard_to_cpu(model.export_hard())
    torch.save(hard, run_dir / "hard_model_final.pt")
    json.dump({"steps": args.steps, "games_per_step": args.games, "lr": args.lr,
               "temp": args.temp, "ent_w": args.ent_w, "baseline": args.baseline,
               "init": args.init, "best_score": best_score, "hist": hist},
              open(run_dir / "final.json", "w"), indent=2)
    metrics.close()
    print(f"[rl] done. best heuristic score = {best_score:+.3f} "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()