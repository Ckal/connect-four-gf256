"""Build an augmented Connect-Four corpus and append it to corpus.txt.

Three sources, all encoded with `corpus_aug.game_dense` (stride 10, one
newline per GAME) so `splitlines()` yields whole games and `build_batch`
stops padding 95% of every window with newlines:

  1. THREAT   -- single forced decisions (win_now / block_now / make_threat /
                 safe) from `threat_corpus.classify`. Self-play trajectories
                 dilute the one move that matters; these do not.
  2. LADDER   -- full games where one side plays `minimax_policy` (depth-3
                 alpha-beta) and the other plays a weaker mixture. The
                 winning side's trajectory is what teaches real play.
  3. MODEL    -- the current checkpoint playing both sides, so its own
                 on-distribution wins are reinforced.

Nothing is truncated: the existing corpus is read, rewritten in dense form
(legacy bytes are converted, not dropped), and the new material is appended.

Usage:
    python -m src.c4.augment_corpus --run-dir runs/c4/v7mixed_v9 \
        --threat 60000 --ladder 4000 --model 2000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

from . import P1, P2, COLS, ROWS
from .board import Board, is_terminal
from .corpus import Game, winners_only
from .corpus_aug import (GAME_DENSE, game_dense, moves_of_dense,
                         plies_of_dense, validate_file)
from .opponents import (heuristic_policy, lookahead_policy, minimax_policy,
                        random_policy)
from .threat_corpus import classify
from .transcript import (COL_BYTE_OFFSET, NO_LAST, TOK_TURN_P1, PLY_LEN)
from .strong_solver import strong_policy, clear_tt as _clear_solver_tt


# ---------------------------------------------------------------------------
# legacy -> dense conversion
# ---------------------------------------------------------------------------

def _is_ply0(rec: bytes) -> bool:
    """True if `rec` is a ply-0 record (all heights 0, last=NO_LAST, turn=X)."""
    return (all(rec[c] == 0x30 for c in range(COLS))
            and rec[COLS] == NO_LAST
            and rec[COLS + 1] == TOK_TURN_P1)


def regroup_legacy(raw: bytes) -> List[List[bytes]]:
    """Split a legacy (interleaved-newline) corpus back into per-ply records
    and regroup them into games on the ply-0 marker."""
    recs = [ln for ln in raw.splitlines() if ln]
    games: List[List[bytes]] = []
    cur: List[bytes] = []
    for r in recs:
        if len(r) != PLY_LEN:
            continue                      # not a ply record -- drop silently
        if _is_ply0(r) and cur:
            games.append(cur)
            cur = []
        cur.append(r)
    if cur:
        games.append(cur)
    return games


def legacy_to_dense(raw: bytes) -> tuple[bytes, dict]:
    """Convert a legacy corpus blob to dense form. Returns (blob, stats)."""
    games = regroup_legacy(raw)
    kept = 0
    dropped = 0
    out: List[bytes] = []
    plies_total = 0
    for g in games:
        blob = b"".join(g)
        if len(blob) % PLY_LEN:
            dropped += 1
            continue
        try:
            mv = moves_of_dense(blob)
        except ValueError:
            dropped += 1
            continue
        # validate the replay -- a corrupted legacy game is not recoverable
        b = Board()
        legal = True
        for c in mv:
            if c not in b.valid_actions():
                legal = False
                break
            b.apply_move(c)
        if not legal:
            dropped += 1
            continue
        out.append(blob)
        kept += 1
        plies_total += len(g)
    return (b"\n".join(out) + b"\n" if out else b"",
            {"games_in": len(games), "kept": kept, "dropped": dropped,
             "plies": plies_total})


# ---------------------------------------------------------------------------
# source 1: forced threat decisions
# ---------------------------------------------------------------------------

FORCING = ("win_now", "block_now", "make_threat")


def threat_records(n: int, rng: random.Random,
                   walk_max: int = 40) -> tuple[List[bytes], Counter]:
    """Emit single-ply forced-decision records in dense form.

    A record is exactly one 10-byte ply (header + forced move), so it costs
    10 bytes and carries no padding waste.

    `threat_corpus._walk` returns at the FIRST position carrying ANY label,
    and `safe` matches nearly every quiet position -- so reusing it directly
    yields almost nothing but `safe`. We walk the same ladder policies but
    stop only on a FORCING label, which is what actually teaches threat
    recognition. `safe` records are kept at a small quota so the model still
    sees ordinary moves, and they are the fill when forcing positions run out.
    """
    out: List[bytes] = []
    counts: Counter = Counter()
    seen: set[bytes] = set()
    from .transcript import encode_state

    ladder = [random_policy, lookahead_policy, heuristic_policy, minimax_policy]
    weights = [0.40, 0.30, 0.22, 0.08]
    n_safe_quota = max(1, n // 10)

    tries = 0
    limit = n * 400 + 20_000
    while len(out) < n and tries < limit:
        tries += 1
        b = Board()
        found = None
        for _ in range(rng.randint(10, walk_max)):
            if is_terminal(b):
                break
            pol = rng.choices(ladder, weights=weights, k=1)[0]
            c = pol(b, [])
            if c not in b.valid_actions():
                c = rng.choice(b.valid_actions())
            b.apply_move(c)
            side = b.turn
            res = classify(b, side)
            if res is None:
                continue
            cat, col = res                       # classify() -> (category, column)
            if cat in FORCING:
                found = (side, col, cat)
                break
            # keep a `safe` example only while we still need filler
            if counts["safe"] < n_safe_quota and rng.random() < 0.05:
                found = (side, col, cat)
                break
        if found is None:
            continue
        side, col, cat = found
        if col not in b.valid_actions():
            continue
        rec = encode_state(b, b.last_move, side) + bytes([COL_BYTE_OFFSET + col])
        if rec in seen:
            continue
        seen.add(rec)
        out.append(rec)
        counts[cat] += 1
    return out, counts


# ---------------------------------------------------------------------------
# source 2: ladder games (minimax vs weaker mixture)
# ---------------------------------------------------------------------------

def ladder_games(n: int, rng: random.Random) -> tuple[List[Game], Counter]:
    """Full games: depth-6 negamax (validated teacher) on one side, a weaker
    mixture on the other. Both orientations, so we get wins for both player
    ids. The old `minimax_policy` was depth-2 and chance-level -- the
    forced-win rate against random was 47%, so the ladder material it
    produced would just dilute the corpus. Duel of the new solver: vs
    heuristic 78%, vs lookahead 82% (n=120 per pair)."""
    strong = strong_policy
    weak_pool = [lookahead_policy, heuristic_policy, random_policy]
    weak_w = [0.45, 0.35, 0.20]
    out: List[Game] = []
    counts: Counter = Counter()
    for _ in range(n):
        b = Board()
        moves: List[int] = []
        strong_side = rng.choice([P1, P2])
        while True:
            side = b.turn
            if side == strong_side:
                pol = lambda b, m, sp=strong: sp(b, m, depth=6, budget=0.02)
            else:
                pol = rng.choices(weak_pool, weights=weak_w, k=1)[0]
            c = pol(b, moves)
            if c not in b.valid_actions():
                c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
            w = b.is_win()
            if w:
                counts["strong_wins" if w == strong_side else "weak_wins"] += 1
                break
            if not b.valid_actions():
                counts["draws"] += 1
                break
        _clear_solver_tt()
        out.append((moves, w))
    return out, counts


# ---------------------------------------------------------------------------
# source 3: current model's own wins
# ---------------------------------------------------------------------------

def model_games(n: int, rng: random.Random, ckpt: str,
                device: str = "cuda") -> tuple[List[Game], Counter]:
    """Games where the live checkpoint plays both sides."""
    import torch
    from .agent import Agent
    from . import gf256  # noqa: F401  (ensures package init side-effects)

    # reuse the UI server's loader so the agent is byte-identical to serving
    from .ui_server import _load_agent
    agent, _info = _load_agent(ckpt, device=device)
    out: List[Game] = []
    counts: Counter = Counter()
    for _ in range(n):
        b = Board()
        moves: List[int] = []
        while True:
            c = agent.choose(b, moves, use_hard=True, temperature=0.8,
                             rng=rng)
            if c not in b.valid_actions():
                c = rng.choice(b.valid_actions())
            moves.append(c)
            b.apply_move(c)
            w = b.is_win()
            if w:
                counts["model_wins" if w == b.turn else "other"] += 1
                break
            if not b.valid_actions():
                counts["draws"] += 1
                break
        out.append((moves, w))
    return out, counts


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--threat", type=int, default=60_000)
    ap.add_argument("--ladder", type=int, default=4_000)
    ap.add_argument("--model", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0xC4C4_2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, write nothing")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    corpus_path = run_dir / "corpus.txt"
    raw = corpus_path.read_bytes() if corpus_path.exists() else b""

    print(f"=== existing corpus: {len(raw)} bytes ===")
    dense_old, conv = legacy_to_dense(raw)
    print(f"    legacy -> dense: {conv['games_in']} games in, "
          f"{conv['kept']} kept, {conv['dropped']} dropped "
          f"({conv['plies']} plies)")

    rng = random.Random(args.seed)
    report: dict = {"legacy": conv}

    t0 = time.perf_counter()
    recs: List[bytes] = []
    ladder: List[Game] = []
    model_w: List[Game] = []

    # ---- 1. threat records
    if args.threat:
        recs, counts = threat_records(args.threat, rng)
        report["threat"] = {"count": len(recs), **dict(counts)}
        print(f"=== threat records: {len(recs)} "
              f"({time.perf_counter()-t0:.1f}s) ===")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v:7d}  {v/max(len(recs),1)*100:5.1f}%")

    # ---- 2. ladder games
    if args.ladder:
        g, c = ladder_games(args.ladder, rng)
        ladder = winners_only(g)
        report["ladder"] = {"games": len(g), "winners": len(ladder), **dict(c)}
        print(f"=== ladder games: {len(g)} -> {len(ladder)} decided winners ===")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"    {k:12s} {v:7d}")

    # ---- 3. model games
    if args.model:
        try:
            g, c = model_games(args.model, rng,
                               str(run_dir / "checkpoints" / "serve_live.pt"),
                               device=args.device)
        except Exception as e:                      # noqa: BLE001
            print(f"=== model games SKIPPED: {type(e).__name__}: {e} ===")
            report["model"] = {"skipped": str(e)}
        else:
            mgames = winners_only(g)
            model_w.extend(mgames)
            report["model"] = {"games": len(g), "winners": len(mgames), **dict(c)}
            print(f"=== model games: {len(g)} -> {len(mgames)} decided winners ===")

    if args.dry_run:
        print("\n--- dry run, nothing written ---")
        print(json.dumps(report, indent=2))
        return

    # ---- write: converted legacy + new material, atomically
    threat_recs = recs
    new_lines: List[bytes] = []
    new_lines.extend(threat_recs)
    for mv, _w in ladder + model_w:
        new_lines.append(game_dense(mv))

    tmp = corpus_path.with_suffix(".txt.tmp")
    with tmp.open("wb") as f:
        f.write(dense_old)
        f.write(b"\n".join(new_lines) + b"\n" if new_lines else b"")
        f.flush()
    tmp.replace(corpus_path)

    st = validate_file(corpus_path)
    print(f"\n=== wrote {corpus_path} ===")
    print(f"    {st['bytes']} bytes, {st['lines']} sequences, "
          f"{st['plies_total']} plies")
    print(f"    mean sequence {st['mean_bytes_per_line']:.0f} bytes "
          f"(was 10)")
    print(f"    malformed: {st['malformed']}")
    (run_dir / "augment_report.json").write_text(
        json.dumps({**report, "final": st}, indent=2), encoding="utf-8")
    print(f"    report -> {run_dir / 'augment_report.json'}")
    print(f"    elapsed {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())