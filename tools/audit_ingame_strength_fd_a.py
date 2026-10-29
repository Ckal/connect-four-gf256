"""STRICT in-game strength comparison of the shipped fd_a vs constant columns.

WHY THIS EXISTS
---------------
`tmp/audit_ingame_cols_fd_a.py` found: against a random opponent, the
shipped fd_a went W125/L75 over 200 games while always-c3 went W186/L14
under the identical harness. That is a large gap, but a single opponent
and a single seed is not enough to call it.

This script runs the identical `src.c4.duel` harness across MULTIPLE
opponents and MULTIPLE seeds, with EXACT statistical tests.

TWO DEFECTS IN THE FIRST ATTEMPT (both fixed here)
---------------------------------------------------
1. `src.c4.duel.duel(a, b, games, seed=...)` **IGNORES its `seed` argument**:
   the source has no `random.seed(...)` call.  Every "seed" therefore
   produced the same games.  This script seeds the global RNG itself
   immediately before each duel, so the seeds are real.
2. The deterministic opponents (`heuristic`, `lookahead`, `minimax`,
   `centre`) produce only **2 distinct games per 200** -- the A/B seat
   alternation gives 2 trajectories, and the policy is deterministic.
   Counting 600 "games" that are really 6 unique games is a fabricated
   sample size.  This script therefore:
     * explores the OPPONENT (the project's stated protocol: model greedy,
       opponent explores), so the trajectory count is real;
     * reports `unique_games` for every arm, and refuses to draw a
       strength conclusion from an arm whose unique-game count is too
       low to support it.

PROTOCOL
--------
* Model is GREEDY (eps=0) at all times -- the shipping behaviour.
* The opponent gets stochastic exploration so games are genuinely distinct.
* Every arm is measured with the identical loader, seat alternation and
  trajectory accounting.

ACCEPTANCE READING
------------------
  * If the model loses to constant columns across opponents AND seeds, the
    model is NOT stronger than a constant policy in real games, and the
    release claim "board-reading model" is not supported by game evidence.
  * If the difference is within noise, the honest statement is "no
    measurable advantage over a constant column".

Artifact: runs/_probe/ingame_strength_fd_a.json
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4 import duel as D  # noqa: E402
from src.c4.opponents import (  # noqa: E402
    centre_policy,
    heuristic_policy,
    lookahead_policy,
    minimax_policy,
    random_policy,
)
from src.c4.ui_server import _load_agent  # noqa: E402

# Default checkpoint; overridable on the command line so the same harness
# can be pointed at a corrected arm once it exists.
CKPT = "runs/_probe/fd_a_s6000/checkpoints/step6000_hard.pt"
GAMES = 200
SEEDS = (99, 1234, 4242)
MIN_UNIQUE_FOR_CLAIM = 20
OUT = ROOT / "runs/_probe/ingame_strength_fd_a.json"


def binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial p-value against the 50 % null.

    Computed in LOG SPACE.  The naive form
    ``math.comb(n, i) * p**i * (1-p)**(n-i)`` overflows for n >= 1200
    (``OverflowError: int too large to convert to float``) because
    ``math.comb`` returns an int that cannot be converted once it is
    multiplied by a float.  Use ``math.lgamma`` for the log-binomial
    coefficient.
    """
    if n == 0:
        return 1.0

    def log_pmf(i):
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * math.log(p) + (n - i) * math.log1p(-p))

    obs = log_pmf(k)
    tol = obs + 1e-12
    total = 0.0
    for i in range(n + 1):
        li = log_pmf(i)
        if li <= tol:
            total += math.exp(li)
    return min(1.0, total)


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]].

    Row 1 = model's wins/losses, row 2 = reference's wins/losses.

    CORRECTNESS NOTE: an earlier version of this function used the
    ROW marginals as the binomial parameters and summed every table with a
    probability <= the observed one.  That is wrong on two counts: the
    hypergeometric parameters must be the COLUMN marginals, and the
    two-sided rule must include tables at least as EXTREME as the observed
    one in either tail.  The broken version returned p = 1.0 for a
    400-vs-546 win gap, which is obviously wrong.  This version is
    validated against the textbook reference: the two-sided Fisher p for
    [[1, 9], [11, 3]] is 0.00275946, and this function reproduces it.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2 = a + b, c + d          # row totals
    k1, k2 = a + c, b + d          # column totals

    def pmf(x):
        return math.comb(k1, x) * math.comb(k2, r1 - x) / math.comb(n, r1)

    lo = max(0, r1 - k2)
    hi = min(r1, k1)
    obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= obs + 1e-15))


def const_policy(col):
    """A TRUE constant-column opponent.

    The original version was `col if col in legal else legal[0]`.  That is
    NOT a constant policy: once the preferred column fills, it falls back
    to the FIRST legal column, which is a centre-first priority sweep.  An
    earlier measurement of "always-c3" used exactly that fallback and
    therefore played c3 on only 28.2 % of its moves -- the arm was never
    the opponent it claimed to be.

    A true constant policy plays its preferred column while it is legal
    and, once it is full, picks ANOTHER legal column at random.  The
    random fallback is what keeps the arm from degenerating into a
    priority sweep.

    NOTE: this policy is deterministic apart from the fallback, so it is
    wrapped in `exploring()` like every other opponent -- the project
    protocol is model-greedy, opponent-explores.
    """
    state = {"rng": random.Random(1000 + col)}

    def pol(board, moves):
        legal = [c for c in range(7) if board.heights[c] < 6]
        if not legal:
            return 0
        if col in legal:
            return col
        return state["rng"].choice(legal)
    return pol


def exploring(policy, eps, rng_seed):
    """Wrap an opponent policy with stochastic exploration.

    The project protocol is: model greedy, opponent explores.  Exploration
    is applied only to the OPPONENT, and the exploration draw comes from a
    per-call RNG seeded from the global RNG, so re-seeding between games
    actually changes the games.
    """
    state = {"rng": random.Random(rng_seed)}

    def pol(board, moves):
        legal = [c for c in range(7) if board.heights[c] < 6]
        if not legal:
            return 0
        if state["rng"].random() < eps:
            return state["rng"].choice(legal)
        return policy(board, moves)
    return pol


def instrument(agent):
    base = D._model_policy(agent, hard=True, eps=0.0, rng=None)
    seen = collections.Counter()

    def pol(board, moves):
        c = base(board, moves)
        seen[c] += 1
        return c
    return pol, seen


def run_arm(arm_pol, opp, games, seed):
    """Run one arm with a REAL seed (duel ignores its own seed argument)."""
    random.seed(seed)
    r = D.duel(arm_pol, opp, games, seed=seed)
    return r


def main() -> int:
    global CKPT, OUT, GAMES
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT,
                    help="hard checkpoint to measure (path relative to repo root)")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--games", type=int, default=GAMES, help="games per arm per seed")
    a = ap.parse_args()
    CKPT = a.ckpt
    GAMES = a.games
    if a.out:
        OUT = ROOT / a.out if not a.out.startswith("/") else pathlib.Path(a.out)
    agent = _load_agent(str(ROOT / CKPT), device="cpu")

    # epsilon for opponent exploration -- enough to make trajectories
    # genuinely distinct without turning the opponent into noise.
    opponent_specs = {
        "random": (random_policy, 0.0),
        "heuristic": (heuristic_policy, 0.30),
        "lookahead": (lookahead_policy, 0.30),
        "minimax": (minimax_policy, 0.30),
        "centre": (centre_policy, 0.30),
    }

    arms: dict[str, object] = {f"always-c{c}": const_policy(c) for c in range(7)}
    arms["centre_policy"] = centre_policy

    out = {
        "checkpoint": CKPT,
        "protocol": {
            "harness": "src.c4.duel.duel",
            "games_per_arm_per_seed": GAMES,
            "seeds": list(SEEDS),
            "model_policy": "greedy (eps=0), hard path",
            "opponent_exploration_eps": {k: v[1] for k, v in opponent_specs.items()},
            "seed_note": "src.c4.duel.duel IGNORES its seed argument; this "
                         "script seeds the global RNG itself before each duel.",
            "unique_game_requirement": MIN_UNIQUE_FOR_CLAIM,
        },
        "results": [],
    }

    print(f"=== strict in-game strength: {CKPT} ===")
    print(f"  {GAMES} games per arm per seed, seeds={SEEDS}")
    print(f"  NOTE: duel() ignores its seed arg; we seed the global RNG.\n")

    for opp_name, (opp_fn, eps) in opponent_specs.items():
        print(f"--- opponent: {opp_name} (eps={eps}) ---")
        arm_records: dict[str, dict] = {}

        # ---- model ----
        model_w = model_l = model_d = 0
        model_uniq = []
        model_cols = collections.Counter()
        for seed in SEEDS:
            pol, seen = instrument(agent)
            opp = exploring(opp_fn, eps, seed)
            r = run_arm(pol, opp, GAMES, seed)
            model_w += r["A_wins"]
            model_l += r["B_wins"]
            model_d += r["draws"]
            model_uniq.append(r["unique_games"])
            model_cols.update(seen)
        n_model = model_w + model_l + model_d
        p_model = binom_two_sided_p(model_w, model_w + model_l)
        print(f"  MODEL            W{model_w:4d} L{model_l:4d} D{model_d:3d}  "
              f"win%={100*model_w/n_model:5.1f}  p(vs 50%)={p_model:.4f}  "
              f"unique={model_uniq}")
        arm_records["MODEL"] = {
            "w": model_w, "l": model_l, "d": model_d, "n": n_model,
            "p_vs_50": p_model, "unique_games_per_seed": model_uniq,
            "min_unique": min(model_uniq),
        }

        # ---- constant arms ----
        best_arm = None
        for arm_name, arm_pol in arms.items():
            aw = al = ad = 0
            uniq = []
            for seed in SEEDS:
                opp = exploring(opp_fn, eps, seed)
                r = run_arm(arm_pol, opp, GAMES, seed)
                aw += r["A_wins"]
                al += r["B_wins"]
                ad += r["draws"]
                uniq.append(r["unique_games"])
            n = aw + al + ad
            p = binom_two_sided_p(aw, aw + al)
            arm_records[arm_name] = {
                "w": aw, "l": al, "d": ad, "n": n, "p_vs_50": p,
                "unique_games_per_seed": uniq, "min_unique": min(uniq),
            }
            print(f"  {arm_name:16s} W{aw:4d} L{al:4d} D{ad:3d}  "
                  f"win%={100*aw/n:5.1f}  p(vs 50%)={p:.4f}  unique={uniq}")
            if best_arm is None or aw > arm_records[best_arm]["w"]:
                best_arm = arm_name

        if best_arm is None:
            print("  >>> no constant arm produced a result; skipping comparison\n")
            continue
        bw = arm_records[best_arm]["w"]
        bl = arm_records[best_arm]["l"]
        fisher = fisher_exact_2x2(model_w, model_l, bw, bl)
        print(f"  >>> best constant: {best_arm} W{bw} L{bl};  "
              f"Fisher exact model-vs-best p={fisher:.4f}")

        # Two INDEPENDENT questions must not be conflated:
        #   (a) is the comparison statistically valid (enough distinct games)?
        #   (b) given that, is the difference significant, and in which direction?
        # The previous wording printed "COMPARABLE -- MODEL LOSES ...", which
        # reads as if "COMPARABLE" qualified the strength claim.  Separate them.
        model_uniq_ok = arm_records["MODEL"]["min_unique"] >= MIN_UNIQUE_FOR_CLAIM
        best_uniq_ok = arm_records[best_arm]["min_unique"] >= MIN_UNIQUE_FOR_CLAIM
        valid = "COMPARISON VALID" if (model_uniq_ok and best_uniq_ok) else \
                "NOT COMPARABLE (too few unique games)"
        if fisher < 0.05:
            direction = ("MODEL BEATS best constant" if model_w > bw
                         else "MODEL LOSES to best constant")
            stat = f"significant difference -- {direction} (p={fisher:.4f})"
        else:
            stat = f"no significant difference (p={fisher:.4f})"
        verdict = f"{valid} -- {stat}"
        print(f"  >>> VERDICT: {verdict}\n")

        out["results"].append({
            "opponent": opp_name,
            "opponent_eps": eps,
            "model": arm_records["MODEL"],
            "model_cols": {str(c): model_cols.get(c, 0) for c in range(7)},
            "arms": arm_records,
            "best_constant_arm": best_arm,
            "fisher_exact_model_vs_best_constant_p": fisher,
            "verdict": verdict,
        })

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
