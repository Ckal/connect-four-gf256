"""Behaviour filtering for dense Connect Four corpus rows.

The problem this solves
----------------------
`corpus_quality.py` grades a row by asking *"did the winner ever pass up a
winning move?"*. That is a tactical-blunder test. It is necessary but it is
not the question we actually care about, and on this corpus it answers a
different one.

The model is a **next-byte predictor**. A row supervises it by saying "given
the board transcript so far, the next byte is `X`". That statement is true
and useful only when `X` is the move a competent player would have played.
A row can contain zero dropped wins and still be worthless:

    * a random-vs-random game where the winner happened to connect four
      without ever being offered a win early on,
    * a game where one side was flailing but the transcript shows no *missed
      block*, because the other side never built a threat.

Both grade GOOD under a blunder test. Both are exactly the "legal but bad"
data the user wants gone. Worse, the blunder test is *harsher* on real
teacher games than on noise: a strong game often contains a ply where the
teacher walked past an available win because it was already playing for a
higher-value line. That is correct play, graded as a blunder.

The correct test is behavioural: **replay the row, ask a strong policy what
it would have played at every ply, and measure how often the actual move
agreed.** A row that tracks the teacher is a demonstration. A row that does
not is noise wearing the costume of a game.

What is measured
----------------
For each row, at each ply, before the move is applied:

    top1  = the column the reference policy prefers, restricted to legal
    frac  = agreement = (#plies where actual == top1) / nplies

Two reference policies are scored, because "agreement" only means something
relative to a specific teacher:

    heuristic  1-ply, near-free, good enough to catch flailing
    lookahead  2-ply, ~150x faster than the solver and unbeaten by the
               heuristic head-to-head (`scripts/duel_teachers.py` W20 L0)

The negamax solver is deliberately NOT used here: at ~0.3 s/game it would
need minutes per corpus sweep and, per the measured head-to-head matrix
(solver vs heuristic W10 L10), it is not dominant enough to serve as the
definition of "right".

Only the WINNER's plies are scored. The loser's plies are noise by
construction -- we know they lost, and we do not want to fit them -- but
they still count in the denominator for the ply count, because a game where
the winner played perfectly and the loser threw the game should NOT be
excluded for low overall agreement.

Thresholds are read off the measured distribution, not guessed. See
`reports/` and `scripts/check_v12_behavior.py`.

Nothing here deletes anything. As with `corpus_quality.py`, `corpus.txt` is
never rewritten; exclusion happens at training-example construction and every
exclusion is counted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from .board import Board
from .opponents import heuristic_policy, lookahead_policy

# A Game is (moves, winner); the trainer's existing alias.
Game = Tuple[List[int], int]

#: Policies available as a behaviour reference, by name.
REFERENCE_POLICIES: Dict[str, Callable[[Board, List[int]], int]] = {
    "heuristic": heuristic_policy,
    "lookahead": lookahead_policy,
}


def _top1(policy: Callable[[Board, List[int]], int], b: Board) -> int:
    """The column `policy` wants in the current position.

    Guarded: every reference policy is required to return a legal column.
    An illegal answer is replaced with the first legal column rather than
    being allowed to corrupt the board or abort the row.
    """
    legal = b.valid_actions()
    if not legal:
        return -1
    try:
        c = policy(b, list(legal))
    except Exception:
        c = -1
    return c if c in legal else legal[0]


@dataclass
class Behavior:
    """Behaviour profile of one dense corpus row."""

    nplies: int
    winner: int
    #: agreement over ALL plies, in [0, 1]
    agree_all: float
    #: agreement over the WINNER's plies only, in [0, 1]
    agree_winner: float
    #: per-ply agreement over the winner's plies, for auditing a rejection
    winner_hits: int = 0
    winner_total: int = 0


def score_moves(moves: Sequence[int], winner: int,
                policy: Callable[[Board, List[int]], int]) -> Behavior:
    """Score one move list against a reference policy."""
    b = Board()
    hits_all = 0
    hits_win = 0
    n_win = 0
    n = 0
    for k, c in enumerate(moves):
        legal = b.valid_actions()
        if not legal:
            break
        player = (k % 2) + 1
        want = _top1(policy, b)
        n += 1
        if c == want:
            hits_all += 1
            if player == winner:
                hits_win += 1
        if player == winner:
            n_win += 1
        b.apply_move(c)
        if b.is_win():
            # `winner` is authoritative from the replay grade; if the caller
            # passed 0 (unresolved) adopt the real winner.
            if winner == 0:
                winner = b.is_win()
            break
    return Behavior(
        nplies=n,
        winner=winner,
        agree_all=(hits_all / n) if n else 0.0,
        agree_winner=(hits_win / n_win) if n_win else 0.0,
        winner_hits=hits_win,
        winner_total=n_win,
    )


def behavior_gate_games(games: Sequence[Game],
                         *,
                         policy_name: str = "lookahead",
                         min_winner_agreement: float = 0.60,
                         verbose: bool = True,
                         ) -> Tuple[List[Game], dict]:
    """Keep `(moves, winner)` rows whose winner played like the reference.

    This is the form the trainer uses: `quality_gate` has already decoded the
    dense rows into `(moves, winner)` pairs, so re-decoding them would be
    wasted work. `behavior_gate` is the byte-level convenience wrapper.

    Only the winner's plies are scored. `min_winner_agreement` is the share
    of the winner's plies that must match the reference top-1.

    Returns `(games, report)`.
    """
    policy = REFERENCE_POLICIES[policy_name]
    kept: List[Game] = []
    buckets: Dict[str, int] = {"<0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0,
                               "0.8-0.9": 0, ">=0.9": 0}
    scanned = 0
    unresolved = 0
    kept_pl = 0
    hist: List[Tuple[float, float]] = []

    for moves, winner in games:
        if winner == 0:
            unresolved += 1
            continue
        if not moves:
            continue
        bh = score_moves(moves, winner, policy)
        if bh.nplies == 0:
            continue
        scanned += 1
        a = bh.agree_winner
        hist.append((a, bh.agree_all))
        if a < 0.4:
            buckets["<0.4"] += 1
        elif a < 0.6:
            buckets["0.4-0.6"] += 1
        elif a < 0.8:
            buckets["0.6-0.8"] += 1
        elif a < 0.9:
            buckets["0.8-0.9"] += 1
        else:
            buckets[">=0.9"] += 1
        if a >= min_winner_agreement:
            kept.append((list(moves), winner))
            kept_pl += bh.nplies

    hist.sort()

    def pct(p: float) -> float:
        if not hist:
            return 0.0
        i = min(len(hist) - 1, max(0, int(p * len(hist))))
        return hist[i][0]

    report = {
        "policy": policy_name,
        "min_winner_agreement": min_winner_agreement,
        "scanned": scanned,
        "unresolved_skipped": unresolved,
        "kept": len(kept),
        "kept_plies": kept_pl,
        "agreement_buckets": buckets,
        "winner_agreement_percentiles": {
            "p10": pct(0.10), "p25": pct(0.25), "p50": pct(0.50),
            "p75": pct(0.75), "p90": pct(0.90), "p99": pct(0.99),
        },
        "mean_winner_agreement": (
            sum(h[0] for h in hist) / len(hist) if hist else 0.0),
        "mean_allply_agreement": (
            sum(h[1] for h in hist) / len(hist) if hist else 0.0),
    }

    if verbose:
        print(f"behaviour gate vs {policy_name} "
              f"(min winner agreement {min_winner_agreement:.2f}): "
              f"{scanned} rows scanned -> {len(kept)} kept", flush=True)
        for k in ("<0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", ">=0.9"):
            print(f"    agreement {k:8s}  {buckets[k]:6d}", flush=True)
        p = report["winner_agreement_percentiles"]
        print(f"    winner-agreement  mean {report['mean_winner_agreement']:.3f}  "
              f"p25 {p['p25']:.2f}  p50 {p['p50']:.2f}  p75 {p['p75']:.2f}",
              flush=True)
    return kept, report


def behavior_gate(lines: Sequence[bytes],
                  moves_of_dense: Callable[[bytes], List[int]],
                  winner_of_dense: Callable[[bytes], int],
                  **kw,
                  ) -> Tuple[List[Game], dict]:
    """Byte-level convenience wrapper around :func:`behavior_gate_games`.

    Decodes each dense row with the injected functions and delegates. Used by
    `scripts/measure_behavior.py`, which has raw rows rather than games.
    """
    games = [(moves_of_dense(ln), winner_of_dense(ln))
             for ln in lines if ln]
    return behavior_gate_games(games, **kw)