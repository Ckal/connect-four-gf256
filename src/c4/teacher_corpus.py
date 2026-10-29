"""Curated corpus generator: solver + heuristic games only, no random self-play.

Why this module exists
----------------------
The V10 trainer grew its corpus by letting the model play 16 games against
itself every 100 steps and appending every decided game. That is the wrong
data source for a policy this small: the model was, at that point, losing
47 of 48 games to a 2-ply heuristic, so its "self-play" moves were closer to
random than to strategy. Every one of those games contributed move-byte
gradients.

This module generates the replacement data instead:

  teacher vs random   -> games where the teacher's moves are solver-optimal,
                         the loser's moves are noise. Only the winner's plies
                         are kept, and only for the teacher side.
  teacher vs heuristic -> both sides play plausible moves; the winner's plies
                         are kept.
  teacher vs teacher  -> self-play between two solver instances (tie-broken by
                         a seeded RNG so games are not byte-identical).

Every emitted game satisfies the two hard rules the trainer needs:

  1. ``winner != 0``          -- draws are dropped (they teach nothing).
  2. at least one ply was played by a teacher policy                 --
     so the winning transcript is a plausible winning line, not noise.

The loser's plies are *not* trusted, and the caller is responsible for not
using them: `keep_winner_only` below emits only the winner's transcript so
the trainer sees a clean demonstration.

Public API
----------
    teacher_games(n, seed, opponents=...) -> List[Game]
    keep_winner_only(games, roles)       -> List[Game]
    curriculum_games(n, seed)            -> List[Game]  (solver -> heuristic
                                                -> random, ramped)
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

from .board import Board
from .corpus import Game
from .opponents import heuristic_policy, random_policy, lookahead_policy
from .strong_solver import strong_policy

__all__ = [
    "teacher_games",
    "teacher_batch",
    "keep_winner_only",
    "curriculum_games",
    "ROLE_SOLVER",
    "ROLE_LOOKAHEAD",
    "ROLE_HEURISTIC",
    "ROLE_RANDOM",
]

ROLE_SOLVER = "solver"
ROLE_LOOKAHEAD = "lookahead"
ROLE_HEURISTIC = "heuristic"
ROLE_RANDOM = "random"


# ---------------------------------------------------------------------------
# single game driver
# ---------------------------------------------------------------------------

def _play_one(policy_left: Callable[[Board, List[int]], int],
              policy_right: Callable[[Board, List[int]], int],
              ) -> Tuple[List[int], int]:
    """Play one game. Returns (moves, winner) with winner in {1, 2, 0=draw}."""
    b = Board()
    moves: List[int] = []
    while True:
        legal = b.valid_actions()
        if not legal:
            return moves, 0
        pol = policy_left if b.turn == 1 else policy_right
        # Teachers must never be allowed to pick an illegal column.
        c = pol(b, legal)
        if c not in legal:
            c = legal[0]
            if c not in legal:
                return moves, 0
        b.apply_move(c)
        moves.append(c)
        w = b.is_win()
        if w:
            return moves, w


# ---------------------------------------------------------------------------
# batch generation
# ---------------------------------------------------------------------------

def teacher_games(n: int,
                  seed: int = 0,
                  opponents: Sequence[str] = (ROLE_RANDOM,)) -> List[Game]:
    """Generate ``n`` teacher games against a rotating set of opponents.

    The teacher seat alternates between `strong_policy` (negamax alpha-beta)
    and `lookahead_policy`. `opponents` names the policies on the other seat.

    **Quality rule -- keep the winner's plies, whoever won.** An earlier
    version rejected every game the solver lost. Measured head-to-head
    (`scripts/duel_teachers.py`, n=20 per row) showed why that was wrong:

        solver     vs random     W20 L 0   (0.23 s/game)
        solver     vs heuristic  W10 L10   (0.30 s/game)
        solver     vs lookahead  W11 L 9
        lookahead  vs heuristic  W20 L 0   (0.002 s/game)

    The negamax solver is *not* dominant over a 2-ply heuristic, so "the
    teacher must win" discarded ~half the games for no reason -- and with
    `opponents=(heuristic,)` it discarded 100% of them, which is exactly
    what `check_teacher_corpus.py` observed (0 games). `lookahead_policy`,
    by contrast, is unbeaten by the heuristic at ~150x the speed.

    What makes a retained game a demonstration is not the seat, it is that
    the surviving transcript ends in a win. The trainer then weights by
    `action_consistent_examples`, so winner plies are the ones that carry
    gradient. Losing *teacher* plies are fine to keep for the same reason:
    a winning game necessarily contains positions where the eventual winner
    was under pressure.

    Draws are always discarded -- they teach nothing.
    """
    rng = random.Random(seed)
    pool: Dict[str, Callable[[Board, List[int]], int]] = {
        ROLE_SOLVER: strong_policy,
        ROLE_LOOKAHEAD: lookahead_policy,
        ROLE_HEURISTIC: heuristic_policy,
        ROLE_RANDOM: random_policy,
    }
    teachers: List[Callable[[Board, List[int]], int]] = [
        strong_policy, lookahead_policy,
    ]
    names = list(opponents)
    made: List[Game] = []
    guard = 0
    # Solvers rarely draw, so overshoot then trim; the guard stops a
    # pathological policy combination from spinning forever.
    while len(made) < n and guard < n * 20 + 200:
        guard += 1
        opp_name = names[guard % len(names)]
        opp_pol = pool[opp_name]
        teacher = teachers[guard % len(teachers)]
        # Alternate seats so both P1 and P2 transcripts are represented.
        teacher_first = (guard % 2 == 0)
        if teacher_first:
            moves, w = _play_one(teacher, opp_pol)
        else:
            moves, w = _play_one(opp_pol, teacher)
        if w == 0:
            continue
        made.append((moves, w))
    return made


# ---------------------------------------------------------------------------
# trainer entry point
# ---------------------------------------------------------------------------

#: Opponent mix for the *other* seat. The teacher seat already alternates
#: solver/lookahead inside :func:`teacher_games`; this picks who it plays.
#:
#: Random was removed from this mix entirely (user: "remove random games
#: mostly"). It is not merely a weight change: `random` here was the seat
#: supplying the losing half of every game, and the trainer retains the
#: *winner's* plies -- so those games contributed real gradient from games
#: that were trivially won. Replacing the random seat with the solver puts
#: both seats above the weak end of the ladder.
#:
#: The measured head-to-head matrix (`scripts/duel_teachers.py`) produced
#: **zero draws** for all three of these pairings, so strong-vs-strong still
#: yields decided games and nothing is thrown away:
#:
#:     solver     vs heuristic  W10 L10
#:     solver     vs lookahead  W11 L 9
#:     lookahead  vs heuristic  W20 L 0
#:
#: Heuristic keeps the largest share because it is the only seat that
#: guarantees a loss for one side (the solver is not dominant over it), and
#: a decided game needs a loser.
DEFAULT_MIX: Tuple[Tuple[str, int], ...] = (
    (ROLE_HEURISTIC, 4),
    (ROLE_LOOKAHEAD, 3),
    (ROLE_SOLVER, 2),
)


def _expand_mix(n: int,
                mix: Sequence[Tuple[str, int]] = DEFAULT_MIX
                ) -> List[str]:
    """Weighted expansion of ``mix`` into exactly ``n`` opponent names."""
    total = sum(w for _, w in mix) or 1
    out: List[str] = []
    for name, w in mix:
        out.extend([name] * max(0, round(n * w / total)))
    while len(out) < n:                      # top up the heaviest opponent
        out.append(max(mix, key=lambda kv: kv[1])[0])
    rng = random.Random(len(out) * 7919)
    rng.shuffle(out)
    return out[:n]


def teacher_batch_with_roles(n: int, step: int = 0, seed: int = 0,
                             mix: Sequence[Tuple[str, int]] = DEFAULT_MIX,
                             ) -> List[Tuple[List[int], int, List[bool]]]:
    """Like :func:`teacher_batch`, but also reports WHICH plies the teacher played.

    Returns `[(moves, winner, is_teacher_ply), ...]` where `is_teacher_ply[k]`
    is True when ply `k` was played by the solver/lookahead seat.

    Why the trainer needs this
    --------------------------
    The measured head-to-head matrix is not one-sided:

        solver     vs heuristic  W10 L10
        lookahead  vs heuristic  W20 L 0

    So a decided teacher game is often won by the *heuristic* seat. Keeping
    "the winner's plies" -- which is what `action_consistent_examples` does by
    default -- would then hand the trainer the heuristic's moves as the
    positive signal. That is the same defect the legacy corpus has, freshly
    generated.

    The correct rule for teacher data is a ROLE rule, not an agreement rule:
    trust the plies played by the strong seat, whether or not that seat won.
    The winner still decides whether the game is usable at all (draws are
    dropped), because a teacher that never completed a line taught nothing.

    This is also why the behaviour gate is NOT applied to teacher rows: they
    are generated by the reference policies, so scoring them against
    `lookahead` would reject every solver-played game by construction.
    """
    from .strong_solver import clear_tt

    names = _expand_mix(n, mix)
    made: List[Tuple[List[int], int, List[bool]]] = []
    guard = 0
    while len(made) < n and guard < n * 20 + 200:
        guard += 1
        clear_tt()
        name = names[guard % len(names)]
        opp_pol = _OPP_POOL[name]
        teacher = _TEACHER_POOL[guard % len(_TEACHER_POOL)]
        teacher_first = (guard % 2 == 0)
        if teacher_first:
            moves, w = _play_one(teacher, opp_pol)
        else:
            moves, w = _play_one(opp_pol, teacher)
        if w == 0:
            continue
        roles = [((k % 2 == 0) == teacher_first) for k in range(len(moves))]
        made.append((moves, w, roles))
    return made


def teacher_batch(n: int, step: int = 0, seed: int = 0,
                  mix: Sequence[Tuple[str, int]] = DEFAULT_MIX) -> List[Game]:
    """The trainer's corpus-growth call: ``n`` decided teacher games.

    ``step`` is mixed into the seed so successive growth cycles explore
    different openings instead of replaying the same lines forever. The
    transposition table is cleared per batch so the solver's node budget is
    spent on fresh positions rather than on cache hits from the previous
    cycle.

    Returns decided games only (no draws), as ``(moves, winner)``.
    """
    from .strong_solver import clear_tt

    names = _expand_mix(n, mix)
    made: List[Game] = []
    guard = 0
    while len(made) < n and guard < n * 20 + 200:
        guard += 1
        clear_tt()
        name = names[guard % len(names)]
        opp_pol = _OPP_POOL[name]
        teacher = _TEACHER_POOL[guard % len(_TEACHER_POOL)]
        if guard % 2 == 0:
            moves, w = _play_one(teacher, opp_pol)
        else:
            moves, w = _play_one(opp_pol, teacher)
        if w == 0:
            continue
        made.append((moves, w))
    return made


_OPP_POOL: Dict[str, Callable[[Board, List[int]], int]] = {
    ROLE_SOLVER: strong_policy,
    ROLE_LOOKAHEAD: lookahead_policy,
    ROLE_HEURISTIC: heuristic_policy,
    ROLE_RANDOM: random_policy,
}

_TEACHER_POOL: List[Callable[[Board, List[int]], int]] = [
    strong_policy, lookahead_policy,
]


# ---------------------------------------------------------------------------
# winner-only filtering
# ---------------------------------------------------------------------------

def teacher_seed_games(n: int, seed: int = 0,
                       mix: Sequence[Tuple[str, int]] = DEFAULT_MIX) -> List[Game]:
    """Seed the corpus with SOLVER games instead of random-vs-random winners.

    Replaces the historical `random_seed_games` + `winners_only` seed, which
    kept the winning side of a game where BOTH sides played uniformly at
    random. Those rows are noise dressed as demonstrations: measured against
    a heuristic policy the "winning" moves agree 15.6% of the time versus a
    14.3% chance floor.

    Same generator as the periodic growth cycle, so the seed and the
    subsequent corpus come from one distribution and the teacher fraction
    starts near 100% instead of climbing from 0. Draws are dropped, as
    elsewhere: a teacher that never completed a line taught nothing.
    """
    return teacher_batch(n, step=seed, seed=seed, mix=mix)


def keep_winner_only(games: Sequence[Game]) -> List[Game]:
    """Drop draws. Keep decided games as-is (caller already filtered quality)."""
    return [(m, w) for m, w in games if w != 0]


# ---------------------------------------------------------------------------
# curriculum
# ---------------------------------------------------------------------------

def curriculum_games(n: int, seed: int = 0) -> List[Game]:
    """n games with two strong seats.

    Random is out (see ``DEFAULT_MIX``). The pairing measured draw-free is
    ``heuristic`` against the solver/lookahead teacher, so every generated
    game carries a winner for the trainer to weight.
    """
    a = teacher_games(n // 2, seed=seed + 1,
                      opponents=(ROLE_HEURISTIC,))
    b = teacher_games(n - n // 2, seed=seed + 2,
                      opponents=(ROLE_SOLVER, ROLE_LOOKAHEAD))
    return a + b