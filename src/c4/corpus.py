"""Self-play data generation for Connect-Four with the GF(256) Markov LM.

`Policy` is anything that, given a board and a move history, returns a
column index [0..6]. The default (`random_policy`) is a uniform-random
move among the valid columns. The agent's own policy is plugged in by
the training driver.

Game = Tuple[moves, outcome] where outcome is the player number that won
(1, 2) or 0 for a draw. Only WINNERS (no draws) are appended to the
corpus -- a winners-only corpus is the simplest pure bootstrap and it
removed the +1 noise class once the agent was strong enough to draw.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

from . import P1, P2
from .board import Board, Outcome, is_terminal
from .transcript import encode_game

# Type aliases used across corpus + training.
Game = Tuple[List[int], int]
Winners = List[Game]
Policy = Callable[[Board, List[int]], int]


def random_policy(b: Board, _moves: List[int]) -> int:
    """Default: pick uniformly among the legal columns."""
    return random.choice(b.valid_actions())


def play_random_game(rng: random.Random,
                     policy: Policy = random_policy) -> Game:
    """Roll out one game with `policy` on every ply. Returns (moves, winner)."""
    board = Board()
    moves: List[int] = []
    while True:
        col = policy(board, moves)
        moves.append(col)
        board.apply_move(col)
        winner = board.is_win()
        if is_terminal(board):
            return moves, winner


def play_mixed_game(rng: random.Random,
                    p1: Policy,
                    p2: Policy) -> Game:
    """Roll out one game with `p1` playing side 0 and `p2` playing side 1.

    This is the right primitive for self-play with a strong opponent: the
    winners of these games are winning *because* they played well, not
    because their opponent blundered. Wins concentrated in `p1` mean
    those are the trajectories we want to learn from.

    Returns (moves, winner) where winner is one of {0:draw, 1:p1_won, 2:p2_won}.
    """
    board = Board()
    moves: List[int] = []
    while True:
        col = (p1 if board.turn == 0 else p2)(board, moves)
        moves.append(col)
        board.apply_move(col)
        if board.is_win():
            return moves, 1 if board.turn == 0 else 2
        if not board.valid_actions():
            return moves, 0


def play_games(n: int, rng: random.Random,
               policy: Policy = random_policy) -> List[Game]:
    """Roll out `n` games, return their (moves, winner) lists."""
    out: List[Game] = []
    for _ in range(n):
        out.append(play_random_game(rng, policy))
    return out


def play_mixed_games(n: int, rng: random.Random,
                     p1_weight: dict[str, float],
                     p2_weight: dict[str, float],
                     policies: dict[str, Policy]) -> List[Game]:
    """Roll out `n` mixed games.

    `p1_weight` and `p2_weight` are name -> weight dicts over `policies`
    (e.g. {"random":0.25, "heuristic":0.25, "lookahead":0.25, "minimax":0.25}).
    Weights are normalised before sampling. Common pattern: pick one set of
    weights to use as both sides (symmetric mixed play).

    Winner is from the *game*'s perspective (1 = p1, 2 = p2, 0 = draw).
    """
    keys1 = list(p1_weight.keys()); w1 = [p1_weight[k] for k in keys1]
    keys2 = list(p2_weight.keys()); w2 = [p2_weight[k] for k in keys2]
    s1 = sum(w1) or 1.0; w1 = [x / s1 for x in w1]
    s2 = sum(w2) or 1.0; w2 = [x / s2 for x in w2]
    out: List[Game] = []
    for _ in range(n):
        k1 = keys1[_rng_index(rng, w1)]
        k2 = keys2[_rng_index(rng, w2)]
        out.append(play_mixed_game(rng, policies[k1], policies[k2]))
    return out


def _rng_index(rng: random.Random, weights: List[float]) -> int:
    """Sample an index proportional to weights (matches Python 3 random.choices semantics)."""
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return len(weights) - 1


def winners_only(games: List[Game]) -> Winners:
    """Drop draws. These go into the corpus; draws are noise at this stage."""
    return [(m, w) for m, w in games if w != 0]


def games_to_lines(games: List[Game]) -> List[bytes]:
    """Encode games as one big stream of newline-terminated ply records."""
    out: List[bytes] = []
    for moves, _winner in games:
        out.append(encode_game(moves))
    return out


def append_corpus(path: str, decided: List[Game]) -> int:
    """Append winners' games as bytes to `path`, returning new line count."""
    n = 0
    with open(path, "ab") as f:
        for line in games_to_lines(winners_only(decided)):
            f.write(line)
            n += 1
    return n


def stats(games: List[Game]) -> dict:
    """Sanity report on a batch of self-play games."""
    p1, p2, dr = 0, 0, 0
    nplies = []
    for _m, w in games:
        if w == 1:
            p1 += 1
        elif w == 2:
            p2 += 1
        else:
            dr += 1
    return dict(p1=p1, p2=p2, draws=dr,
                total=len(games),
                avg_plies=(sum(len(m) for m, _ in games) / max(1, len(games))))
