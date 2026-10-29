"""Headless A/B duel harness for Connect-Four checkpoints.

Why this exists
---------------
`play.py` only plays against a human, so there was no way to answer the one
question that matters before publishing: *is this checkpoint actually better
than the one already live?* Hard-acc on a held-out byte set is a proxy -- two
checkpoints can tie on it while differing sharply in game outcomes.

Design notes
------------
* **Alternating seats.** Every (A, B) pairing is played with A as P1 and then
  again with A as P2. Connect Four's first move is a real advantage, so
  reporting a single seat would bias the verdict.
* **Opponent is pluggable.** Heuristic by default (so the question is "did
  this get better at Connect Four"), plus `random` for a floor and `exact`
  for the endgame quality floor.
* **Forced-win detection.** `Board.is_win()` ends the game immediately, so we
  stop at the first four. No rubber-banding.
* **Exploration matters -- this harness used to lie (bug fixed here).**
  `play_one` took a `seed` argument and never used it, and every built-in
  opponent plus the model's own greedy decode is a *deterministic* function of
  (board, moves). So N "games" collapsed to 2 unique games: the same P1 game
  repeated N/2 times and the same P2 game repeated N/2 times. A 30/30-0/30
  record was not 60 trials -- it was ONE game per seat. And an earlier
  A-vs-A "symmetry check" that returned exactly 50% was not evidence of seat
  symmetry at all; it was this same artifact.

  Two fixes, because one alone is not enough:
    1. ``--eps`` plays A epsilon-greedy, so repeated games actually diverge
       and the sample carries information about tactics.
    2. EVERY result prints ``unique_games`` (distinct move sequences actually
       observed) and warns loudly when it is below the requested count, so a
       strength gate cannot quote a number that looks precise and is not.

Usage
-----
    python -m src.c4.duel --a runs/.../best_hard.pt --b runs/.../serve_live.pt \
        --games 200 --eps 0.15
    python -m src.c4.duel --a ckpt.pt --opponent heuristic --games 200 --eps 0.15
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, List, Optional, Tuple

import torch

from . import COLS, P1, P2
from . import fmt
from .agent import Agent
from .board import Board, is_terminal
from .opponents import get as get_opponent
from .ui_server import _load_agent, hard_to_cpu


def _load_hard_agent(ckpt_path: str) -> Agent:
    """Load a checkpoint for HARD-path-only play.

    Delegates to the app's own ``_load_agent`` so the arch is rebuilt exactly
    the way the served game rebuilds it, then repairs the one incompatibility:
    V6-class payloads (d<64) predate ``GFMarkovLM.import_hard`` -- that method
    was added in V7 -- and ``_load_agent`` calls it unconditionally, raising
    AttributeError on an otherwise perfectly loadable V6 model.

    This is harmless for us: ``import_hard`` only copies integer weights into
    the float ``nn.Parameter`` buffers to enable the SOFT ``column_logits()``
    path. The duel plays exclusively through the hardened integer forward,
    which reads ``agent.hard`` directly, so the soft buffers are never touched.
    """
    try:
        return _load_agent(ckpt_path, device="cpu")
    except AttributeError as exc:
        if "import_hard" not in str(exc):
            raise

        from .ui_server import GFMarkovLM_V6, GFMarkovLM_V7

        # weights_only=False is required and matches the repo's own loader:
        # an export_hard() payload is a dict of LISTS of tensors (coef/sbox/
        # wiring/readout) with (channel, lag) int-tuple wiring entries, which
        # weights_only=True cannot unpickle. These files are produced by this
        # repo's own training runs.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        d = int(payload["emb"].shape[1])
        cls = GFMarkovLM_V7 if d >= 64 else GFMarkovLM_V6
        model = cls(d=d, layers=len(payload["coef"]),
                    fan_in=int(payload["coef"][0].shape[1]),
                    lags=tuple(int(x) for x in payload["lags"]))
        # Same version rule as ui_server._load_agent: the checkpoint meta
        # names the format, and a checkpoint without one is v1 by definition.
        # Getting this wrong would silently measure a v1 model on v2 bytes.
        #
        # V6-class hard exports do NOT nest their metadata: export_hard()
        # takes no `meta` argument, so export_hard_with_meta promotes the
        # metadata to the TOP level of the payload. There is no raw["meta"]
        # at all. Read the top level first, then fall back to the nested
        # meta (which V7-class exports do use).
        meta = raw.get("meta") if isinstance(raw, dict) else None
        if not isinstance(meta, dict):
            meta = raw if isinstance(raw, dict) else {}
        ckpt_ver = meta.get("transcript")
        version = fmt.normalize(ckpt_ver if ckpt_ver is not None else "v1")
        agent = Agent(model, torch.device("cpu"), greedy=True, version=version)
        agent.hard = hard_to_cpu(payload)
        agent.model.eval()
        return agent


Policy = Callable[[Board, List[int]], int]


def _model_policy(agent: Agent, hard: bool = True, eps: float = 0.0,
                  rng: Optional[random.Random] = None) -> Policy:
    """Wrap an Agent as the (board, moves) -> column callable opponents use.

    Uses the HARDENED integer forward. The checkpoints on disk are
    export_hard() payloads with no float state, so there is no soft path to
    fall back to -- and hard is what actually ships, so it is also the honest
    thing to measure.

    With ``eps > 0`` the model plays epsilon-greedy: on a fraction of plies it
    picks uniformly among the legal columns instead of the argmax. This is the
    only source of variation in a greedy-vs-deterministic matchup, so it is
    what makes N games carry more than one bit of information. It also answers
    the question the strength gate actually asks -- "how good are its tactics"
    -- better than replaying a single line two hundred times.
    """

    def policy(board: Board, moves: List[int]) -> int:
        if eps > 0.0 and rng is not None and rng.random() < eps:
            legal = [c for c in range(COLS) if board.cell(5, c) == 0]
            if legal:
                return rng.choice(legal)
        return int(agent.choose(board, moves, use_hard=hard))

    return policy


def play_one(model_pol: Policy, opp_pol: Policy, model_is_p1: bool,
             ) -> Tuple[Optional[int], Tuple[int, ...]]:
    """Play one game. Returns ``(1/0/None, move_sequence)``.

    The move sequence comes back so the caller can count DISTINCT games: two
    runs that produce the same sequence are the same game and must not be
    counted as two independent observations.

    The old signature took a ``seed`` and silently discarded it -- that is what
    let a deterministic matchup masquerade as an N-game sample.
    """
    b = Board()
    moves: List[int] = []

    while not is_terminal(b):
        model_side = P1 if model_is_p1 else P2
        pol = model_pol if b.turn == model_side else opp_pol
        col = pol(b, moves)

        if not (0 <= col < COLS) or b.heights[col] >= 6:
            raise ValueError(
                f"illegal move col={col} heights={b.heights} moves={moves}")

        # apply_move flips self.turn, so is_win() must be called after.
        b.apply_move(col)
        moves.append(col)

        winner = b.is_win()
        if winner:
            model_side = P1 if model_is_p1 else P2
            return (1 if winner == model_side else 0), tuple(moves)

    return None, tuple(moves)


def _explore(pol: Policy, eps: float,
             rng: Optional[random.Random]) -> Policy:
    """Wrap any policy in epsilon-greedy noise.

    Applied to the OPPONENT, never to A by default. Blundering A is not a
    measurement of A -- it measures A plus the blunders we inflicted on it. The
    opponent is the thing whose strength we already believe (it is fixed and
    calibrated), so injecting the sampling noise there costs nothing and keeps A
    playing exactly the greedy line that ships in production.
    """

    if eps <= 0.0 or rng is None:
        return pol

    def noisy(board: Board, moves: List[int]) -> int:
        if rng.random() < eps:
            legal = [c for c in range(COLS) if board.cell(5, c) == 0]
            if legal:
                return rng.choice(legal)
        return pol(board, moves)

    return noisy


def duel(a_pol: Policy, b_pol: Policy, games: int, seed: int = 0,
         ) -> dict:
    """Play `games` games, alternating who is P1. Returns a result dict."""
    a_w = a_l = draws = 0
    a_p1 = a_p2 = a_p2_wins = a_p1_wins = 0
    seen = set()
    lengths = []

    for g in range(games):
        a_is_p1 = (g % 2 == 0)
        r, seq = play_one(a_pol, b_pol, a_is_p1)
        seen.add(seq)
        lengths.append(len(seq))
        if r is None:
            draws += 1
        elif r == 1:
            a_w += 1
            if a_is_p1:
                a_p1_wins += 1
            else:
                a_p2_wins += 1
        else:
            a_l += 1
        if a_is_p1:
            a_p1 += 1
        else:
            a_p2 += 1

    decided = a_w + a_l
    return {
        "games": games,
        "unique_games": len(seen),
        "A_wins": a_w,
        "B_wins": a_l,
        "draws": draws,
        "A_win_rate": a_w / games if games else 0.0,
        "A_score_pct": (a_w + 0.5 * draws) / games * 100 if games else 0.0,
        "A_as_P1": f"{a_p1_wins}/{a_p1}",
        "A_as_P2": f"{a_p2_wins}/{a_p2}",
        "mean_plies": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (float state)")
    ap.add_argument("--b", help="checkpoint B; omit to duel A vs an opponent")
    ap.add_argument("--opponent", default="heuristic",
                    help="opponent policy name when --b is omitted")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soft", action="store_true",
                    help="use the float forward for A instead of the hard one")
    ap.add_argument("--eps", type=float, default=0.0,
                    help="epsilon-greedy rate for the OPPONENT (default 0.0). "
                         "This is the only source of variation: A always "
                         "plays its greedy shipping line. Use eps>0 or a "
                         "deterministic matchup will replay one game per "
                         "seat instead of sampling.")
    ap.add_argument("--eps-a", type=float, default=0.0,
                    help="epsilon-greedy rate for A itself; hand handicaps A, "
                         "so it measures A+noise rather than A. Off by default.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # _load_agent rebuilds the exact GFMarkovLM from the payload's own
    # self-describing arch, then loads the integer tables. hard_forward is a
    # CPU path, so the duel always runs on CPU -- that also keeps it from
    # competing with an in-flight training run for the GPU.
    agent_a = _load_hard_agent(args.a)
    pol_a = _model_policy(agent_a, hard=not args.soft, eps=args.eps_a, rng=rng)

    if args.b:
        agent_b = _load_hard_agent(args.b)
        pol_b = _model_policy(agent_b)
        label_b = f"B={args.b.replace(chr(92), '/').split('/')[-1]}"
    else:
        pol_b = get_opponent(args.opponent)
        label_b = f"opponent={args.opponent}"

    pol_b = _explore(pol_b, args.eps, rng)

    label_a = f"A={args.a.replace(chr(92), '/').split('/')[-1]}"

    print(f"A: {label_a}")
    print(f"B: {label_b}")
    print(f"epsilon     : opponent={args.eps} A={args.eps_a}")
    print()

    res = duel(pol_a, pol_b, args.games, args.seed)

    print(f"games        : {res['games']}")
    print(f"unique games : {res['unique_games']}   <-- distinct move sequences")
    print(f"A wins       : {res['A_wins']}")
    print(f"B wins       : {res['B_wins']}")
    print(f"draws        : {res['draws']}")
    print(f"A score      : {res['A_score_pct']:.1f}%  (win + 0.5*draw)")
    print(f"A as P1      : {res['A_as_P1']}")
    print(f"A as P2      : {res['A_as_P2']}")
    print(f"mean plies   : {res['mean_plies']:.1f}")
    print()

    uniq = res["unique_games"]
    if uniq <= 2:
        print(f"!! WARNING: only {uniq} distinct game(s) out of {res['games']} "
              f"requested.")
        print("!! Both sides are deterministic, so this is a REPLAY, not a "
              "sample -- the")
        print("!! record above carries the information of a single game. The "
              "only honest")
        print("!! reading is per-seat: 'wins the game where it moves first, "
              "loses the game")
        print("!! where it moves second.' Re-run with --eps 0.15 to sample.")
        print()

        p1_w, p1_n = (int(x) for x in res["A_as_P1"].split("/"))
        p2_w, p2_n = (int(x) for x in res["A_as_P2"].split("/"))
        if p1_n and p2_n:
            rate = (p1_w / p1_n + p2_w / p2_n) / 2 * 100
            print(f"VERDICT (per-seat, the only honest reading here): A wins "
                  f"{rate:.0f}% of the two seats it can reach.")
            return
    elif uniq < res["games"]:
        # Some duplicates are legitimate: once a game ends, several epsilon
        # draws can still land on the same line. Flag it, but still report a
        # verdict -- the sample is mostly real.
        print(f"note: {uniq} distinct games across {res['games']} runs "
              f"(epsilon draws can retrace a line).")
        print()

    # Connect Four cannot tie in wins-vs-losses by chance; anything past 60%
    # is a real difference. Below 55% is inside two-sigma noise at n=200.
    verdict = ("A is better" if res["A_score_pct"] >= 60
               else "B is better" if res["A_score_pct"] <= 40
               else "too close to call")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()