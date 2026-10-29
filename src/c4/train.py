"""Self-play training for the Connect-Four GF(256) Markov LM.

Recipe is locked to the V6 configuration that worked on tinyshakespeare:
  d=32  nlayers=3  fan_in=3  steps=20000
  hard_start=0     hard_ramp=4000  hard_w=0.5
  ent_max=2.0  perm_w=1.0

  hard_start used to be 12000. That was wrong: the artifact that ships is
  `export_hard()` -- the one-hot forward -- so for the first 12000 steps the
  shipping artifact was an UNTRAINED approximation of the soft function.
  Measured over V23's checkpoints, the exported model was provably
  board-blind before step 12000 (board-change logit delta exactly 0.0000,
  hard column 3 on 120/120 positions) and only read the board afterwards.
  The default is now 0: train the path you ship, from step 0.

The loop alternates two phases:
  1. SELF-PLAY -- the current model plays games against itself. Decided
     games (winner != draw) are appended to a winners-only corpus. The
     model samples with temperature so early games have variety; later
     games use a shrinking temperature that anneals toward greedy.
  2. FIT -- a batch of prompt/next-move pairs drawn from the corpus is
     trained with the V10 loss: byte-level CE masked to MOVE bytes only,
     with extra weight on winner-side moves, plus the ramped hard (STE)
     CE on the same mask, plus the entropy + permutation regularizers.

The byte stream is dominated by per-ply headers (h0..h6, turn, last) and
trailing newline padding: only ~12% of targets are actual decisions. The
V9 objective cross-entropy over ALL 256 byte classes pushes the model to
fit the trivially predictable header first and treats every move as
equiprobable over an arbitrary alphabet of 256 values. V10 instead spends
the gradient on what the agent is actually asked to do at serve time --
pick one of the seven legal columns -- and discards the gradient that
went to the loser's moves, so the policy converges toward "what plays to
win" instead of "what did the eventual loser do next".

At step 0 there is no corpus, so the corpus is seeded with random play and
the model only ever sees its own games from the very first update -- there
is no supervised warm start from a human dataset.

Checkpoints are integer exports (`model.export_hard()`), the same format
`play.py --hard` consumes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .board import Board, is_terminal
from .corpus import (Game, games_to_lines, play_games, play_mixed_game,
                     play_mixed_games, stats, winners_only)
from .corpus_aug import encode_games, game_dense
from .corpus_quality import quality_gate
from .corpus_behavior import behavior_gate_games
from .teacher_corpus import (teacher_batch, teacher_batch_with_roles,
                            teacher_seed_games)
from .opponents import OPPONENTS
from . import fmt
from .transcript import COL_BYTE_OFFSET, COL_SLICE
from .fmt import PLY_LEN, MOVE_OFFSET   # 10 in BOTH formats; stride arithmetic is safe
# v2 header length (9 = PLY_LEN - 1): the byte count of a ply's board header.
from .transcript_v2 import HEADER_LEN
from .movemask import move_mask_from_starts, validate_stream
from .move_loss import (build_move_targets, header_legality_from_x,
                         header_visibility)
from ..gf256_markov_lm_v6 import GFMarkovLM
from ..gf256_markov_lm import GFMarkovLM as GFMarkovLM_V7


# The opponent whose ladder net-wins is the PRIMARY promotion key. The
# value is only meaningful when paired with the protocol below, so the
# protocol is stored in every checkpoint's metadata and checked on resume.
LADDER_PRIMARY_OPP = "heuristic"
LADDER_PROTOCOL_VERSION = 2  # bump when the measurement protocol changes


# Upper bound on the mirror-equivariance penalty.  It is a Jeffreys
# divergence between two distributions over the SAME common legal support,
# so it cannot exceed 2*log(7) ~ 3.892.  A value far above that means the
# reflected target was built from an already-masked log-softmax, which
# degenerates the KL term into p*1e9 -- the "regulariser" then becomes a
# constant pull rather than a constraint.  That shipped once at 4.25e7 and
# the run trained on a garbage gradient without ever erroring, so the
# trainer now REFUSES to continue rather than silently optimising noise.
SYM_PENALTY_MAX = 2.0 * math.log(7.0) + 1e-6


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------

def _board_sensitivity(model, device, version=None, corpus_rows=None,
                       n_positions: int = 12, hard=None):
    """Does the EXPORTED (hard/one-hot) model react to a change of the board?

    The shipping artifact is `export_hard()`. If the one-hot forward was
    never trained, the export is a constant-column policy that ignores the
    stones entirely -- it scores against weak opponents and loses every
    real game.

    Test: take real positions, and for each build a DIFFERENT position
    whose move history differs in its last move, so the board differs and
    both histories stay legal. Measure how far the 7 column logits move.
    A model that reads the board cannot return exactly 0; a board-blind
    one returns exactly 0.

    Returns the mean max-abs logit delta, or None if the test could not
    be run (no usable positions) -- the caller must then NOT assume
    anything, and must not treat None as "blind" either.
    """
    import copy as _copy
    from .board import Board, Outcome
    from .agent import Agent, hard_to_cpu

    try:
        if hard is None:
            hard = model.export_hard()
        hard = hard_to_cpu(hard)
    except Exception:
        return None

    # ---- gather candidate move histories -------------------------------
    lines = []
    for s in (corpus_rows or [])[:n_positions * 8]:
        nply = len(s) // PLY_LEN
        if nply < 3:
            continue
        mv = []
        ok = True
        for p in range(nply):
            mc = s[p * PLY_LEN + MOVE_OFFSET] - COL_BYTE_OFFSET
            if mc < 0 or mc > 6:
                ok = False
                break
            mv.append(mc)
        if ok and len(mv) >= 3:
            lines.append(mv)
        if len(lines) >= n_positions * 3:
            break

    if not lines:
        # Synthetic openings: a board-blind model is blind on these too.
        lines = [[], [3], [3, 3], [3, 2], [2, 3], [3, 2, 3], [3, 2, 4],
                 [2, 3, 4], [3, 4, 3], [2, 3, 2]]

    # ---- build (board, history) pairs ----------------------------------
    pairs = []
    for mv in lines:
        try:
            b = Board()
            for mc in mv:
                if mc not in b.valid_actions():
                    raise ValueError("illegal")
                b.apply_move(mc)
        except Exception:
            continue
        if b.outcome() != Outcome.ONGOING:
            continue
        legal = b.valid_actions()
        if len(legal) < 2:
            continue
        # A DIFFERENT position: drop the last move of the history and play a
        # different legal column instead. Both histories are then legal and
        # the boards differ by one stone.
        mv_a = list(mv)
        if not mv:
            # Empty board: any two legal columns give two different boards.
            mv_b = [legal[0]]
            mv_a = [legal[1]]
        else:
            _alt = [c for c in legal if c != mv[-1]]
            if not _alt:
                continue
            mv_b = list(mv[:-1]) + [_alt[0]]
        try:
            b2 = Board()
            for mc in mv_b:
                if mc not in b2.valid_actions():
                    raise ValueError("illegal")
                b2.apply_move(mc)
        except Exception:
            continue
        if b2.outcome() != Outcome.ONGOING:
            continue
        pairs.append((b, mv_a, b2, mv_b))
        if len(pairs) >= n_positions:
            break

    if not pairs:
        return None

    # ---- measure -------------------------------------------------------
    deltas = []
    try:
        ag = Agent(model, device, greedy=True, temperature=0.0,
                   version=version)
    except Exception:
        return None
    for b, mv_a, b2, mv_b in pairs:
        try:
            l1 = ag.hard_column_logits(b, mv_a, hard=hard)
            l2 = ag.hard_column_logits(b2, mv_b, hard=hard)
        except Exception:
            continue
        deltas.append(float((l1 - l2).abs().max()))
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _eval_opponents(model, device, kinds: dict, n_games: int,
                    rng: random.Random, temperature: float,
                    version: str | None = None,
                    explore: float = 0.35) -> dict:
    """Play `n_games` model-vs-opponent for each opponent in `kinds`.

    Returns ``{name: {"wins", "losses", "draws", "games", "distinct"}}``.

    WHY `distinct` AND WHY `explore`
    --------------------------------
    The old version reported only nominal W/L, and that number was measured
    to be meaningless: against the deterministic heuristic, 48 nominal games
    collapsed to 2 distinct full sequences and a single opening ([2,3,2] in
    48/48). "W47-L1" was one memorised line counted 47 times. Any metric whose
    games can repeat can be inflated without the model improving at all.

    So the OPPONENT is randomised (`explore` = per-ply probability of a
    uniform random move) and every finished game is hashed into a set. A
    model that only knows one line shows distinct << games; a genuinely
    strong model shows distinct ~ games. Read the pair together:
    high win-rate with low distinct means memorisation, not strength.

    The MODEL is never handicapped here -- it plays its greedy shipping line
    (always greedy now; it was `greedy = temperature <= 1e-3`, which with
    the default `--eval-temp 0.2` meant FALSE, so every ladder number in
    every log from V15 onward graded a SAMPLED policy instead of the
    shipping one). All exploration lives on the opponent, because an
    exploration-ε model measures the sampler, not the policy.
    """
    from .agent import Agent
    from .board import Outcome

    results = {}
    for name in kinds:
        opp = OPPONENTS[name]
        wins = losses = draws = 0
        seqs = set()
        for colour_swap in (False, True):
            agent = Agent(model, device,
                          greedy=True,
                          temperature=temperature,
                          rng=rng,
                          version=version)
            for _ in range(n_games):
                board = Board()
                moves: List[int] = []
                done = False
                while not done:
                    mine = (board.turn == 1) != colour_swap
                    if mine:
                        # use_hard=True: the integerised export IS the shipping
                        # artifact, so the ladder must measure THAT. The old
                        # call used the soft path and silently graded a model
                        # nobody will ever play.
                        c = agent.choose(board, moves, use_hard=True)
                    elif explore > 0 and rng.random() < explore:
                        c = rng.choice(board.valid_actions())
                    else:
                        c = opp(board, moves)
                    moves.append(c)
                    if c not in board.valid_actions():
                        # safety
                        break
                    board.apply_move(c)
                    o = board.outcome()
                    if o == Outcome.ONGOING:
                        continue
                    if o == Outcome.DRAW:
                        draws += 1
                    elif (o == Outcome.WIN_P1) != colour_swap:
                        wins += 1
                    else:
                        losses += 1
                    seqs.add(tuple(moves))
                    done = True
        results[name] = {"wins": wins, "losses": losses,
                         "draws": draws, "games": wins + losses + draws,
                         "distinct": len(seqs)}
    return results


def _play_one_snapshot(model, opp_name: str, step: int,
                        device: str, rng: random.Random,
                        version: str | None = None) -> str:
    """Play one agent-vs-opponent game and render the board every ply.

    Returned string is suitable for ``append`` to snapshots.txt: a header
    line plus an ASCII board per ply. The text format mirrors what
    ``play.py`` prints so the user can read it directly in a log viewer.

    A broken snapshot never crashes training -- if anything goes wrong we
    return a one-line "snapshot failed: ..." string instead.
    """
    try:
        from .agent import Agent
        from .board import Outcome

        agent = Agent(model, device, greedy=True, temperature=0.0, rng=rng,
                              version=version)
        opp = OPPONENTS[opp_name]
        out: List[str] = []

        def render(b: Board, last_move: int | None, who: str | None,
                   ply: int) -> str:
            glyph = {0: ".", 1: "X", 2: "O"}
            rows = []
            for r in range(5, -1, -1):
                row = []
                for c in range(7):
                    mark = glyph[b.cell(r, c)]
                    # mark the piece that just dropped (lowercase)
                    if last_move is not None and 0 <= b.heights[last_move] - 1 == r \
                            and c == last_move:
                        v = b.cell(r, c)
                        mark = "x" if v == 1 else "o"
                    row.append(mark)
                rows.append(" ".join(row))
            rows.append("0 1 2 3 4 5 6")
            header = f"  ply {ply:2d}: "
            if last_move is not None:
                header += "model" if who == "agent" else opp_name
                header += f" -> col {last_move}"
            return header + "\n" + "\n".join(rows)

        board = Board()
        moves: List[int] = []
        out.append(f"=== step {step}  model-vs-{opp_name}  "
                   f"{'agent=P1' if board.turn == 1 else 'agent=P2'} ===")
        while True:
            who = "agent" if board.turn == 1 else opp_name
            col = agent.choose(board, moves) if who == "agent" else opp(board, moves)
            board.apply_move(col)
            moves.append(col)
            out.append(render(board, col, who, len(moves)))
            o = board.outcome()
            if o == Outcome.ONGOING:
                continue
            if o == Outcome.DRAW:
                out.append(">>> DRAW")
            else:
                winner = "model" if who == "agent" else opp_name
                out.append(f">>> {'P1' if o == 1 else 'P2'} WINS  ({winner})")
            break
        return "\n".join(out)
    except Exception as e:  # pragma: no cover - defensive
        return f"[snapshot failed at step {step}: {type(e).__name__}: {e}]"


# --------------------------------------------------------------------------
# self-play
# --------------------------------------------------------------------------

def self_play_games(model, device, n_games: int, rng: random.Random,
                    temperature: float) -> List[Game]:
    """Play `n_games` games of the model against itself.

    Each ply is chosen by sampling from the model's column logits, masked
    to legal columns. Draws are discarded by the caller (`winners_only`).
    """
    from .agent import prompt_tokens

    games: List[Game] = []
    model.eval()
    for _ in range(n_games):
        board = Board()
        moves: List[int] = []
        while True:
            legal = board.valid_actions()
            toks = prompt_tokens(moves).to(device).unsqueeze(0)   # (1,T)
            with torch.no_grad():
                logits = model(toks)[0, -1, COL_SLICE]             # (7,)
            logits = logits.float().cpu() / max(temperature, 1e-6)
            mask = torch.full((7,), float("-inf"))
            for c in legal:
                mask[c] = 0.0
            probs = F.softmax(logits + mask, dim=-1)
            col = int(torch.multinomial(probs, 1).item())
            moves.append(col)
            board.apply_move(col)
            # is_win() returns the winning player number, 0 = no win. A draw
            # is a full board with no winner -> recorded as winner 0 and
            # dropped later by winners_only().
            w = board.is_win()
            if is_terminal(board):
                break
        games.append((moves, w))
    model.train()
    return games


def mixed_opponent_games(model, device, n_games: int, rng: random.Random,
                         temperature: float) -> List[Game]:
    """Self-play that mixes random + heuristic + lookahead + minimax on
    *both* sides. Half the games the model is P1, half P2.

    This is the right corpus for Connect Four:
        - winners of these games are winning because they played well
        - corpora full of games between weak players teach the model
          "what legal moves look like" but not "what wins games"
        - the model still moves are the policy targets (preserves the
          imitation setup); opponents just raise the quality bar of
          the moves the *opponent* makes, so game trajectories are
          shorter and more decisive.
    """
    from .agent import prompt_tokens

    # Symmetric weights. Force a strong baseline so most games terminate
    # in a real win, not in 42-ply drawn-out randomness.
    weights = {
        "random":    0.15,
        "heuristic":  0.30,
        "lookahead":  0.30,
        "minimax":    0.25,
    }
    policies = dict(OPPONENTS)
    # `model` is the network trained so far; wrap it as a Policy for P1 only
    # (callers decide which side the model is).
    def model_policy(b: Board, moves: List[int]) -> int:
        legal = b.valid_actions()
        toks = prompt_tokens(moves).to(device).unsqueeze(0)
        with torch.no_grad():
            logits = model(toks)[0, -1, COL_SLICE]
        logits = logits.float().cpu() / max(temperature, 1e-6)
        mask = torch.full((7,), float("-inf"))
        for c in legal:
            mask[c] = 0.0
        probs = F.softmax(logits + mask, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    games: List[Game] = []
    model.eval()
    for i in range(n_games):
        # alternate: even i -> model=P1; odd i -> model=P2
        p1_weights = weights if (i % 2 == 0) else {"heuristic": 1.0}
        p2_weights = {"heuristic": 1.0} if (i % 2 == 0) else weights
        # Insert model as a "model" entry in the dict; build a tmp copy each game
        # to avoid mutating the shared OPPONENTS dict.
        local_policies = {**policies, "model": model_policy}
        local_p1 = dict(p1_weights)
        # the "model" entry is used by the caller manually below; we don't want
        # to add it to the random weights dict because that would compete with
        # the opponents for the model slot. Instead, decide the model's side
        # up-front and route explicitly.
        if i % 2 == 0:
            # Model is P1. Pick one P2 opponent from the weight distribution.
            games.append(play_mixed_game(rng, model_policy,
                                         _sample_opp(rng, weights, policies)))
        else:
            # Model is P2. Pick one P1 opponent from the weight distribution.
            games.append(play_mixed_game(rng,
                                         _sample_opp(rng, weights, policies),
                                         model_policy))
    model.train()
    return games


def _sample_opp(rng: random.Random, weights: dict, policies: dict):
    """Sample one opponent Policy using the weight distribution."""
    names = list(weights.keys())
    ws = [weights[n] for n in names]
    s = sum(ws) or 1.0
    ws = [w / s for w in ws]
    r = rng.random()
    acc = 0.0
    for n, w in zip(names, ws):
        acc += w
        if r < acc:
            return policies[n]
    return policies[names[-1]]


def random_seed_games(n_games: int, rng: random.Random) -> List[Game]:
    """Uniform-random play, used only to seed the corpus at step 0."""
    games: List[Game] = []
    for _ in range(n_games):
        board = Board()
        moves: List[int] = []
        while True:
            c = rng.choice(board.valid_actions())
            moves.append(c)
            board.apply_move(c)
            w = board.is_win()
            if is_terminal(board):
                break
        games.append((moves, w))
    return games


# --------------------------------------------------------------------------
# corpus -> training tensors
# --------------------------------------------------------------------------

def games_to_examples(games: Sequence[Game],
                      version: str | None = None) -> List[bytes]:
    """Each decided game becomes ONE training sequence: its full transcript.

    The transcript already encodes, at every ply header, the state (heights,
    last move, side). The loss is next-token prediction over the whole
    sequence, so the model is trained to predict every ply's move byte from
    its own header plus history -- the same inference-time operation.

    ENCODER CONTRACT: this must be the DENSE layout, byte-identical to the
    rows in corpus.txt and to what `agent.py` feeds at serving time. The
    earlier `encode_game` call here produced the SPACED legacy layout
    (11 bytes/ply, newline after every ply). That was invisible while the
    resume path copied raw corpus lines straight into `examples`, because
    nothing re-encoded them -- but the first time a Game record had to be
    re-encoded (teacher append, quality gate) the sequences stopped matching
    the tags and the serving layout. `action_consistent_examples` already
    used `game_dense`, so the two disagreed by one byte per ply.

    NOTE: outcome is intentionally dropped here. See
    `action_consistent_examples` below for the outcome-aware loss.
    """
    return [game_dense(moves, version) for moves, _ in games]



def action_consistent_examples(
        games: Sequence[Game],
        role_masks: Optional[Sequence[Sequence[bool]]] = None,
        version: str | None = None,
) -> List[torch.Tensor]:
    """Per-sequence winner mask parallel to a `games_to_examples` row.

    The text-prediction loss keeps the same `y` byte slab but only the
    positions where that byte is a column decision (0x41..0x47) need to
    be weighted -- headers and the trailing newline carry no move
    information, and the trainer already masks them out via
    `_move_byte_mask`. A 1.0 mark on byte k means "the player who played
    that column was the game's winner" and 0.0 means the loser. Both
    plies of a game are tagged this way: the same Game record yields
    two complementary masks, one for each side.
    """
    # 0 in the byte means "P1 to move" (mirroring the transcript), so the
    # winner-side tag for ply k is parity: even k -> P1 -> winner-1 == 0
    # iff winner == 1.
    from .transcript import PLY_LEN  # local: avoid cycle on first import
    from .corpus_aug import game_dense  # local: same layout the trainer loads

    out_masks: List[torch.Tensor] = []
    for gi, (moves, winner) in enumerate(games):
        # MUST use the same encoder the corpus loader uses. `encode_game` is
        # the SPACED layout (9 header + move + newline = 11 bytes/ply); the
        # corpus on disk is DENSE (10 bytes/ply, no interior newline). Mixing
        # them makes the mask drift one byte per ply.
        seq = game_dense(moves, version)
        L = len(seq)
        assert L == len(moves) * PLY_LEN, (
            f"dense encode gave {L} bytes for {len(moves)} plies, "
            f"expected {len(moves) * PLY_LEN}")
        winner_mask = torch.zeros(L - 1, dtype=torch.float32)
        # `role_masks` (optional) overrides the winner rule per ply. Teacher
        # games supply one: the heuristic seat wins ~half of solver-vs-
        # heuristic games (measured W10 L10), so winner-only tagging would
        # label the heuristic's moves as the demonstration. See
        # `teacher_batch_with_roles`.
        roles = role_masks[gi] if role_masks is not None else None
        for k in range(len(moves)):
            # Dense layout: ply k's 10 bytes start at k*PLY_LEN; the move byte
            # is the last of them (MOVE_OFFSET = 9). Next-byte
            # prediction shifts x -> y by -1, so the same byte sits at
            # k*PLY_LEN + PLY_LEN - 2 in the target slab.
            tgt = k * PLY_LEN + (PLY_LEN - 2)
            if tgt >= L - 1:
                continue                              # last move: no next target
            if roles is not None:
                winner_mask[tgt] = 1.0 if roles[k] else 0.0
            else:
                player = (k % 2) + 1
                winner_mask[tgt] = 1.0 if player == winner else 0.0
        out_masks.append(winner_mask)
    return out_masks


def build_batch(examples: Sequence[bytes], B: int, device: torch.device,
                rng: np.random.Generator, ctx: int,
                is_teacher: Optional[Sequence[bool]] = None,
                max_legacy_frac: float = 0.0,
                snap_to_end: bool = False):
    """Sample B (x, y) next-token pairs of length <= ctx+1 from the corpus.

    Returns (x, y, chosen) where `chosen` is the numpy index array so the
    caller can mirror the same windowing onto the outcome tags without a
    second stochastic draw.

    `is_teacher` / `max_legacy_frac` implement the V12 teacher-dominant mix.
    Legacy rows stay resident in `examples` for the whole run while teacher
    rows only accumulate per growth cycle, so uniform sampling keeps legacy
    at ~88% of every batch forever (measured on V11.1). When `is_teacher` is
    given, draw at most `max_legacy_frac` of the batch from legacy rows --
    proportionally re-weighting, never dropping rows from the corpus file.

    Returns `(x, y, chosen, starts)`. `starts[k]` is the byte offset inside
    `examples[chosen[k]]` that window k was taken from -- build_batch is the
    ONLY place a window start is drawn, so the caller must thread `starts`
    into `build_outcome_batch` rather than letting it re-roll. (It used to:
    it drew from the *global* np.random while this draws from the injected
    `rng`, so the two were independent and the role weights landed on the
    wrong plies ~47% of the time -- see tg_tagcheck2.py.)
    """
    T = ctx + 1
    L = ctx + 1
    starts = np.zeros(B, dtype=np.int64)
    if is_teacher is not None and max_legacy_frac < 1.0:
        t_idx = np.fromiter((i for i in range(len(examples))
                             if is_teacher[i]), dtype=np.int64)
        l_idx = np.fromiter((i for i in range(len(examples))
                             if not is_teacher[i]), dtype=np.int64)
        n_t, n_l = len(t_idx), len(l_idx)
        if n_t and n_l:
            # sample the batch with the requested legacy share, capped by
            # what each pool can actually supply this step.
            want_l = min(n_l, int(round(B * max_legacy_frac)))
            want_t = min(n_t, B - want_l)
            # if teacher ran out, give the slack back to legacy
            want_l = min(n_l, B - want_t)
            picks = np.concatenate([
                t_idx[rng.integers(0, n_t, size=want_t)],
                l_idx[rng.integers(0, n_l, size=want_l)],
            ])
            rng.shuffle(picks)
            chosen = picks[:B]
        else:
            # only one pool exists -- fall back to uniform
            chosen = rng.integers(0, len(examples), size=B)
    else:
        chosen = rng.integers(0, len(examples), size=B)

    # ---- guarantee the labelled ply is inside the window --------------
    # Imported rows carry a soft target only on their FINAL ply. With
    # ctx=192 and sequences of ~250 bytes only 3.9% of uniformly-drawn
    # windows contained that ply, so 96% of the imported supervision would
    # have been silently discarded and the run would look like "the data
    # does not help". Rather than enlarge the model's context, snap the
    # window to END at the sequence end whenever the sequence is longer than
    # the window -- the labelled ply is then always the final position of the
    # batch, at zero cost in context length.
    xs, ys = [], []
    for k, i in enumerate(chosen):
        seq = examples[i]
        if len(seq) <= 1:
            xs.append(seq + b"\n")
            ys.append(seq + b"\n")
            continue
        # random window of length L ending at or before the sequence end
        if len(seq) >= L:
            start = int(rng.integers(0, len(seq) - L + 1))
        else:
            start = 0
        if snap_to_end:
            # pin the window to the END of the sequence so the labelled final
            # ply is always inside it (see the note above build_batch)
            start = max(0, len(seq) - L)
        starts[k] = start
        window = seq[start:start + L]
        window = window + b"\n" * (L - len(window))
        arr = np.frombuffer(window, dtype=np.uint8)
        xs.append(arr[:-1])
        ys.append(arr[1:])
    x = torch.tensor(np.stack(xs).astype(np.int64), device=device)
    y = torch.tensor(np.stack(ys).astype(np.int64), device=device)
    return x, y, chosen, starts


def build_outcome_batch(outcome_tags: Sequence[torch.Tensor],
                        chosen: np.ndarray, B: int, ctx: int,
                        device: torch.device,
                        starts: Optional[np.ndarray] = None) -> torch.Tensor:
    """Pair a `(B,T-1)` winner-tag tensor to the SAME windows as `build_batch`.

    `starts` must be the array `build_batch` returned for this batch. It is
    the single source of truth for where each window began: this function
    must never draw a start of its own, or the role weights land on the
    wrong plies. That regression is exactly what `tg_tagcheck2.py` measures
    (46.8% of move bytes mis-weighted before the fix).

    `starts` is a byte offset into `examples[i]`, which is DENSE-encoded
    (PLY_LEN bytes per ply). Tag vectors are indexed in the same TARGET byte
    space that `y` occupies, i.e. tag[t] describes y[t] = seq[t+1]. A window
    starting at `starts[k]` therefore contributes tag[start : start + T].

    If `starts` is None (legacy callers that pass only `chosen`), fall back
    to the old re-roll so behaviour degrades loudly rather than silently
    changing shape; those callers should migrate.
    """
    T = ctx
    L = ctx + 1
    out = np.zeros((B, T), dtype=np.float32)
    for k, i in enumerate(chosen):
        tag = outcome_tags[int(i)]
        n_avail = tag.numel()
        if n_avail == 0:
            continue
        if starts is None:
            # Deprecated path: independent draw (see docstring).
            start = (0 if n_avail + 1 < L
                     else int(np.random.randint(0, n_avail + 1 - L + 1)))
        else:
            # Same window build_batch actually used.
            start = int(starts[k])
            if start + T > n_avail:
                start = max(0, n_avail - T)
        sub = tag[start:start + T].numpy()       # (T,) aligned with y[:, k]
        out[k, :sub.shape[0]] = sub
    return torch.tensor(out, device=device)



# --------------------------------------------------------------------------
# evaluation: how good is the model as a Connect-Four player?
# --------------------------------------------------------------------------

def evaluate_play(model, device, n_games: int, rng: random.Random,
                  temperature: float = 0.7, reference_centre: bool = True):
    """Play the model against a trivial centre-column opponent.

    This is a *sanity* metric, not a strength rating. It tells us whether
    the model has learned to block an immediate win and to build its own --
    i.e. whether the self-play signal reached the weights.

    BUG (fixed 2026-10-23): the model used to be played with
    ``torch.multinomial`` ALWAYS, no matter what ``temperature`` was. A
    comment above the call site claimed "evaluation play uses its OWN
    temperature... A next-byte predictor deployed in the UI is played
    greedily (argmax over legal columns), so that is how we must measure
    it" -- but the code sampled anyway. Every "vs_ref" number in every
    training log from V15 onward therefore graded a STOCHASTIC policy, not
    the greedy line that actually ships. That is the same class of bug as
    the zero move loss: the measurement did not measure the thing it
    claimed to measure.

    Now: ``temperature <= 1e-3`` means GREEDY (argmax over the legal
    columns), which is the shipping behaviour. A higher temperature keeps
    the old sampling behaviour for anyone who wants it.
    """
    from .agent import prompt_tokens

    wins = losses = draws = 0
    greedy = temperature <= 1e-3
    model.eval()
    for g in range(n_games):
        board = Board()
        moves: List[int] = []
        model_is_p1 = (g % 2 == 0)
        while True:
            legal = board.valid_actions()
            side = board.turn
            if (side == 1) == model_is_p1:
                toks = prompt_tokens(moves).to(device).unsqueeze(0)
                with torch.no_grad():
                    logits = model(toks)[0, -1, COL_SLICE].float().cpu()
                mask = torch.full((7,), float("-inf"))
                for c in legal:
                    mask[c] = 0.0
                if greedy:
                    masked = logits + mask
                    col = int(masked.argmax())
                else:
                    logits = logits / max(temperature, 1e-6)
                    probs = F.softmax(logits + mask, dim=-1)
                    col = int(torch.multinomial(probs, 1).item())
            else:
                # trivial opponent: centre if available, else random
                if reference_centre and 3 in legal:
                    col = 3
                else:
                    col = rng.choice(legal)
            moves.append(col)
            board.apply_move(col)
            o = board.is_win()
            if is_terminal(board):
                break
        model_won = (o == 1 and model_is_p1) or (o == 2 and not model_is_p1)
        if o == 0:
            draws += 1
        elif model_won:
            wins += 1
        else:
            losses += 1
    model.train()
    return {"wins": wins, "losses": losses, "draws": draws,
            "n": n_games}


def _move_byte_mask(y: torch.Tensor,
                    starts=None) -> torch.Tensor:
    """(B,T-1) bool: True where `y` holds a column DECISION.

    Two implementations, and the choice matters:

    * `starts` given -> PURELY POSITIONAL. Ply k's move byte lives at
      k*PLY_LEN + MOVE_OFFSET, and build_batch returns the byte offset each
      window began at, so the mask is arithmetic that cannot be fooled by a
      copied `last` byte, by 0x0A padding, or by a full column's 0xFF.

    * `starts` omitted -> the legacy VALUE test (0x41..0x47). It is kept only
      for diagnostics that score a raw (B,T-1) block with no window offsets
      (scripts/diag_*.py). Under v2 it also matches the `last` byte, which is
      why the trainer itself never uses it.

    Why this stopped being a value test: v1's header bytes were heights
    (0x30..0x36), so 0x41..0x47 meant "move byte or the previous move". v2
    moves columns to 0x81..0xFF but MUST keep the move range at 0x41..0x47
    (it is the argmax range, COL_SLICE). So position is now the only thing
    that separates the two, and a value test silently trains on a
    free-to-predict duplicate.
    """
    if starts is not None:
        return move_mask_from_starts(starts, y.shape[1], device=y.device)
    return (y >= COL_BYTE_OFFSET) & (y <= COL_BYTE_OFFSET + 6)


def capture_train_state(opt, rng, nprng):
    """Snapshot everything a resume needs beyond the weights.

    Returns a dict with the Adam moments and the three RNG states.  It is
    stored NEXT TO `{"state": ..., "meta": ...}` in the float checkpoint,
    never inside `meta`, because `meta` is also handed to
    `export_hard_with_meta` and a hard export must remain a lean integer
    table payload.

    Without this, every resume restarts Adam from empty moments and
    silently changes the sampling sequence, so a resumed run is NOT the
    same trajectory as an uninterrupted one.
    """
    return {
        "opt_state": opt.state_dict(),
        "rng_state": {
            "python": rng.getstate(),
            "numpy": nprng.bit_generator.state,
            "torch": torch.get_rng_state(),
        },
    }


def save_snapshot(model, run_dir, device, tag: str, meta: dict,
                  train_state: dict | None = None):
    """Write a periodic soft+hard weight pair, INDEPENDENT of the promotion gate.

    `train_state` (from `capture_train_state`) is attached to the SOFT
    checkpoint only.  The hard export stays a lean integer-table payload:
    it is the artifact that ships, and it cannot carry Adam moments.

    Why this exists
    ---------------
    `--save-every` used to trigger only the ladder evaluation and the text
    snapshot -- it never wrote weights. The sole weight writers were the
    promotion gate and the final save at `--steps`, so stopping a run
    mid-flight destroyed every step since the last promotion.

    That is not hypothetical. V17's strongest measured model was step 11000
    (W686 L109 = 86.1% vs random, the best ladder row in any run), it was
    never promoted because the gate keyed on held-out optimal-move accuracy,
    which sat at 32.9% for 7,000 steps -- and it existed ONLY in GPU memory.
    Stopping the process threw it away. `best_soft.pt` was still step 1500.

    These snapshots are deliberately NOT candidates for promotion; they exist
    so that "the model at step N" is always recoverable from disk, and so a
    human can pick the strongest one by hand after the fact.
    """
    was_training = model.training
    model.to("cpu")
    try:
        atomic_torch_save(export_hard_with_meta(model, meta=meta),
                          run_dir / "checkpoints" / f"{tag}_hard.pt")
        soft_payload = {"state": model.state_dict(), "meta": meta}
        if train_state:
            soft_payload.update(train_state)
        atomic_torch_save(soft_payload,
                          run_dir / "checkpoints" / f"{tag}_soft.pt")
    finally:
        # Always restore the device, including on exception. The promotion
        # gate had this same bug for a while: it moved the model to cpu to
        # save and never moved it back, so the NEXT training step died with
        # "indices should be either on cpu or on the same device".
        model.to(device)
        model.train(was_training)


# A dedicated Random instance for the atomic-save retry jitter.  It is
# deliberately NOT the global `random` module: the trainer seeds the global
# module and its state is part of the run's reproducibility contract, so
# consuming draws from it here would make the retry behaviour depend on
# how many times a checkpoint happened to collide with a reader.
_JITTER_RNG = random.Random(0xA70)


def atomic_torch_save(obj, path, retry_budget_s: float = 30.0):
    """Write a checkpoint so a reader NEVER sees a half-written file.

    `torch.save(obj, path)` truncates the target and then writes into it.
    A reader (the UI server loading `best_hard.pt`, or the promotion
    scorer) that opens the file during that window sees a truncated
    pickle and dies with an unhelpful "unexpected EOF" / "invalid load
    key" error.

    That matters here because the whole point of a long run is that the
    best-so-far checkpoint is playable WHILE the run continues. The
    standard fix is to write to a sibling temp file and rename it over
    the target: `os.replace` is atomic on POSIX and on Windows when the
    destination exists, so a reader sees either the old file or the new
    one, never a partial one.

    WINDOWS SPECIFIC
    ----------------
    On Windows, `os.replace()` fails with `PermissionError` (WinError 5)
    if ANY other process currently has the destination open WITHOUT
    `FILE_SHARE_DELETE`.  Python's `open()` does not use that flag, so a
    concurrent `torch.load()` of the same path blocks the rename for as
    long as it holds the handle.

    The retry budget therefore has to exceed the time a reader holds the
    handle.  A 203 MB float checkpoint takes roughly 0.5-2 s to load, and
    a reader that loops tightly (the atomicity test does) can hold the
    handle almost continuously -- the previous 2 s budget was exhausted
    and the write FAILED with PermissionError, which for a checkpoint
    publication is a lost checkpoint, not a cosmetic error.

    The budget is therefore 30 s by default, with jitter so a tight
    reader loop cannot starve the writer by always winning the race.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    # Retry the rename until the reader releases its handle.
    #
    # The jitter uses a DEDICATED Random instance, NOT the global
    # `random` module.  The global module is seeded by the trainer and its
    # state is part of the run's reproducibility contract; using it here
    # would consume a draw from the training stream and make the retry
    # behaviour depend on how many times a checkpoint happened to collide
    # with a reader.
    deadline = time.monotonic() + float(retry_budget_s)
    attempt = 0
    while True:
        try:
            os.replace(tmp, path)
            return path
        except PermissionError:
            # Windows: the destination is open by a reader without
            # FILE_SHARE_DELETE.  Wait for the reader to let go.
            attempt += 1
            if time.monotonic() >= deadline:
                raise
            # Jitter: sleep between 5 ms and 45 ms.  A reader that loops
            # without pause would otherwise be able to win every race.
            time.sleep(0.005 + (_JITTER_RNG.random() * 0.04))
        except FileNotFoundError:
            # The SOURCE temp file vanished between the write and the
            # rename.  In production this means something else deleted it
            # (a concurrent prune, or a second writer on the same path).
            # Re-create it and try again rather than losing the
            # checkpoint -- a lost checkpoint is a lost 20 minutes of
            # training, not a cosmetic error.
            attempt += 1
            if time.monotonic() >= deadline:
                raise
            torch.save(obj, tmp)
            time.sleep(0.005 + (_JITTER_RNG.random() * 0.04))


def _snapshot_step(path) -> int:
    """Extract the step number from a `step<N>_soft.pt` / `step<N>_hard.pt`."""
    name = Path(path).name
    try:
        return int(name.split("_", 1)[0].replace("step", ""))
    except (ValueError, IndexError):
        return -1


def _prune_checkpoints(run_dir, args) -> None:
    """Keep the disk bounded during a long run.

    WHY
    ---
    A float snapshot (weights + Adam moments) is ~203 MB and a hard export
    is ~34 MB.  A 100k-step run with a snapshot every 2,000 steps would
    write 50 x 237 MB = ~12 GB.  On a nearly-full volume that fills the
    disk and kills the run, and the failure surfaces as an unrelated
    "No space left on device" in the middle of a long training run.

    WHAT IS KEPT
    ------------
    * The most recent `--keep-last` FLOAT snapshots (default 3).  Those are
      the ones a resume actually uses, and they carry the optimiser + RNG
      state.
    * `best_soft.pt` and `best_hard.pt` -- the promoted artifact.
    * The most recent `--keep-last` HARD exports (default 3).

    Disk model for a 100k-step run with a snapshot every 2,000 steps
    (50 snapshots), at the measured sizes (float 203 MB, hard 34 MB):

        unbounded : 50 x (203 + 34) MB           = 11.9 GB   <- does NOT fit
        pruned    :  3 x (203 + 34) MB + best    =  0.9 GB   <- fits easily

    `--keep-hard-all` keeps every hard export (50 x 34 MB = 1.7 GB plus the
    float snapshots, about 2.6 GB total) if the full hard history is
    wanted.

    `best_hard.pt`, `best_soft.pt`, `hard_model_final.pt` and
    `final_float.pt` are never touched.
    """
    ckdir = Path(run_dir) / "checkpoints"
    if not ckdir.is_dir():
        return
    keep = max(1, int(getattr(args, "keep_last", 3)))

    # Float snapshots, newest first.
    softs = sorted(ckdir.glob("step*_soft.pt"),
                   key=_snapshot_step, reverse=True)
    for p in softs[keep:]:
        try:
            p.unlink()
            print(f"[prune] removed old float snapshot {p.name}", flush=True)
        except OSError as e:
            print(f"[prune] could not remove {p.name}: {e}", flush=True)

    if getattr(args, "keep_hard_all", False):
        return

    # The hard export that pairs with a pruned float snapshot goes too --
    # but the most recent `keep` hard exports always survive.
    hards = sorted(ckdir.glob("step*_hard.pt"),
                   key=_snapshot_step, reverse=True)
    for p in hards[keep:]:
        try:
            p.unlink()
            print(f"[prune] removed old hard export {p.name}", flush=True)
        except OSError as e:
            print(f"[prune] could not remove {p.name}: {e}", flush=True)


def export_hard_with_meta(model, meta: dict):
    """`export_hard` across both architectures.

    v7's `GFMarkovLM.export_hard(meta=...)` stamps the checkpoint dict; v6's
    takes no arguments at all. The transcript stamp lives in that meta and
    the UI server reads it to pick the byte format, so it is not optional --
    wrap instead of branching at every call site.
    """
    try:
        return model.export_hard(meta=meta)
    except TypeError:
        out = model.export_hard()
        if isinstance(out, dict) and meta:
            out.update({k: v for k, v in meta.items() if k not in out})
        return out


class Holdout:
    """Held-out positions, scored by whether the model's pick is OPTIMAL.

    bpb is not the metric we want here. A model can have low bpb while
    consistently picking the lowest-index of several equally-optimal moves,
    which is exactly the failure V14 had. So we score the decision: is the
    argmax among the optimal columns? Ties count as a hit, because any
    optimal move is a correct move.
    """

    def __init__(self, corpus_path: str, targets_path: str, device):
        import numpy as _np
        self.seqs = [l for l in open(corpus_path, "rb").read().split(b"\n")
                     if l]
        _z = _np.load(targets_path)
        T = _z["target"]
        if len(T) != len(self.seqs):
            raise SystemExit(
                f"hold-out corpus has {len(self.seqs)} rows but sidecar has "
                f"{len(T)}; refusing to report a metric on misaligned data.")
        # The sidecar schema differs between the two holdout artifacts:
        #
        #   * the OLD holdout sidecar is (N, 7)   -- ONE target row per game,
        #     for the game's final ply;
        #   * the PRODUCTION per-ply sidecar is (N, K, 7) -- one target row
        #     per LABELLED PLY of the game.
        #
        # The old code assumed (N, 7) and did `T.max(axis=1)`.  With a
        # per-ply sidecar that produced `optimal` of shape (N, K, 7), and
        # `bool(opt[pick])` then raised
        #   RuntimeError: Boolean value of Tensor with more than one value
        #     is ambiguous
        # because `opt` was a (7,) vector rather than a scalar.
        #
        # That path was never exercised before, because run_v24.sh did not
        # pass --holdout-corpus/--holdout-targets at all.  Handle BOTH
        # schemas explicitly rather than letting the shape decide.
        self.per_ply = (T.ndim == 3)
        if self.per_ply:
            # Per-ply: the target for the game's FINAL ply lives at the
            # SLOT whose `ply_of` entry equals the game's last ply index.
            # The slot index is NOT the ply index -- that is exactly the
            # sidecar subtlety the per-ply training path was fixed for.
            ply_of = _z["ply_of"] if "ply_of" in _z else None
            n_plies = _z["n_plies"] if "n_plies" in _z else None
            if ply_of is None:
                raise SystemExit(
                    "hold-out sidecar is per-ply (N,K,7) but carries no "
                    "'ply_of'; refusing to guess the final-ply slot.")
            _T_final = _np.zeros((len(self.seqs), 7), dtype=T.dtype)
            for i in range(len(self.seqs)):
                _nply = (int(n_plies[i]) if n_plies is not None
                         else len(self.seqs[i]) // PLY_LEN)
                _target_ply = max(0, _nply - 1)
                _slots = _np.nonzero(ply_of[i] == _target_ply)[0]
                if len(_slots) == 0:
                    # No label for the final ply in this row: fall back to
                    # the LAST labelled slot, which is the closest thing.
                    _labelled = _np.nonzero(ply_of[i] >= 0)[0]
                    if len(_labelled) == 0:
                        continue
                    _slots = _labelled[-1:]
                _T_final[i] = T[i, _slots[0]]
            T = _T_final
        # (K, 7) boolean: which columns are optimal
        #
        # This was `T > 1e-6`, which is only correct for the old hard 0/1
        # targets (values exactly 0 or 1/k). With SCORE-AWARE targets every
        # legal column carries a small non-zero probability, so `T > 1e-6`
        # marks almost every column "optimal" and the hold-out metric reads
        # 100% for an untrained model -- it scored 100.0% at step 0.
        #
        # Take the argmax instead. `argmax == max` still admits genuinely tied
        # optimal moves (the old targets tie exactly), which is the intended
        # "any equally optimal move counts" behaviour, but it does NOT admit
        # merely-close-to-best moves.
        _best = T.max(axis=1, keepdims=True)
        self.optimal = torch.tensor(_np.isclose(T, _best, rtol=0.0, atol=1e-6),
                                    dtype=torch.bool)
        self.targets = torch.tensor(T, dtype=torch.float32)
        self.device = device
        self.n = len(self.seqs)

    def score(self, model, max_rows: int = 4000) -> dict:
        import numpy as _np
        from .fmt import COL_BYTE_OFFSET
        model.set_ste(False)
        idxs = list(range(self.n))
        if self.n > max_rows:
            rs = _np.random.default_rng(12345)
            idxs = rs.choice(self.n, size=max_rows, replace=False).tolist()
        hit = tot = 0
        edge_hit = edge_tot = 0
        per_col_hit = [0] * 7
        per_col_tot = [0] * 7
        for i in idxs:
            seq = self.seqs[i]
            nply = len(seq) // PLY_LEN
            if nply < 1:
                continue
            # The labelled position is the final ply. Build the window that
            # ends at the sequence end, exactly as training does.
            ctx = min(args_ctx_holder[0], len(seq) - 1)
            start = max(0, len(seq) - (ctx + 1))
            window = seq[start:start + ctx + 1]
            window = window + b"\n" * (ctx + 1 - len(window))
            arr = _np.frombuffer(window, dtype=_np.uint8)
            x = torch.tensor(arr[:-1].astype(_np.int64),
                             device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0]
            pos = len(seq) - 1 - start - 1
            if not (0 <= pos < ctx):
                continue
            lg = logits[pos][COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
            # Legality comes from the header that GOVERNS the move byte we
            # are scoring. The header is the 9 bytes at the ply base; the move
            # byte is at base+MOVE_OFFSET. `logits[pos]` predicts window byte
            # `pos+1`, whose ABSOLUTE index is start+pos+1 -- so the ply base
            # is start+pos+1-MOVE_OFFSET, i.e. window index pos+1-MOVE_OFFSET.
            # Off by one either way and you slice the previous ply's tail
            # (a move byte like 0x42), which decode_state rejects outright.
            hdr_at = pos + 1 - MOVE_OFFSET
            if 0 <= hdr_at < ctx:
                legal = _legal_from_header(None, x[0, hdr_at:hdr_at + 9])
            else:
                legal = torch.ones(7, dtype=torch.bool)
            lg = lg.masked_fill(~legal.to(lg.device), float("-inf"))
            pick = int(lg.argmax())
            opt = self.optimal[i].to(lg.device)
            tot += 1
            is_opt = bool(opt[pick])
            hit += int(is_opt)
            per_col_tot[pick] += 1
            per_col_hit[pick] += int(is_opt)
            # separately track the two EDGE columns, the known weak spot
            if pick in (0, 6) or bool(opt[0]) or bool(opt[6]):
                edge_tot += 1
                edge_hit += int(is_opt)
        model.set_ste(True)
        return {
            "holdout_n": tot,
            "holdout_optimal_pct": 100.0 * hit / max(tot, 1),
            "edge_optimal_pct": 100.0 * edge_hit / max(edge_tot, 1),
            "holdout_pick_dist": [per_col_hit[c] for c in range(7)],
            "holdout_pick_tot": [per_col_tot[c] for c in range(7)],
        }


args_ctx_holder = [191]


def _legal_from_header(hdr, full_bytes=None):
    """Legal columns from a v2 column header.

    The 9-byte header is NOT a plain height bitfield: the column byte packs
    height and per-row ownership together (a height-6 column is 0xc7, whose
    low bits 0x47 are ownership, not a height mask). Scanning for the highest
    set bit therefore reads the wrong height -- it reported a FULL column as
    legal, which would let the metric score an unplayable move as optimal.

    So we reuse the one decoder that is already proven against the board:
    transcript_v2.decode_state. `full_bytes` must be all 9 header bytes.
    """
    from .transcript_v2 import decode_state
    cols, _last, _side = decode_state(bytes(full_bytes if full_bytes is not None
                                            else hdr))
    out = torch.zeros(7, dtype=torch.bool)
    for c, (h, _owners) in enumerate(cols):
        if h < 6:
            out[c] = True
    return out


def _soft_rows_for_batch(examples, chosen, starts, soft_bank, width, ply_len):
    """(B, width-1) long: sidecar row index per position, or -1.

    A position carries a soft target only when it is the FINAL ply of its
    sequence -- the one the dataset labelled.

    Two offsets matter and conflating them was an off-by-10x (a whole ply
    block, since PLY_LEN=10) that tg_softwire.py caught:

      * `final_move_byte` is where the final move byte SITS in the stream:
        len(seq) - 1. It must be inside the window, i.e.
        start <= final_move_byte < start + width. If the window ends before
        the sequence ends, the labelled ply simply is not in this window and
        no position in it may carry a target.
      * `position` is the index into the (B, width-1) target axis:
        final_move_byte - start - 1, because y[0] predicts window byte 1.

    Using len(seq) directly for the position (instead of subtracting start)
    put every target 224 bytes late -- far outside the window, so the run
    would have trained one-hot on every ply and looked merely "unable to
    learn" rather than raising an error.
    """
    B, W = len(chosen), width - 1
    out = torch.full((B, W), -1, dtype=torch.long)
    for k in range(B):
        L = len(examples[chosen[k]])
        st = int(starts[k])
        final_move_byte = L - 1
        if not (st <= final_move_byte < st + width):
            continue                      # window missed the labelled ply
        pos = final_move_byte - st - 1
        if 0 <= pos < W:
            out[k, pos] = int(chosen[k])
    return out


def _align_soft_targets(args, games, version):
    """Load the importer's (N,7) soft targets, aligned to the example rows.

    The sidecar is written in the same order as the corpus file, and
    `games_to_examples` emits exactly one example per game, so row i of the
    sidecar is row i of `examples`. If the counts disagree we refuse rather
    than silently scoring the wrong row -- a misaligned target bank looks
    like a model that merely cannot learn.
    """
    path = getattr(args, "soft_targets", "") or ""
    if not path or args.soft_w <= 0.0:
        return None
    import numpy as _np
    z = _np.load(path)
    T = z["target"]
    if len(T) != len(games):
        raise SystemExit(
            f"soft-target sidecar has {len(T)} rows but the corpus has "
            f"{len(games)} games; refusing to train on misaligned targets.")
    return torch.tensor(T, dtype=torch.float32)


def _column_prior_from_targets(perply):
    """(7,) log p_teacher(c) from the per-ply sidecar, for logit adjustment.

    Measured defect this corrects: the model's column logits carry a
    board-INDEPENDENT per-column offset (gap 2.58) that dwarfs the
    board-dependent signal (spread 0.47), so the argmax answers c3 on
    ~91% of positions regardless of the board. That offset is the
    teacher's marginal column distribution learned by the readout.

    Subtracting log p_teacher(c) from column c's logit makes a UNIFORM
    prediction optimal, so the loss can only be reduced by learning the
    board-dependent part.

    Returns None when there is no per-ply sidecar (nothing to measure).
    """
    if perply is None:
        return None
    import numpy as _np
    _pt = perply[0]
    _po = perply[1]
    valid = _po >= 0                      # (N,P)
    if not bool(valid.any()):
        return None
    # Only rows with a real labelled slot contribute.
    T = _pt[valid]                         # (M,7)
    mass = T.sum(dim=1, keepdim=True)
    good = mass[:, 0] > 0
    if not bool(good.any()):
        return None
    T = T[good] / mass[good]
    p = T.sum(dim=0)
    p = p / p.sum().clamp(min=1e-9)
    p = p.clamp(min=1e-6)
    lp = torch.log(p)
    return lp


def _align_perply_targets(args, games, version):
    """Load a per-ply (N,P,7) target sidecar, aligned to the example rows.

    Returns (target (N,P,7) float32, ply_of (N,P) long, n_plies (N,) long)
    or None when no per-ply sidecar is configured.

    Unlike the single-label sidecar, this carries a target for EVERY ply of
    the transcript, not just the final one. That is the whole point: the
    single-label path supervises one move per 193-byte window, which is
    6.24% label density, and the per-ply path supervises up to 19 -- a 16x
    increase in real move decisions per batch from the SAME bytes.
    """
    path = getattr(args, "perply_targets", "") or ""
    if not path:
        return None
    import numpy as _np
    z = _np.load(path)
    T = z["target"]
    if len(T) != len(games):
        raise SystemExit(
            f"per-ply sidecar has {len(T)} rows but the corpus has "
            f"{len(games)} games; refusing to train on misaligned targets.")
    if T.ndim != 3 or T.shape[2] != 7:
        raise SystemExit(
            f"per-ply sidecar target must be (N,P,7); got {T.shape}.")
    P = z["ply_of"].astype(_np.int64) if "ply_of" in z.files else \
        _np.tile(_np.arange(T.shape[1]), (T.shape[0], 1))
    nplies = z["n_plies"].astype(_np.int64) if "n_plies" in z.files else \
        _np.full(len(T), T.shape[1], dtype=_np.int64)
    return (torch.tensor(T, dtype=torch.float32),
            torch.tensor(P, dtype=torch.long),
            torch.tensor(nplies, dtype=torch.long))


def _perply_corpus_rows(chosen, width, device):
    """(B*width,) long: the CORPUS ROW each flattened position came from.

    A training batch is B windows of `width` bytes, flattened row-major:
    position `k * width + t` belongs to window `k`, which was sampled from
    corpus row `chosen[k]`.  The per-ply sidecar is indexed by that corpus
    row -- NOT by the position inside the batch.

    The original implementation used

        torch.arange(keep.numel()) // (width + 1)

    which yields 0..B-1 -- the BATCH INDEX -- so the trainer read the
    teacher answers of sidecar rows 0..15 of the 320,000-row corpus for
    every step of every run.  The slot index was still taken from the
    correct row, so nothing crashed: the target was a real, well-formed
    label belonging to a DIFFERENT GAME.

    Measured on the production corpus + sidecar (13,041 supervised
    positions): trainer target == correct target on 9.59 % of positions;
    16 distinct sidecar rows read out of 320,000; 41.27 % of targets
    all-zero, which made `_weighted_move_loss` silently fall back to a
    one-hot on the PLAYED move.  With this function the all-zero rate is
    0.00 %.

    Regression test: tmp/test_perply_row_index.py
    """
    return torch.as_tensor(
        np.asarray(chosen, dtype=np.int64),
        device=device).repeat_interleave(width)


def _perply_rows_for_batch(examples, chosen, starts, ply_bank, ply_of,
                            width, ply_len):
    """(B, width-1) long: sidecar SLOT index per position, or -1.

    A position carries a per-ply target when the move byte it predicts is
    the move byte of ply p, AND ply p has a labelled slot in the sidecar.

    The move byte of ply p sits at absolute stream offset
    p * ply_len + MOVE_OFFSET (9) -- the last byte of the ply's record.
    Position t predicts stream byte (start + t + 1), so the ply that owns
    the predicted byte is

        p = (start + t + 1 - MOVE_OFFSET) // ply_len

    and the target is the sidecar slot whose ply_of equals p.

    CRITICAL 1: the floor division above is only meaningful when the byte
    being predicted IS exactly a move byte. Every other byte of the ply
    (its 9-byte header) floors into the SAME ply p, so without the
    exactness check the grid would attach a ply's target to all 10 of its
    bytes -- 9 of them predicting a header byte, not a move. That is a
    silent mislabel of 90% of the supervision, and it would look like
    "the model trains but does not improve".

    CRITICAL 2: MOVE_OFFSET is 9, NOT ply_len - 2 (8). The move byte is
    the LAST byte of the 10-byte record, and using 8 shifts every target
    one byte early -- onto the byte before the move, which the model then
    learns to predict a column where the truth is a header byte. Both the
    implementation and its test must use MOVE_OFFSET.

    The slot index is what identifies the target; the row is `chosen[k]`.
    `ply_of` entries of -1 mark slots the relabeller could not solve;
    those plies are skipped, never faked.
    """
    B, W = len(chosen), width - 1
    out = torch.full((B, W), -1, dtype=torch.long)
    move_off = MOVE_OFFSET
    for k in range(B):
        L = len(examples[chosen[k]])
        st = int(starts[k])
        row = int(chosen[k])
        if row >= len(ply_of):
            continue
        # slot -> ply map for the labelled slots of this row
        slot_of = {}
        for s, p in enumerate(ply_of[row].tolist()):
            if p >= 0:
                slot_of[int(p)] = s
        if not slot_of:
            continue
        for t in range(W):
            abs_byte = st + t + 1
            if abs_byte >= L:
                break
            p = (abs_byte - move_off) // ply_len
            # only the move byte itself carries a move target
            if abs_byte != p * ply_len + move_off:
                continue
            if p in slot_of:
                out[k, t] = slot_of[p]
    return out


def _soft_logits_batch(model, streams, ctx, device):
    """Soft-path column logits for a list of streams, WITH gradient.

    Each stream is windowed independently with the canonical
    `trainer_window` (right-pad, read at the last real token), then the
    windows are right-padded to a common length and forwarded as ONE
    batch.  This is the batched twin of `_soft_logits`; the per-row read
    index is re-derived exactly as `Agent.batch_column_logits` does.

    The batched form is what makes the symmetry penalty affordable: the
    un-batched version cost one forward per position, and with 8
    positions x 2 (original + mirror) it dominated the step time.
    """
    from .agent import trainer_window, RIGHT_PAD_BYTE
    rows = []
    reads = []
    for s in streams:
        x, read_idx = trainer_window(list(s), ctx)
        rows.append(x)
        reads.append(read_idx)
    T = max(len(r) for r in rows)
    padded = [r + [RIGHT_PAD_BYTE] * (T - len(r)) for r in rows]
    t = torch.tensor(padded, dtype=torch.long, device=device)
    logits = model(t)                                     # (B,T,256)
    return torch.stack([logits[b, reads[b], COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
                        for b in range(len(rows))])         # (B,7)


def _hard_logits_batch(model, hard, streams, ctx):
    """Hard-path column logits for a list of streams (no gradient)."""
    from .agent import trainer_window, RIGHT_PAD_BYTE
    rows = []
    reads = []
    for s in streams:
        x, read_idx = trainer_window(list(s), ctx)
        rows.append(x)
        reads.append(read_idx)
    T = max(len(r) for r in rows)
    padded = [r + [RIGHT_PAD_BYTE] * (T - len(r)) for r in rows]
    t = torch.tensor(padded, dtype=torch.long)
    logits = model.hard_forward(t, hard)                    # (B,T,256)
    return torch.stack([logits[b, reads[b], COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
                        for b in range(len(rows))]).float()  # (B,7)


class PerPlyAgreement:
    """Fraction of LABELLED per-ply positions whose model pick is optimal.

    WHY THIS EXISTS
    ---------------
    The promotion gate used to fall back to the byte-level bpb when it had
    no holdout and no ladder measurement.  Byte bpb measures how well the
    model predicts the PLAYED move byte; the trainer's actual objective is
    the masked 7-way loss against the TEACHER's label.  Those are
    different quantities, and they point in different directions: in the
    500-row acceptance run the gate selected step 1500 (byte bpb 5.513)
    over step 3923 (byte bpb 6.798), even though step 3923 fits the
    labels 91.29 % of the time and step 1500 only 62.82 %.

    So the gate was promoting the checkpoint that fits the objective LEAST.
    This scorer measures the objective directly, on the real sidecar, on the
    HARD export (the thing that actually ships), with the same legal mask
    the agent applies at serve time.

    MEASUREMENT PROTOCOL
    --------------------
    * A position is (corpus row, ply).  The label is the sidecar slot for
      that row and ply -- the SAME lookup the training loss performs, so a
      disagreement here is a real disagreement, not an alignment artefact.
    * The PROMPT is the transcript prefix that ends with ply p's own
      9-byte header.  In the v2 layout each ply record is
      ``[9 header bytes][1 move byte]``, and the header of ply p encodes the
      board BEFORE ply p's move.  So the prompt is
      ``seq[:p*PLY_LEN + HEADER_LEN]`` and the move byte to predict is at
      ``p*PLY_LEN + MOVE_OFFSET``.  Cutting the prompt at ``p*PLY_LEN``
      (dropping the header) would show the model a board that does not yet
      contain the current position -- and the legal mask would then come
      from the wrong header.
    * A position counts as a HIT when the model's masked argmax column has
      the maximum target value (ties count, which is the "any equally
      optimal move is correct" semantics the sidecar encodes).
    * ply 0 has no preceding bytes, but it DOES have a header, so it is
      scored normally (unlike the byte-level trainer window, which cannot
      predict the first byte without a preceding context).
    """

    def __init__(self, examples, targets, ply_of, n_plies, device,
                 n_rows: int = 512, seed: int = 32, hard=None, model=None,
                 ctx: int = 192):
        self.examples = examples
        # The callers pass DIFFERENT types for the same fields: the
        # training-corpus scorer passes torch tensors, the holdout scorer
        # passes numpy arrays straight out of `np.load`.  The scoring code
        # below calls `.to(device)` on the targets, which numpy arrays do
        # not have -- so the holdout path died with
        #   AttributeError: 'numpy.ndarray' object has no attribute 'to'
        # and, because that failure was caught and only logged, the run
        # continued SILENTLY WITHOUT the holdout generalisation metric and
        # without the abort rules.
        #
        # Normalise here so the two callers cannot diverge again.
        self.targets = (targets if torch.is_tensor(targets)
                        else torch.as_tensor(targets))
        self.ply_of = (ply_of if torch.is_tensor(ply_of)
                       else torch.as_tensor(ply_of))
        self.n_plies = (n_plies if torch.is_tensor(n_plies)
                        else torch.as_tensor(n_plies))
        self.device = device
        self.n_rows = n_rows
        self.seed = seed
        # The HARD payload to score.  It must be PASSED IN, not re-exported
        # from the float parameters: GFMarkovLM_V6 has no `import_hard`, so
        # a V6 checkpoint loaded through `_load_agent` keeps ZERO float
        # parameters, and `export_hard()` on it yields an all-zero model
        # (every logit 0.0, argmax always column 0 -- a constant, useless
        # policy that scores ~12 % regardless of the checkpoint).
        # The trainer passes the live `hard` it already exports for serving;
        # offline measurement loads the payload straight from the file.
        self.hard = hard
        # A model instance is needed only to CALL `hard_forward` (it is a
        # method on GFMarkovLM).  hard_forward reads only the `hard` payload,
        # so a freshly-constructed model with random float params is fine --
        # the integer tables in `hard` are what actually produce the logits.
        self.model = model
        # The context window the WEIGHTS were trained with.  It MUST match
        # the trainer's --ctx: `trainer_window` slices the last `ctx` real
        # bytes and reads at index ctx-1, so a different ctx shifts the read
        # position and feeds the model a window length it never saw.  The
        # acceptance run used --ctx 192.
        self.ctx = ctx

    def _sample_rows(self):
        import numpy as _np
        n = len(self.examples)
        if n <= self.n_rows:
            return list(range(n))
        rs = _np.random.default_rng(self.seed)
        return sorted(rs.choice(n, size=self.n_rows, replace=False).tolist())

    def score(self, model, max_positions: int = 4000) -> dict:
        """Return {"perply_agreement_pct", "perply_n", "perply_mass"}."""
        from .agent import trainer_window, RIGHT_PAD_BYTE

        hard = self.hard
        if hard is None:
            # No hard payload supplied: fall back to the float weights, and
            # say so, rather than silently scoring a zeroed export.
            hard = None

        rows = self._sample_rows()
        slot_of = {}
        for r in rows:
            if r >= len(self.ply_of):
                continue
            m = {}
            for s, p in enumerate(self.ply_of[r].tolist()):
                if p >= 0:
                    m[int(p)] = s
            if m:
                slot_of[r] = m

        if not slot_of:
            return {"perply_agreement_pct": -1.0, "perply_n": 0,
                    "perply_mass": -1.0}

        streams, keys = [], []
        for r in rows:
            m = slot_of.get(r)
            if not m:
                continue
            seq = self.examples[r]
            nply = len(seq) // PLY_LEN
            for p in sorted(m.keys()):
                if p >= nply:
                    continue
                base = p * PLY_LEN
                # The prompt must END with ply p's own header (9 bytes),
                # because that header encodes the board before ply p's move.
                end = base + HEADER_LEN
                if end > len(seq):
                    continue
                streams.append(seq[:end])
                keys.append((r, p, m[p]))
        if not streams:
            return {"perply_agreement_pct": -1.0, "perply_n": 0,
                    "perply_mass": -1.0}
        if len(streams) > max_positions:
            streams = streams[:max_positions]
            keys = keys[:max_positions]

        model = model if model is not None else self.model
        was_ste = getattr(model, "ste", None)
        if model is not None:
            try:
                model.set_ste(False)
            except Exception:
                pass

        hit = tot = 0
        mass = 0.0
        # Per-column pick histogram + per-column label histogram.  These are
        # the inputs to the "the model never picks c1" abort rule: a column
        # that is ABSENT from the labels and never picked is a dead column,
        # while a column that IS in the labels but never picked is a
        # RANKING defect.  Without the label histogram the two are
        # indistinguishable and the abort rule fires on the wrong cause.
        picks = [0] * 7
        label_cols = [0] * 7          # how often each column is OPTIMAL
        label_col_hit = [0] * 7       # how often each optimal column is picked
        CH = 64
        try:
            for i in range(0, len(streams), CH):
                chunk = streams[i:i + CH]
                wins, reads = [], []
                for s in chunk:
                    x, read_idx = trainer_window(list(s), self.ctx)
                    wins.append(x)
                    reads.append(read_idx)
                T = max(len(w) for w in wins)
                padded = [w + [RIGHT_PAD_BYTE] * (T - len(w)) for w in wins]
                t = torch.tensor(padded, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    if hard is not None:
                        # Score the HARD export: that is the artefact that
                        # ships and that the UI serves.  Scoring the float
                        # path would measure a model that is never deployed.
                        logits = model.hard_forward(t.cpu(), hard)
                    else:
                        if model is None:
                            raise RuntimeError(
                                "PerPlyAgreement.score needs either a hard "
                                "payload or a live model")
                        logits = model(t)
                for b in range(len(chunk)):
                    lg = logits[b, reads[b], COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
                    r, p, slot = keys[i + b]
                    tgt = self.targets[r, slot].to(lg.device)
                    best = tgt.max()
                    if float(best) <= 0.0:
                        continue
                    legal = _legal_for_ply(self.examples[r], p, lg.device)
                    if legal is not None:
                        lg = lg.masked_fill(~legal, float("-inf"))
                    pick = int(lg.argmax())
                    ok = bool(tgt[pick] >= best - 1e-6)
                    hit += int(ok)
                    tot += 1
                    picks[pick] += 1
                    # Which columns are optimal here?  A tie counts for
                    # every tied column.
                    opt_cols = (tgt >= best - 1e-6).nonzero().flatten().tolist()
                    for c in opt_cols:
                        label_cols[c] += 1
                        if c == pick:
                            label_col_hit[c] += 1
                    probs = torch.softmax(lg, dim=-1)
                    mass += float(probs[pick])
        finally:
            if model is not None:
                try:
                    model.set_ste(True if was_ste is None else was_ste)
                except Exception:
                    pass

        return {
            "perply_agreement_pct": 100.0 * hit / max(tot, 1),
            "perply_n": tot,
            "perply_mass": mass / max(tot, 1),
            "perply_picks": picks,
            "perply_label_cols": label_cols,
            "perply_label_col_hit": label_col_hit,
            # Fraction of the OPTIMAL opportunities on each column that the
            # model actually converted.  This is the per-column RECALL that
            # separates "c1 is absent from the data" from "c1 is present but
            # ranked below the others".
            "perply_col_recall": [
                (label_col_hit[c] / label_cols[c]) if label_cols[c] else -1.0
                for c in range(7)
            ],
        }


def _legal_for_ply(seq, ply, device):
    """Legal columns for the position BEFORE ply `ply` is played.

    In the v2 layout ply `ply`'s record is
    ``[9 header bytes][move byte]`` at ``ply*PLY_LEN``, and that header
    encodes the board BEFORE the move -- which is exactly the position the
    label describes.  So the header at ``ply*PLY_LEN`` is the right one.
    """
    if ply < 0:
        return None
    base = ply * PLY_LEN
    if base + HEADER_LEN > len(seq):
        return None
    hdr = seq[base:base + HEADER_LEN]
    try:
        legal = _legal_from_header(None, hdr)
    except Exception:
        return None
    return legal.to(device)


def _symmetry_penalty(model, chosen, examples, keep, ctx, device, version,
                      n_positions, use_hard=False):
    """The mirror-equivariance penalty for the current batch.

    Picks up to `n_positions` supervised move positions out of the batch,
    rebuilds each position's board, constructs the REFLECTED position's
    stream with the canonical mirror helpers, evaluates the model on both
    streams, and returns the Jeffreys divergence between the mirrored
    prediction and the reflection of the original prediction.

    Everything here uses the CANONICAL mirror path (`src.c4.symmetry` ->
    `src.c4.mirror`), and the reflected stream is built from the reflected
    MOVE LIST, which is asserted elsewhere to reproduce `mirror_board`.
    A hand-rolled mirror is what produced a false "the model always plays
    c0" result earlier in this project.

    GRADIENT NOTE: the hard export is a CPU-only integerised table and
    cannot supply a gradient.  The penalty is therefore always evaluated
    on the SOFT path, which is what the hard path is a straight-through
    estimator of.  The `use_hard` flag only selects which path the
    penalty is REPORTED on; it never changes the gradient.

    PERFORMANCE NOTE: the original + mirrored streams are forwarded as
    ONE batch each (not one forward per position).  The un-batched form
    made this arm ~3x slower than the other two and would have made the
    A/B unfair on wall-clock rather than on quality.
    """
    from .symmetry import (decision_stream, mirrored_decision_stream,
                            mirror_equivariance_loss)

    # Flat (B*W) index -> row index.
    keep_flat = keep.reshape(-1)
    idxs = torch.nonzero(keep_flat, as_tuple=False).reshape(-1).cpu().tolist()
    if not idxs or not len(chosen):
        return torch.tensor(0.0, device=device), None
    width = len(keep_flat) // len(chosen)
    if width <= 0:
        return torch.tensor(0.0, device=device), None

    # Deterministic spread over the batch: evenly spaced indices, so the
    # penalty sees a variety of positions rather than the first few.
    if len(idxs) > n_positions:
        step = len(idxs) / float(n_positions)
        idxs = [idxs[int(i * step)] for i in range(n_positions)]

    moves_list = []
    for flat_i in idxs:
        row = flat_i // width
        if row >= len(chosen) or row >= len(examples):
            continue
        try:
            moves = _decode_moves(examples[chosen[row]], version)
        except Exception:
            continue
        if not moves:
            continue
        # The position must be a real, non-terminal board.
        b = Board()
        ok = True
        for c in moves:
            if c not in b.valid_actions():
                ok = False
                break
            b.apply_move(c)
        if not ok or b.is_win() != 0 or not b.valid_actions():
            continue
        moves_list.append(moves)

    if not moves_list:
        return torch.tensor(0.0, device=device), None

    orig_streams = [decision_stream(m, version) for m in moves_list]
    mirr_streams = [mirrored_decision_stream(m, version) for m in moves_list]

    lo = _soft_logits_batch(model, orig_streams, ctx, device)
    lm = _soft_logits_batch(model, mirr_streams, ctx, device)

    legal = []
    for m in moves_list:
        b = Board()
        for c in m:
            b.apply_move(c)
        legal.append(torch.tensor([c in b.valid_actions() for c in range(7)],
                                  dtype=torch.bool, device=device))
    la = torch.stack(legal)
    # The mirrored board's legal columns are the reflection of the
    # original's.
    lam = la.flip(-1)

    if use_hard:
        # Report the penalty on the HARD path (the path that ships), but
        # keep the gradient from the soft path.  The hard export is a
        # CPU-only integerised table; the STE makes it a straight-through
        # estimator of the soft function during training.
        #
        # DEVICE NOTE: the hard tables live on the CPU by design, so the
        # hard logits AND the legal masks they are compared against must
        # be on the CPU too.  Passing a CUDA legal mask next to CPU hard
        # logits raises "indices should be either on cpu or on the same
        # device as the indexed tensor".
        with torch.no_grad():
            hard = _hard_export(model)
            ho = _hard_logits_batch(model, hard, orig_streams, ctx)
            hm = _hard_logits_batch(model, hard, mirr_streams, ctx)
            hard_pen = mirror_equivariance_loss(
                ho.cpu(), hm.cpu(), la.cpu(), lam.cpu())
        soft_pen = mirror_equivariance_loss(lo, lm, la, lam)
        # The gradient comes from the soft path; the hard value is logged.
        return soft_pen, float(hard_pen.detach().item())

    return mirror_equivariance_loss(lo, lm, la, lam), None


def _decode_moves(line, version):
    """Decode a corpus line into its move list."""
    if version == "v1":
        from .transcript import decode_game
    else:
        from .transcript_v2 import decode_game
    return decode_game(line)


def _hard_export(model):
    from .agent import hard_to_cpu
    return hard_to_cpu(model.export_hard())


def _hard_logits(model, hard, tokens, ctx):
    """Hard-path column logits at the decision position (no gradient)."""
    from .agent import trainer_window
    x, read_idx = trainer_window(list(tokens), ctx)
    lg = model.hard_forward(
        torch.tensor([x], dtype=torch.long), hard)[0, read_idx]
    return lg[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].float()


def _soft_logits(model, tokens, ctx, device):
    """Soft-path column logits at the decision position, with gradient."""
    from .agent import trainer_window
    x, read_idx = trainer_window(list(tokens), ctx)
    lg = model(torch.tensor([x], dtype=torch.long, device=device))[0, read_idx]
    return lg[COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7].float()


def _weighted_move_loss(logits_flat, keep, move_col, legal_flat, w_flat,
                        dist=None, col_prior=None):
    """Masked 7-way NLL over move bytes, weighted by the outcome weight.

    `move_loss` is the unweighted mean; here each move position is weighted by
    `w_flat` (0.5 neutral resume rows, 1.0 winners, 0 losers) so the outcome
    weighting survives the switch to the 7-way head.

    `dist` is an optional (N, 7) target distribution aligned with the full
    (unfiltered) position axis. When present the loss is
    -sum_c dist[c] * log p[c] instead of -log p[move_col], which is what lets
    an equally-optimal tie share its mass instead of being collapsed to the
    lowest-index column.
    """
    w = w_flat[keep]
    tot = w.sum()
    if float(tot) <= 0.0:
        return logits_flat.sum() * 0.0
    from .fmt import COL_BYTE_OFFSET
    sub = logits_flat[keep][:, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
    lg = legal_flat[keep]
    dead = ~lg.any(dim=1)
    if bool(dead.any()):
        lg = lg.clone()
        lg[dead] = True
    # Mask illegal columns with a LARGE FINITE negative value, not -inf.
    #
    # With -inf, log_softmax puts -inf on illegal columns, and the target
    # distribution `dist` has exactly 0.0 there. The product 0 * -inf is
    # NaN, which propagates into the whole loss and destroys the model.
    # This was latent until the outcome-weight fix made the move loss live
    # for the first time: with -inf every supervised row produced NaN.
    #
    # A finite floor keeps log_softmax finite, so 0 * (-1e9) == 0.
    NEG = -1e9
    sub = sub.masked_fill(~lg, NEG)
    if col_prior is not None:
        # COLUMN-PRIOR CORRECTION (logit adjustment).
        #
        # Measured defect: the hard export's column logits carry a large
        # BOARD-INDEPENDENT per-column offset (gap 2.58) that dwarfs the
        # board-dependent signal (spread 0.47). The readout layers learn
        # "c3 is usually right" from the skewed teacher distribution, and
        # that constant is enough to win the argmax on ~91% of positions.
        # The model then never needs to read the board.
        #
        # Logit adjustment removes the constant at the SOURCE: subtracting
        # log p_teacher(c) from column c's logit makes a uniform predicted
        # distribution optimal, so the ONLY way to reduce the loss is to
        # learn the board-dependent part. This is the standard fix for
        # label-prior-driven collapse (Menon et al., "Long-tail learning
        # via logit adjustment") and it is applied to the SOFT training
        # path only -- the hard export inherits the corrected weights.
        #
        # `col_prior` is a (7,) tensor of log p_teacher(c) for the columns
        # that are legal on average. Subtract it from the logits.
        prior = col_prior.to(sub.device).view(1, 7)
        sub = sub - prior
    lp = torch.log_softmax(sub, dim=-1)
    if dist is not None:
        d = dist[keep]
        # Safety: never let the target put mass on an illegal column.
        d = d * lg.float()
        tot_d = d.sum(dim=1, keepdim=True)
        # If a row somehow had all its mass on illegal columns, fall back to
        # the one-hot target rather than dividing by zero.
        dead_t = (tot_d.squeeze(1) <= 0.0)
        if bool(dead_t.any()):
            d = d.clone()
            d[dead_t] = F.one_hot(move_col[keep][dead_t].clamp(min=0), 7).float()
            tot_d = d.sum(dim=1, keepdim=True)
        d = d / tot_d.clamp(min=1e-12)
        nll = -(d * lp).sum(dim=1)
    else:
        nll = -lp.gather(1, move_col[keep].unsqueeze(1)).squeeze(1)
    return (nll * w).sum() / tot


def _header_index_from_starts(starts, width: int, device=None):
    """(B, T-1) long: x-index of the header governing each position.

    For position t (which predicts sequence byte t+1), the owning ply is
    k = (start + t + 1) // PLY_LEN and that ply's header begins at absolute
    sequence byte k*PLY_LEN, so its x-index is k*PLY_LEN. Returns 0 for any
    position outside the ply grid, which is never consulted because
    `header_legality_from_x` masks the result by the move mask anyway.
    """
    from .fmt import PLY_LEN
    s = torch.as_tensor(starts, dtype=torch.long, device=device)
    t = torch.arange(width, device=s.device).view(1, -1)
    # Position t of row r predicts the byte at ABSOLUTE stream index
    # start + t + 1 (y[r, t] == stream[start + t + 1]). That byte is the move
    # of ply k = (start + t + 1) // PLY_LEN, whose 7-byte header begins at
    # ABSOLUTE index k*PLY_LEN -- that is, 9 bytes earlier in the same window.
    # `x` is WINDOW-LOCAL (x[r, i] == stream[start + i]), so the local index
    # is k*PLY_LEN - start. Using the absolute index here would read a header
    # up to `start` bytes too far into the window.
    k = (s.view(-1, 1) + t + 1) // PLY_LEN
    return k * PLY_LEN - s.view(-1, 1)


def evaluate_bpc(model, examples: Sequence[bytes], device, chunk: int = 64):
    """Move-byte bits-per-byte and top-1 accuracy, soft and hard.

    Both paths predict the same raw byte stream; the hard path just runs the
    integerised tables. Scoring is restricted to move bytes (see
    _move_byte_mask), so a lower bpc here means "predicts the right column
    with more confidence", not "learned the header encoding".

    The hard path is run at the SAME prompt length the agent uses
    (`hard_min_len_for(lags)`, right-padded with RIGHT_PAD_BYTE), NOT at the
    chunk width. GF hard_forward clamps any lag wider than the prompt
    (`if l > T: l = T - 1`), so scoring a ctx=192 chunk with V7's lag-256
    taps shifts in a real token where the agent would have padded -- the
    two paths compute different functions and the reported hard bpb would
    not be the bpb of the model that actually plays. Position i predicts
    byte i+1 in both paths, so the hard logits are simply truncated back to
    the soft path's width before scoring; the extra tail is discarded.
    """
    from .agent import hard_to_cpu, hard_min_len_for, RIGHT_PAD_BYTE

    seqs = [e for e in examples if len(e) > 1]
    if not seqs:
        return (float("nan"),) * 5
    hard = model.export_hard()
    hcpu = hard_to_cpu(hard)
    # length the agent would actually feed the hardened forward
    hard_len = max(hard_min_len_for(hcpu["lags"]), 1)
    import math

    soft_nll = hard_nll = 0.0
    soft_hit = hard_hit = ntok = 0
    # 7-way legal-column accuracy. Under the masked 7-way loss the model is
    # never taught to make the move byte the argmax over all 256 values, so
    # the byte-level `hard_hit` can FALL while the policy gets stronger. This
    # is the metric that matches the training objective: argmax over the
    # legal columns only.
    legal_hit = legal_n = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), chunk):
            batch = [b for b in seqs[i:i + chunk]]
            maxlen = max(len(b) for b in batch)
            arr = np.zeros((len(batch), maxlen), dtype=np.uint8)
            for j, b in enumerate(batch):
                arr[j, :len(b)] = np.frombuffer(b, dtype=np.uint8)
            xb = torch.tensor(arr.astype(np.int64), device=device)
            yb = torch.tensor(arr[:, 1:].astype(np.int64), device=device)
            # POSITIONAL mask. evaluate_bpc feeds each row from its own byte
            # 0, so every row's start is 0 and the mask is pure stride
            # arithmetic. The legacy value test would also mark the copied
            # `last` byte, which is free to predict and would inflate the
            # reported accuracy for a model that never learned the board.
            m = _move_byte_mask(yb, np.zeros(len(batch), dtype=np.int64))
            # Exclude positions that only exist because this row was
            # zero-padded up to the chunk's longest game. Without this,
            # a 40-byte row sitting in a 120-byte chunk gets mask
            # positions whose TARGET is 0x00 -- a byte the model
            # assigns essentially no probability, so each one adds ~6
            # bits of NLL. Measured 35.4% of masked positions were
            # padding in a representative chunk (3 of 4 rows), which
            # is what pushed reported bpb to 16+ and understated
            # accuracy. The old value-based mask hid this because 0x00
            # is outside 0x41..0x47.
            lens_t = torch.tensor([len(b) for b in batch], device=device)
            valid = (torch.arange(yb.shape[1], device=device)[None, :]
                     < (lens_t[:, None] - 1))
            m = m & valid
            # (B,T-1) — must stay 2-D: tgt is (B,T-1) too, and a (B,T-1,1)
            # mask would right-broadcast to (B,T-1,T-1) and blow up.
            ntok += int(m.sum().item())
            if ntok == 0:
                continue

            # soft path
            lp = F.log_softmax(model(xb), dim=-1)          # (B,T,256)
            tgt = lp[:, :-1, :].gather(2, yb.unsqueeze(2)).squeeze(2)   # (B,T-1)
            soft_nll += -(tgt * m).sum().item()
            soft_hit += int(((lp[:, :-1, :].argmax(-1) == yb) & m).sum().item())

            # hard path (integerised tables, CPU)
            hlp = F.log_softmax(model.hard_forward(xb.cpu(), hcpu).float(), dim=-1)
            htgt = hlp[:, :-1, :].gather(2, yb.cpu().unsqueeze(2)).squeeze(2)
            hard_nll += -(htgt * m.cpu()).sum().item()
            hard_hit += int(((hlp[:, :-1, :].argmax(-1) == yb.cpu()) &
                              m.cpu()).sum().item())

            # legal-column argmax from the HARD logits (the shipping path)
            if legal_n is not None or True:
                keep2, mcol2 = build_move_targets(yb.cpu(), m.cpu())
                if int(keep2.sum()) > 0:
                    hi2 = _header_index_from_starts(
                        np.zeros(len(batch), dtype=np.int64),
                        yb.shape[1])
                    lg2 = header_legality_from_x(xb.cpu(), m.cpu(), hi2)
                    vis2 = header_visibility(
                        np.zeros(len(batch), dtype=np.int64), yb.shape[1])
                    k2 = keep2 & vis2.reshape(-1)
                    if int(k2.sum()) > 0:
                        # hlp is (B,T,256); k2 is flat over (B,T-1).
                        # Drop the last position (no target) to line up.
                        flat_h = hlp[:, :-1, :].reshape(-1, 256)
                        sub = flat_h[k2][:, COL_BYTE_OFFSET:COL_BYTE_OFFSET + 7]
                        mask2 = lg2.reshape(-1, 7)[k2]
                        dead = ~mask2.any(dim=1)
                        if bool(dead.any()):
                            mask2 = mask2.clone()
                            mask2[dead] = True
                        sub = sub.masked_fill(~mask2, float("-inf"))
                        legal_hit += int((sub.argmax(-1) ==
                                          mcol2[k2]).sum().item())
                        legal_n += int(k2.sum())
    model.train()

    n = max(ntok, 1)
    ln2 = math.log(2)
    return (soft_nll / n / ln2, hard_nll / n / ln2,
            soft_hit / n, hard_hit / n,
            legal_hit / max(legal_n, 1))


# --------------------------------------------------------------------------
# main training loop
# --------------------------------------------------------------------------

def train(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Column prior for logit adjustment. Computed from the per-ply sidecar
    # at corpus-load time; None means "no adjustment" (historical
    # behaviour). Initialised here so every code path has it bound.
    col_prior = None

    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = args.device
    if getattr(args, "arch", "v7") == "v7":
        # Canonical V7 architecture from gf256_markov_lm.py
        # (d=64, layers=6, fan_in=4, lags=(0..256), n_coef=16).
        arch_cls = GFMarkovLM_V7
        d, layers, fan_in = 64, 6, 4
    else:
        arch_cls = GFMarkovLM  # V6
        d, layers, fan_in = args.d, args.layers, args.fan_in
    # --lags overrides the architecture's default lag set.
    #
    # WHY (2026-10-27): the 9-byte V2 header [c0..c6, last, turn] sits at
    # stream offsets -8..-1 relative to the read position (the TURN byte).
    # With the historical lags (0,1,2,4,8,16,32,64) and 3 layers, a column
    # is visible only if the lag SET admits a path to its offset -- and the
    # measured per-column channel counts are lopsided:
    #     c0 18 chans | c1  1 chan | c2  6 chans
    #     c6 21 chans | c5 10 chans | c4 18 chans
    # All three mirror pairs (c0/c6, c1/c5, c2/c4) are ASYMMETRIC, which is
    # a structural cause of the measured mirror inconsistency. Denser lags
    # (0..9 plus 10,20) make every header byte reachable in ONE hop, so
    # the model's view of the board becomes left/right symmetric.
    # ctx 192 is unaffected: the lag set does not change the window size.
    lags = None
    if getattr(args, "lags", ""):
        lags = tuple(int(x) for x in str(args.lags).split(",") if x.strip() != "")
        if 0 not in lags:
            raise SystemExit("--lags must contain 0 (the identity slot)")

    # ---- resume: load the checkpoint BEFORE building the model ------------
    # The checkpoint records the lag set it was trained with.  If the model is
    # built from `--lags` instead, a run trained with the dense 12-lag set
    # (0..10,20) is resumed into an 8-lag model (the V6 default) and dies
    # deep inside the forward pass with
    #   IndexError: index 9 is out of bounds for dimension 0 with size 8
    # because the exported lag table has 12 entries but the rebuilt model's
    # lag table has 8.  The lag set is part of the model's STRUCTURE, not a
    # hyperparameter the caller may freely change, so on resume it comes
    # from the checkpoint.  An explicit --lags that disagrees is refused
    # rather than silently ignored.
    start_step = 0
    resume_step = 0
    ckpt_meta = None
    ckpt = None
    if args.resume:
        # weights_only=False is REQUIRED here and is safe: this file is a
        # checkpoint this trainer wrote itself. It holds tensors, plain
        # scalars and small lists -- no pickled code objects. weights_only=True
        # would refuse it, and the resume source is an operator-supplied path
        # on the operator's own machine (same trust boundary as the run dir).
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ck_lags = None
        if isinstance(ckpt, dict):
            if "lags" in ckpt and ckpt["lags"] is not None:
                ck_lags = tuple(int(x) for x in ckpt["lags"])
            elif isinstance(ckpt.get("meta"), dict) and ckpt["meta"].get("lags"):
                ck_lags = tuple(int(x) for x in ckpt["meta"]["lags"])
        if ck_lags:
            if lags is not None and tuple(lags) != tuple(ck_lags):
                raise SystemExit(
                    f"refusing to resume: the checkpoint was trained with lags "
                    f"{list(ck_lags)} but --lags says {list(lags)}. The lag set "
                    f"is part of the model structure, so it cannot be changed "
                    f"on resume. Drop --lags to take the checkpoint's own set.")
            lags = ck_lags
            print(f"resume: adopting the checkpoint's lag set {list(ck_lags)}",
                  flush=True)

    model_kw = dict(d=d, layers=layers, fan_in=fan_in, bigram_logp=None,
                    seed=seed, table_bias=args.table_bias)
    if lags is not None:
        model_kw["lags"] = lags
    if arch_cls is GFMarkovLM_V7:
        model_kw["n_coef"] = 16
    model = arch_cls(**model_kw).to(device)
    model.use_ckpt = not args.no_ckpt
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params {n_par/1e6:.2f}M | arch={getattr(args,'arch','v6')} | "
          f"d={d} layers={layers} fan_in={fan_in} steps={args.steps} | "
          f"device={device}", flush=True)

    rng = random.Random(seed + 1)
    nprng = np.random.default_rng(seed + 2)

    # ---- LEARNING-RATE SCHEDULE -------------------------------------------
    # Claude's review is right that a constant LR will start oscillating
    # once the loss is near its floor: Adam with a fixed step size keeps
    # bouncing around the minimum.  A cosine decay over the FULL run
    # (including any resume) removes that failure mode without touching
    # the objective.
    #
    # It is OPT-IN (`--lr-schedule cosine`) so that every historical run
    # reproduces exactly: a schedule keyed on `step / args.steps` would
    # silently change the trajectory of any run that is resumed with a
    # different `--steps` target.
    _lr_sched = None
    if args.lr_schedule == "cosine":
        # Computed from the ABSOLUTE step, not from steps taken this
        # process, so a resumed run stays on the same curve.
        def _lr_at(st: int) -> float:
            _frac = min(1.0, max(0.0, st / max(1, args.steps)))
            return args.lr * (0.5 * (1.0 + math.cos(math.pi * _frac)))
        _lr_sched = _lr_at
        print(f"LR schedule: cosine {args.lr:g} -> "
              f"{args.lr * 0.5:g} over {args.steps} steps", flush=True)
    elif args.lr_schedule != "none":
        raise SystemExit(f"unknown --lr-schedule {args.lr_schedule!r}")

    # ---- EXPONENTIAL MOVING AVERAGE --------------------------------------
    # EMA weights are the standard defence against the late-training
    # oscillation a constant-LR Adam run shows: the EMA copy is much
    # flatter around the optimum than the raw weights, and it is the copy
    # that should be evaluated and shipped.
    #
    # OPT-IN (`--ema-decay`), and it applies to the FLOAT weights only.
    # The hard export is built from whichever weights we choose to export,
    # so `--ema-export` decides whether the EMA copy or the raw weights
    # are hardened and promoted.
    #
    # IMPORTANT: the shadow copy is INITIALISED AFTER the resume has
    # loaded the checkpoint weights (see `_ema_init` below).  Initialising
    # it earlier would seed the EMA from the RANDOM INIT weights of a
    # resumed run, and the EMA would then drag the true weights back
    # towards random for hundreds of steps.
    _ema = None
    _ema_decay = float(args.ema_decay)
    _ema_warmup = int(args.ema_warmup)

    def _ema_init() -> None:
        """Seed the EMA shadow from the CURRENT (possibly resumed) weights."""
        nonlocal _ema
        _ema = {k: v.detach().clone().float()
                for k, v in model.state_dict().items()
                if v.dtype.is_floating_point}

    def _ema_update(decay: float) -> None:
        if _ema is None:
            return
        with torch.no_grad():
            _sd = model.state_dict()
            for k in _ema:
                _ema[k].mul_(decay).add_(_sd[k].detach().float(),
                                        alpha=1.0 - decay)

    if _ema_decay > 0:
        print(f"EMA: decay {_ema_decay:g} (warmup {_ema_warmup} steps)",
              flush=True)

    # ---- optimiser / RNG state carried across a resume --------------------
    # Adam's moments are part of the optimisation trajectory.  Restarting
    # them from empty on every resume makes the first steps after a restart
    # behave like a fresh optimiser (a visible loss spike) and silently
    # changes the effective LR schedule.  The float checkpoint therefore
    # carries `opt_state` and `rng_state` alongside the weights.
    #
    # A HARD export cannot carry them: it is a pure integer table payload.
    # Resuming from one therefore restarts the optimiser, and that is
    # REPORTED rather than hidden.
    _resume_opt_state = None
    _resume_rng_state = None
    if args.resume:
        if isinstance(ckpt, dict):
            # Top level, NOT inside `meta`: `meta` is also handed to
            # `export_hard_with_meta`, and a hard export must stay a lean
            # integer-table payload.
            _resume_opt_state = ckpt.get("opt_state")
            _resume_rng_state = ckpt.get("rng_state")
        if _resume_opt_state is not None:
            try:
                opt.load_state_dict(_resume_opt_state)
                print("[resume] restored Adam moments from the checkpoint",
                      flush=True)
            except Exception as e:
                print(f"[resume] could NOT restore Adam moments ({e}); "
                      f"the optimiser restarts from empty moments", flush=True)
                _resume_opt_state = None
        else:
            print("[resume] checkpoint carries no optimiser state "
                  "(a hard export cannot): Adam restarts from empty moments",
                  flush=True)
        if _resume_rng_state is not None:
            try:
                rng.setstate(_resume_rng_state["python"])
                nprng.bit_generator.state = _resume_rng_state["numpy"]
                torch_rng = _resume_rng_state.get("torch")
                if torch_rng is not None:
                    torch.set_rng_state(torch_rng)
                print("[resume] restored RNG states (python/numpy/torch)",
                      flush=True)
            except Exception as e:
                print(f"[resume] could NOT restore RNG state ({e}); the run "
                      f"continues with a different sampling sequence",
                      flush=True)

    version = fmt.normalize(args.transcript)
    print(f"transcript format: {version} "
          f"({'per-cell ownership' if version == 'v2' else 'heights only'})",
          flush=True)

    # ---- resume from a hard export checkpoint -------------------------------
    # The FLOAT checkpoint ({"state": ..., "meta": ...}) now also carries the
    # optimiser moments and the RNG states, so a resumed run continues the
    # SAME optimisation trajectory rather than restarting Adam from empty
    # moments (which produces a loss spike at every restart).
    #
    # A HARD export cannot carry optimiser state -- it is a pure integer
    # table payload -- so resuming from one still restarts Adam's moments.
    # That path is explicitly reported below instead of pretending otherwise.
    #
    # The checkpoint was ALREADY loaded above (before the model was built, so
    # the lag set could be taken from it). Do not load it again.
    if args.resume:
        # --resume has to accept BOTH checkpoint shapes. `best_soft.pt` is
        # {"state": state_dict, "meta": {...}} (float parameters, what training
        # actually continues from); `best_hard.pt` is the integer export
        # {emb, coef, sbox, wiring, readout, bias, ...}. The old code called
        # `model.import_hard(ckpt)`, a method that does not exist on the model
        # at all, so ANY --resume died with AttributeError before training a
        # single step -- which is why every resume attempt so far has failed.
        # Prefer the float state; fall back to the integer export so a hard
        # checkpoint is still usable rather than a second dead end.
        if isinstance(ckpt, dict) and "state" in ckpt and isinstance(ckpt["state"], dict):
            missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
            if missing:
                print(f"[resume] {len(missing)} tensors absent from checkpoint, "
                      f"left at init: {list(missing)[:6]}", flush=True)
            if unexpected:
                print(f"[resume] {len(unexpected)} checkpoint tensors unused: "
                      f"{list(unexpected)[:6]}", flush=True)
        elif hasattr(model, "import_hard"):
            model.import_hard(ckpt)
        else:
            raise SystemExit(
                f"cannot resume from {args.resume}: it is neither a "
                f"{{'state': ...}} float checkpoint nor a hard export the "
                f"model can import.")
        model.to(device)
        ckpt_meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
        resume_step = int(ckpt_meta.get("step", 0)) if \
            isinstance(ckpt_meta, dict) else 0
        start_step = resume_step
        # Resuming into a target step that is at or below the checkpoint's own
        # step used to leave the training loop body unentered:
        #   for step in range(start_step, args.steps + 1)
        # is empty when start_step > args.steps, so `step` stayed UNBOUND and
        # the first reference to it raised
        #   UnboundLocalError: local variable 'step' referenced before assignment
        # That made it impossible to re-open a finished run for a short
        # extension -- exactly the operation a long V24 run needs. Fail loudly
        # with the reason instead of crashing on an unbound name later.
        if start_step > args.steps:
            raise SystemExit(
                f"nothing to resume: the checkpoint is already at step "
                f"{start_step}, which is beyond --steps={args.steps}. "
                f"Pass a larger --steps (the run continues from step "
                f"{start_step}), or drop --resume to start fresh.")
        # A checkpoint records the format its weights were trained on. Refuse a
        # mismatch instead of silently training on bytes the weights have
        # never seen -- that reads as pure noise and still produces a
        # plausible-looking loss curve.
        ckpt_ver = ckpt_meta.get("transcript") if isinstance(ckpt_meta, dict) else None
        if ckpt_ver is not None and fmt.normalize(ckpt_ver) != version:
            raise SystemExit(
                f"refusing to resume: checkpoint was trained on transcript "
                f"{fmt.normalize(ckpt_ver)!r} but --transcript is "
                f"{version!r}. Start a fresh run instead.")
        print(f"resumed weights from {args.resume} "
              f"(checkpoint step {resume_step})", flush=True)

    # Seed the EMA shadow from the weights that are ACTUALLY in the model
    # now -- the checkpoint's, if this was a resume, otherwise the init.
    # Doing it here (after the resume) rather than at construction is what
    # keeps a resumed run's EMA from dragging the weights back to random.
    if _ema_decay > 0:
        _ema_init()

    # ---- seed the corpus with random play (step 0 has no model yet) ----
    corpus_path = run_dir / "corpus.txt"
    corpus: List[Game] = []
    corpus_size = 0
    # The encoded training rows. Assigned on every path below (imported
    # corpus, self-play corpus, seed corpus) but declared here so the
    # board-sensitivity gate can never see an unbound name.
    examples: List[bytes] = []
    # (N, 7) soft targets aligned to `examples`, or None. Loaded lazily from
    # the importer sidecar so the 7-way head can learn the tie structure
    # instead of an argmax's lowest-index habit.
    soft_bank: Optional[torch.Tensor] = None
    row_soft: Optional[torch.Tensor] = None
    # Per-ply sidecar: (target (N,P,7), ply_of (N,P), n_plies (N,)). When
    # present, EVERY labelled ply of a transcript inside the window
    # contributes a move target, not just the final ply. This is the lever
    # that raises real supervision density from ~6.9% to ~100% without
    # adding a single byte of corpus.
    perply: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    # Lines already on disk and represented in `examples`. The quality gate
    # may exclude some of those lines from training, but they still occupy
    # disk, so this counter tracks the FILE, not the training set.
    consumed_lines = 0
    # The per-ply label-agreement scorer is the promotion key's PRIMARY
    # component when a per-ply sidecar is loaded.  It is bound HERE, before
    # the corpus branch that may create it, so the name is always bound
    # (the branch assigns it; this guarantees the default).
    perply_scorer = None
    # `--load-corpus` uses an already-populated corpus.txt WITHOUT entering the
    # weight-resume path (which needs a checkpoint) and without entering the
    # seed path (which OVERWRITES corpus.txt). The importer writes 106k
    # labelled positions into the run dir; the seed branch would discard them,
    # so the run silently trained on 2,000 self-play games instead.
    if (args.load_corpus and corpus_path.exists()
            and corpus_path.stat().st_size > 0):
        raw_lines = [ln for ln in corpus_path.read_bytes().splitlines() if ln]
        if not raw_lines:
            raise SystemExit(
                f"refusing to train: {corpus_path} contains no usable rows "
                f"(the file is {corpus_path.stat().st_size} bytes of blank "
                f"lines). If you are resuming, the run dir must contain the "
                f"same corpus the original run used -- copy it in, or point "
                f"--run-dir at the original run directory.")
        probe = next((ln for ln in raw_lines if len(ln) >= PLY_LEN), None)
        if probe is not None:
            try:
                validate_stream(probe, version)
            except ValueError as e:
                raise SystemExit(
                    f"refusing to train: {corpus_path} is not a {version!r} "
                    f"corpus ({e}).")
        # ---- corpus integrity gate ------------------------------------
        # The `probe` above checks ONE row. That is not enough: this corpus
        # had 126,537 of 320,000 rows (39.5%) whose LAST-MOVE header byte
        # contradicted their own move bytes, starting at row 193,459.
        # `probe` took row 0, which was clean, so the run trained on a
        # wrong byte for 39.5% of the data.
        #
        # Worse, the corrupted rows are MIRROR TWINS of earlier clean
        # rows: their board bytes are reflected, but their last-move byte
        # is IDENTICAL to their twin's instead of being reflected. The
        # dataset therefore literally teaches "a board and its mirror get
        # the same answer" -- the mirror asymmetry the model then showed
        # (62.73% mirror consistency).
        #
        # Scan EVERY row for the cheap last-move invariant, and refuse to
        # train on a corpus that violates it.
        if version != "v1":
            from .corpus_integrity import assert_corpus_consistent
            _irep = assert_corpus_consistent(
                corpus_path, context=str(corpus_path))
            print(f"corpus integrity: {_irep['rows_checked']} rows, "
                  f"{_irep['rows_with_last_field_violations']} with an "
                  f"inconsistent last-move field", flush=True)
        # Imported rows are all solver-labelled, so the quality gate would
        # only discard good data; the sidecar already carries the labels.
        examples = list(raw_lines)
        is_teacher = [True] * len(examples)
        if not examples:
            # A resume whose run-dir has no corpus used to fall through to
            # `build_batch`, which called `rng.integers(0, 0)` and died with
            # `ValueError: high <= 0` -- a traceback that says nothing about
            # the real problem (the run dir has no corpus).
            #
            # The corpus is what the resumed weights were trained ON, so a
            # resume against a different or empty corpus is not a resume at
            # all: it silently continues on different data. Refuse loudly
            # and name the fix.
            raise SystemExit(
                f"refusing to train: {corpus_path} has no usable rows, so "
                f"there is nothing to resume on. A resume must continue the "
                f"SAME corpus the checkpoint was trained on -- copy the "
                f"original corpus.txt (and the matching perply_targets.npz) "
                f"into {corpus_path.parent}, or point --run-dir at the "
                f"original run directory.")
        # The outcome tag must span the WHOLE window, not one ply.
        #
        # This was PLY_LEN (10 bytes) against a ctx+1 (193 byte) window.
        # build_outcome_batch copies tag[start:start+T] into the (B,T) weight
        # grid, so a 10-byte tag filled only positions 0..9 and left the
        # rest at zero. The move decisions live at positions 1, 11, 21, ...
        # 191 -- i.e. the labelled final position was never in the weighted
        # region. _weighted_move_loss then hit `w.sum() <= 0` and returned
        # `logits.sum() * 0.0`: the 7-WAY MOVE LOSS WAS EXACTLY ZERO.
        #
        # Consequence: every run from V15 to V20 trained the model on the
        # 256-way byte alphabet ONLY. It learned to spell Connect Four
        # transcripts and was never taught to play -- which is why the
        # policy collapsed onto the most common move byte ('F' = column 5,
        # ~94% of picks) and why holdout sat at ~30.6% regardless of
        # targets, temperature, or label coverage.
        #
        # Imported rows carry no outcome/role information (they are all
        # solver-labelled positions, not games with a winner), so every
        # position in the window gets weight 1.0.
        _tag_len = args.ctx + 1
        outcome_tags = [torch.ones(_tag_len, dtype=torch.float32)
                        for _ in examples]
        corpus_size = len(examples)
        consumed_lines = len(raw_lines)
        print(f"loaded corpus: {len(examples)} sequences, "
              f"{corpus_path.stat().st_size} bytes "
              f"(soft-w {getattr(args, 'soft_w', 0.0)})", flush=True)
        soft_bank = _align_soft_targets(args, raw_lines, version)
        if soft_bank is not None:
            print(f"soft targets: {tuple(soft_bank.shape)}", flush=True)
            # A position dataset is FIXED: the sidecar has exactly one row
            # per corpus line. Any corpus growth (self-play/teacher batches)
            # pushes len(examples) past soft_bank, and the sidecar then
            # addresses the wrong rows -- an off-by-N that trains on the
            # label of a different position. Refuse rather than corrupt.
            if args.selfplay_games or args.teacher_games:
                raise SystemExit(
                    "--load-corpus with a soft-target sidecar requires "
                    "--selfplay-games 0 --teacher-games 0: corpus growth "
                    "would desynchronise the sidecar from the corpus.")
        perply = _align_perply_targets(args, raw_lines, version)
        if perply is not None:
            _pt, _po, _np_ = perply
            print(f"per-ply targets: {tuple(_pt.shape)} "
                  f"(mean labelled plies/row "
                  f"{float(_np_.float().mean()):.1f})", flush=True)
            if args.selfplay_games or args.teacher_games:
                raise SystemExit(
                    "--load-corpus with a per-ply sidecar requires "
                    "--selfplay-games 0 --teacher-games 0: corpus growth "
                    "would desynchronise the sidecar from the corpus.")
            # The promotion gate must measure the OBJECTIVE, not the byte
            # bpb.  Byte bpb scores the PLAYED move; the loss scores the
            # TEACHER's label.  When no holdout and no ladder measurement
            # exist, the old gate fell back to byte bpb and promoted the
            # checkpoint that fits the labels LEAST (measured: step 1500 at
            # 62.82 % label agreement was promoted over step 3923 at
            # 91.29 %).
            perply_scorer = PerPlyAgreement(raw_lines, _pt, _po, _np_,
                                             device, n_rows=512,
                                             ctx=args.ctx)
        _cp = _column_prior_from_targets(perply)
        if _cp is not None and args.col_prior > 0.0:
            # Logit adjustment: subtract tau * log p_teacher(c) from column
            # c's logit. tau=1 fully removes the teacher's marginal column
            # prior; tau=0 leaves the historical behaviour untouched.
            print(f"column prior (logit adjustment, tau={args.col_prior}): "
                  + " ".join(f"c{c}={float(_cp[c]):+.3f}"
                             for c in range(7)), flush=True)
            col_prior = args.col_prior * _cp
        else:
            col_prior = None
    elif args.resume and corpus_path.exists() and corpus_path.stat().st_size > 0:
        # The corpus lines ARE the encoded training sequences, so the on-disk
        # file can be replayed as examples without re-deriving Game records.
        #
        # The dense rows do not carry a winner marker, but they DO carry the
        # full board, so replaying recovers who won and whether the winner
        # played sensibly. Without this the only option was a flat 0.5 tag,
        # which trains on loser moves and on 1,500 outcome-less rows at half
        # strength. The gate below grades in memory and drops the unusable
        # rows from the TRAINING SET only -- corpus.txt is never rewritten,
        # and every exclusion is counted so the loss is auditable.
        raw_lines = [ln for ln in corpus_path.read_bytes().splitlines() if ln]
        # A corpus written by another transcript version is not "slightly
        # worse" -- every row is a different byte alphabet, so the loss
        # becomes noise while still looking plausible. Refuse it here rather
        # than debugging a mysterious flat curve later. corpus.txt is never
        # rewritten; a v2 run simply starts its own file.
        if raw_lines:
            probe = next((ln for ln in raw_lines if len(ln) >= PLY_LEN), None)
            if probe is not None:
                try:
                    validate_stream(probe, version)
                except ValueError as e:
                    raise SystemExit(
                        f"refusing to resume: {corpus_path} is not a "
                        f"{version!r} corpus ({e}). Point --run-dir at a "
                        f"fresh directory, or pass --transcript to match it.")
        graded_games, qreport = quality_gate(raw_lines, clean=not args.keep_low_quality)

        # ---- behaviour gate -------------------------------------------------
        # The tactical grade above answers "did the winner ever pass up a
        # win?". That is NOT the same question as "is the next byte on this
        # row the right next byte", and it was measured to be the wrong test:
        # legacy rows average 0.37-0.41 winner-ply agreement with a strong
        # reference policy, against 0.67-0.74 for genuine teacher games and
        # 0.19 for pure random play. The corpus is largely random-grade, and
        # a blunder-free random game still grades GOOD.
        #
        # So rows are additionally required to actually PLAY like a teacher.
        # Measured on this corpus, requiring >=60% winner-ply agreement with
        # lookahead keeps ~1 row in 4 (vs 1 in 15 under the tactical gate
        # alone) -- the survivors are the genuinely well-played games.
        if not args.keep_low_quality:
            beh_games, breport = behavior_gate_games(
                graded_games,
                policy_name=args.behavior_ref,
                min_winner_agreement=args.min_agreement,
                verbose=True,
            )
            print(f"  behaviour gate: {len(graded_games)} tactical-graded rows "
                  f"-> {len(beh_games)} kept as demonstrations", flush=True)
            graded_games = beh_games
            qreport["behavior"] = breport

        if not args.load_corpus:
            examples = games_to_examples(graded_games, version)
            soft_bank = _align_soft_targets(args, graded_games, version)
        outcome_tags = action_consistent_examples(graded_games, version=version)
        # V12: every row loaded from disk at resume is FILTERED LEGACY, not a
        # teacher demonstration. Track that so `build_batch` can enforce the
        # teacher-dominant mix. corpus.txt is untouched; this is sampling-only.
        is_teacher = [False] * len(examples)
        corpus_size = len(examples)
        print(f"resumed corpus: {corpus_path.stat().st_size} bytes on disk, "
              f"{len(raw_lines)} rows -> {corpus_size} training sequences", flush=True)
        if args.gate_report:
            Path(args.gate_report).write_text(json.dumps(qreport, indent=2),
                                              encoding="utf-8")
            print(f"  quality gate report -> {args.gate_report}", flush=True)
        # The next append must not double-count rows the gate excluded: track
        # how many on-disk lines are already represented in `examples`.
        consumed_lines = len(raw_lines)
    else:
        # WHY THIS IS NOT RANDOM. The old seed was `random_seed_games`: both
        # sides played uniformly at random, and `winners_only` then kept the
        # WINNER'S side of that game. So every seed row was "imitate random
        # play that happened to win". Measured against a heuristic policy
        # those moves agree only 15.6% of the time against a 14.3% chance
        # floor -- statistically indistinguishable from random. Training on
        # it teaches the model to play badly and call it exemplary.
        #
        # `teacher` seeds from the same solver used for the growth cycles, so
        # the run starts with competent demonstrations instead of noise. It is
        # slower (solver self-play is CPU-bound) but it is the only seed that
        # is worth imitating.
        if args.seed_source == "teacher":
            seed_games = teacher_seed_games(args.seed_games, rng)
            corpus = [g for g in seed_games]
            seed_tag = "teacher"
        else:
            seed_games = random_seed_games(args.seed_games, rng)
            corpus = winners_only(seed_games)
            seed_tag = "random"
        corpus_size = len(corpus)
        lines = games_to_lines(corpus)
        corpus_path.write_bytes(b"\n".join(lines) + b"\n" if lines else b"")
        print(f"seed corpus ({seed_tag}): {stats(seed_games)} -> "
              f"{corpus_size} decided games "
              f"({corpus_path.stat().st_size} bytes)", flush=True)
        examples = games_to_examples(corpus, version)
        # We DO know the winner for the seed games -- tag them properly.
        outcome_tags = action_consistent_examples(corpus, version=version)
        # Uniform-random seed games are the weakest possible demonstrations:
        # mark them legacy so the mix cap treats them the same way. A teacher
        # seed is a genuine demonstration and must be labelled as one, or
        # `--max-legacy-frac 0` would drop the only competent rows we have.
        if seed_tag == "teacher":
            is_teacher = [True] * len(examples)
        else:
            is_teacher = [False] * len(examples)

    eval_every = args.eval_every
    save_every = args.save_every

    # How many rows were on disk before the gates filtered them. 0 means
    # there was no corpus at all (a fresh seed phase, or a resume whose
    # run directory never received the corpus).
    n_rows_on_disk = consumed_lines

    # ---- no-training-data guard -----------------------------------------
    # Every path above can legitimately end with ZERO training sequences:
    #   * a resume whose run-dir has no corpus.txt (nothing to continue on);
    #   * a resume whose corpus was filtered to nothing by the quality /
    #     behaviour gates;
    #   * a seed phase that produced no games.
    #
    # The old behaviour was to fall through to `build_batch`, which called
    # `rng.integers(0, 0)` and died with `ValueError: high <= 0` -- a
    # traceback three frames away from the actual cause, and it happened
    # AFTER a "[snapshot saved -> step6000]" line, which made it look like
    # a resume had worked.
    #
    # An empty training set is never a valid state: refuse loudly and say
    # what to do instead.
    if not examples:
        if args.resume:
            why = (
                f"The corpus at {corpus_path} produced no usable rows"
                + (f" after the quality/behaviour gates ({n_rows_on_disk} "
                   f"rows on disk)" if n_rows_on_disk else
                   " (the run directory has no corpus, or it is empty)")
                + ". A resume must continue the SAME corpus the checkpoint "
                  "was trained on -- copy the original corpus.txt (and the "
                  "matching perply_targets.npz) into the run dir, or point "
                  "--run-dir at the original run directory. To relax the "
                  "gates, pass --keep-low-quality.")
        else:
            why = ("The seed corpus produced no games. Raise --seed-games, "
                   "or pass --load-corpus with a real corpus file.")
        raise SystemExit(
            "refusing to train: the training set is EMPTY (0 sequences). "
            + why)
    print(f"training set: {len(examples)} sequences "
          f"({sum(1 for t in is_teacher if t)} teacher)", flush=True)
    # The held-out set is the DECISION metric: positions the model has never
    # seen, scored by whether the argmax is one of the optimal columns (ties
    # count as correct). bpb and in-sample legal accuracy can both move while
    # this stays flat, which is exactly what V15 did -- it degraded on its
    # training objective while we could not tell whether generalisation held.
    # Without this the holdout files are parsed and then thrown away.
    holdout = None
    if args.holdout_corpus:
        if not args.holdout_targets:
            raise SystemExit("--holdout-corpus given without --holdout-targets")
        holdout = Holdout(args.holdout_corpus, args.holdout_targets, device)
        print(f"holdout: {holdout.n} positions "
              f"(scoring a random {min(holdout.n, 4000)} each eval)", flush=True)
    else:
        print("holdout: NONE -- in-sample metrics only, not a decision metric",
              flush=True)

    # ---- HOLDOUT per-ply label agreement -----------------------------------
    # The promotion key's PRIMARY term is per-ply label agreement, but the
    # scorer samples the TRAINING corpus.  That measures fit, not
    # generalisation: a model can keep its training agreement rising while
    # its held-out agreement is flat or falling, and nothing in the metrics
    # would say so.
    #
    # The holdout sidecar has the SAME schema as the training sidecar
    # (target / ply_of / teacher / n_plies), so the same scorer applies.
    # This is a DIAGNOSTIC and an ABORT input, not a promotion key: the
    # promoted artifact is chosen on the training-corpus metric the
    # objective defines, and the holdout tells us whether that metric is
    # still tracking generalisation.
    holdout_perply_scorer = None
    # History of (step, holdout_label_agreement) used by the abort rules.
    _holdout_label_history = []
    if args.holdout_corpus and args.holdout_targets:
        try:
            import numpy as _np
            _h = _np.load(args.holdout_targets)
            _h_raw = [ln for ln in Path(args.holdout_corpus).read_bytes()
                      .splitlines() if ln]
            if len(_h_raw) != len(_h["target"]):
                raise SystemExit(
                    f"holdout per-ply: corpus has {len(_h_raw)} rows but the "
                    f"sidecar has {len(_h['target'])}; refusing to score "
                    f"misaligned data.")
            holdout_perply_scorer = PerPlyAgreement(
                _h_raw, _h["target"], _h["ply_of"], _h["n_plies"], device,
                n_rows=min(512, len(_h_raw)), seed=32, ctx=args.ctx)
            print(f"holdout per-ply: {len(_h_raw)} rows, "
                  f"scored every eval (disjoint from the training corpus)",
                  flush=True)
        except SystemExit:
            raise
        except Exception as e:
            print(f"holdout per-ply: DISABLED ({e})", flush=True)
    # Promotion key: (move-byte accuracy, -hard bpb, gameplay score, distinct
    # ladder games). Seed it from the checkpoint we are resuming so the first
    # eval after a resume can never demote the model we are continuing from --
    # otherwise a run resumed from a strong checkpoint would need to beat it
    # strictly to keep writing checkpoints at all. Every seed below MUST be a
    # 4-tuple: a 3-tuple raises TypeError when compared against `sel`.
    # 4-tuple. PRIMARY slot is now the LADDER SCORE (net wins vs `heuristic`),
    # not held-out accuracy -- see the `sel` construction further down for why.
    best_sel = (-float("inf"), -1.0, -float("inf"), 0.0)
    # The last MEASURED ladder score (net wins vs `heuristic`), kept so a
    # cheap eval step -- one where `opp_results` is empty -- inherits the
    # previous ladder value instead of a score against a different
    # opponent. Seeded from the resumed checkpoint so a resumed run does
    # not start by demoting itself.
    _last_ladder_net = -float("inf")
    _ste_lr_dropped = False
    if args.resume:
        _meta = ckpt_meta
        if isinstance(_meta, dict):
            _seed_distinct = float(_meta.get("distinct", 0) or 0)
            if _meta.get("hard_acc") is not None:
                # Same 4-tuple shape as `sel`: (ladder_net, holdout,
                # -hard_bpb, distinct). The PRIMARY slot used to be holdout,
                # so seeding it from a holdout/accuracy-derived number would
                # immediately demote the checkpoint we resumed from -- with the
                # old shape best_sel[0] would be 35.1 (a holdout percentage)
                # and every ladder_net comparison would lose.
                # `ladder_net` therefore reads its own metadata key and falls
                # back to -inf, NOT 0: a missing key must never look like
                # "lost every game", or the run can never promote again.
                best_sel = (float(_meta.get("ladder_net", -float("inf"))),
                            float(_meta.get("holdout_optimal_pct", -1.0)),
                            -float(_meta.get("hard_bpb", float("inf"))),
                            _seed_distinct)
                # Same value, but on the LADDER SCALE -- and only if the
                # checkpoint was MEASURED with the same protocol this run
                # uses. `ladder_net` is a signed net-win count, so its
                # meaning depends entirely on the opponent and on whether
                # the model played greedily. A value produced against the
                # centre-column reference, or by a SAMPLED model, is not
                # comparable to a greedy-vs-`heuristic` ladder net.
                #
                # V22 is the concrete failure: it resumed from V21's
                # `best_soft.pt`, whose `ladder_net=1` was a centre-rule
                # score, while V22's own greedy-vs-heuristic ladder nets
                # were around -176. The gate could therefore NEVER promote,
                # and no `best_hard.pt` was ever written. Only the periodic
                # snapshots survived -- which is exactly why the
                # checkpoint-survival fix exists.
                #
                # So: only seed the primary slot when the checkpoint
                # declares the same protocol. Otherwise seed -inf, meaning
                # "unknown on this scale" -- the run then promotes on its
                # first real ladder measurement, which is the only
                # defensible choice.
                _seed_ladder = _meta.get("ladder_net")
                _same_protocol = (
                    _meta.get("ladder_opp") == LADDER_PRIMARY_OPP
                    and bool(_meta.get("ladder_greedy"))
                )
                if not _same_protocol or _seed_ladder is None:
                    _seed_ladder = -float("inf")
                    print("promotion gate: checkpoint ladder score is NOT on "
                          f"the '{LADDER_PRIMARY_OPP}'/greedy scale "
                          f"(meta ladder_opp={_meta.get('ladder_opp')!r}, "
                          f"ladder_greedy={_meta.get('ladder_greedy')!r}) "
                          "-- seeding the primary slot with -inf so this run "
                          "promotes on its first real ladder measurement.",
                          flush=True)
                best_sel = (float(_seed_ladder),
                            float(_meta.get("holdout_optimal_pct", -1.0)),
                            -float(_meta.get("hard_bpb", float("inf"))),
                            _seed_distinct)
                _last_ladder_net = float(_seed_ladder)
                print(f"promotion gate seeded from checkpoint: "
                      f"ladder_net={best_sel[0]:.1f} holdout={best_sel[1]:.1f} "
                      f"hard_bpb={best_sel[2]:.3f} distinct={best_sel[3]:.0f}",
                      flush=True)
            elif _meta.get("score") is not None:
                best_sel = (float(_meta["score"]), -1.0,
                            -float("inf"), _seed_distinct)
    t0 = time.time()
    metrics_path = run_dir / "metrics.jsonl"
    if not args.resume:
        metrics_path.write_text("")  # truncate at start

    def log_metrics(row: dict) -> None:
        """Append a single JSON line for live graphing / timeline review."""
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def log_eval_panel(step: int, soft_bpb: float, hard_bpb: float,
                       soft_acc: float, hard_acc: float, legal_acc: float,
                       play: dict, opp_results: dict,
                       ho: dict | None = None,
                       ppa: dict | None = None,
                       hppa: dict | None = None) -> None:
        """Print the human-readable one-line panel used during the run."""
        # `distinct` is the anti-memorisation guard: it says how many of the
        # nominal games were genuinely different trajectories. A high win rate
        # with distinct << games is one memorised line replayed, not strength
        # (the old run showed exactly that: 48 games, 2 distinct).
        opp_str = " ".join(f"{name} W{r['wins']}L{r['losses']}D{r['draws']}"
                          f"(d{r.get('distinct', '?')})"
                          for name, r in opp_results.items())
        elapsed = time.time() - t0
        # `legal` is argmax over the LEGAL columns only -- the metric that
        # matches the masked 7-way training objective. `hard acc` is argmax
        # over all 256 byte values, which the 7-way loss no longer optimises
        # directly; the two can move in OPPOSITE directions, so read `legal`
        # as the policy metric and the others as byte-alphabet diagnostics.
        line = (f"step {step:5d} | soft {soft_bpb:.3f} hard {hard_bpb:.3f} bpb "
                f"| byte soft {soft_acc:.0%} hard {hard_acc:.0%} "
                f"| LEGAL {legal_acc:.0%} "
                f"| vs-ref W{play['wins']}L{play['losses']}D{play['draws']} "
                f"| {opp_str} "
                + (f"| HOLDOUT {ho['holdout_optimal_pct']:.1f}% "
                   f"(edge {ho['edge_optimal_pct']:.1f}%)" if ho else "")
                + f" | corpus {corpus_size} | t {elapsed:.0f}s"
                + (f" | LABEL {ppa['perply_agreement_pct']:.1f}%"
                   f" (n={ppa['perply_n']})" if ppa else "")
                + (f" | H-LABEL {hppa['perply_agreement_pct']:.1f}%"
                   f" (n={hppa['perply_n']})" if hppa else ""))
        print(line, flush=True)

    for step in range(start_step, args.steps + 1):
        # ---------------- periodic evaluation + checkpoint ----------------
        if step % eval_every == 0 or step == args.steps:
            # --- light bpc on a small sample (every step) ---
            bpc_sample = examples[::max(1, len(examples) // 64)] or examples
            bpc_sample = bpc_sample[:64]
            (soft_bpb, hard_bpb, soft_acc, hard_acc,
             legal_acc) = evaluate_bpc(
                model, bpc_sample, device, chunk=min(16, len(bpc_sample)))
            # --- vs the historical "centre column" reference ---
            if step < args.steps // 2:
                temp = args.temp_start
            else:
                frac = (step - args.steps // 2) / max(1, args.steps - args.steps // 2)
                temp = args.temp_start + (args.temp_end - args.temp_start) * frac
            # Evaluation play uses its OWN temperature, not the sampling
            # temperature. The old code graded the model at `temp_start`=1.2,
            # i.e. it picked moves semi-randomly and we scored its stumbles.
            # A next-byte predictor deployed in the UI is played greedily
            # (argmax over legal columns), so that is how we must measure it.
            play = evaluate_play(model, device, args.eval_games, rng,
                                 temperature=args.eval_temp)
            score = play["wins"] - play["losses"]
            elapsed = time.time() - t0

            # Held-out optimal-move accuracy. Cheap (4k rows, one forward
            # each) and it is the number that decides whether a checkpoint is
            # real, so it is computed FIRST and everything else may read it.
            ho = holdout.score(model) if holdout is not None else None

            # --- PER-PLY LABEL AGREEMENT (the training objective) -------
            # The promotion key must measure what the LOSS measures.  Byte
            # bpb scores the PLAYED move; the masked 7-way loss scores the
            # TEACHER's label.  When the gate had neither a holdout nor a
            # ladder measurement it fell back to byte bpb and promoted the
            # checkpoint that fits the labels LEAST -- measured on the
            # 500-row acceptance run: step 1500 (62.82 % label agreement,
            # byte bpb 5.513) was promoted over step 3923 (91.29 % label
            # agreement, byte bpb 6.798).  This scorer removes that
            # inversion: it reads the real sidecar, at the real ply, with
            # the real legal mask.
            if perply_scorer is not None:
                # Score the HARD export: it is the artefact that ships and
                # the one the UI serves.  Export it from the LIVE model here
                # (the model in memory has the real trained parameters; a
                # checkpoint re-loaded through `_load_agent` would not --
                # GFMarkovLM_V6 has no `import_hard`, so its float buffers
                # stay zero and `export_hard()` on it is an all-zero model).
                perply_scorer.hard = _hard_export(model)
                ppa = perply_scorer.score(model)
            else:
                ppa = None

            # --- HOLDOUT per-ply agreement (generalisation) -----------------
            # Same scorer, same hard export, but on the DISJOINT holdout.
            # The training-corpus number above is fit; this one is
            # generalisation.  It is a diagnostic and an abort input, NOT a
            # promotion key -- the promoted artifact stays the one that best
            # fits the objective the training run optimises.
            hppa = None
            if holdout_perply_scorer is not None:
                try:
                    holdout_perply_scorer.hard = _hard_export(model)
                    hppa = holdout_perply_scorer.score(model)
                except Exception as e:
                    print(f"[holdout per-ply score failed: {e}]", flush=True)
                    hppa = None

            # --- checkpoint selection metric -------------------------------
            # Historically we promoted a checkpoint only when it beat the last
            # saved one against the weak "centre column" reference bot, played
            # at high temperature. That gate threw away steps 6000 and 7000 --
            # the two most accurate models this run ever produced (move-accuracy
            # 54% and 60%, bpb 1.70 and 1.48) -- purely because that reference
            # bot's score dipped. `hard_acc` is measured on real move bytes
            # with illegal columns masked out, so it is the metric that
            # actually tracks play quality. Primary key = hard_acc, tie-broken
            # by a lower hard_bpb, then by the gameplay score.
            # A ladder game that repeats is not evidence. The old run resolved
            # 48 nominal games against `heuristic` down to 2 distinct full
            # sequences on a single opening ([2,3,2] in 48/48), which is how a
            # memorised line produced a fake "W47-L1". So the OPPONENT is
            # played with per-ply exploration and every finished game is hashed.
            # `best_distinct` is a TIE-BREAK ONLY: it is compared after hard_acc
            # and hard_bpb, so it can never promote a weaker-accuracy model -- it
            # can only stop a strictly-better-accuracy model from losing a tie
            # to a degenerate one.
            opp_results = {}
            best_distinct = 0
            if (save_every > 0 and step % save_every == 0) or step == args.steps:
                # Same reasoning as the reference-bot play above: measure at
                # eval temperature, not at the (much noisier) sampling
                # temperature, so the ladder W/L is comparable run to run.
                opp_results = _eval_opponents(model, device,
                                              {"random": 0, "heuristic": 0},
                                              args.eval_games, rng,
                                              args.eval_temp,
                                              version=version)
                if opp_results:
                    best_distinct = min(
                        r.get("distinct", 0) or 0
                        for r in opp_results.values())

            # Primary key is LEGAL accuracy (argmax over the legal columns),
            # because that is what the masked 7-way loss optimises and what
            # the agent actually does at serve time. `hard_acc` (argmax over
            # all 256 byte values) is kept only as a tie-break, so a run still
            # compares sanely against checkpoints trained with the old byte
            # loss.
            # Promotion key. When a holdout is loaded it becomes the PRIMARY
            # key, because it is the only number here measured on positions the
            # model has not seen. V15 promoted on in-sample `legal_acc` while
            # bpb climbed 8->18, i.e. the in-sample signal pointed the wrong
            # way. Keep the 4-tuple shape: every seed and every comparison
            # below must stay a 4-tuple or the `>` raises TypeError.
            ho_pct = ho["holdout_optimal_pct"] if ho else -1.0
            # Promotion key. PRIMARY is now the HARD-MODEL LADDER SCORE
            # against `heuristic` -- net wins. This is a deliberate reversal
            # of the previous holdout-first key, and the reason is visible
            # in the V17 log: from step 2000 to 8000 the ladder improved
            # every single evaluation (W170 L629 -> W213 L587 vs heuristic,
            # W610 L189 -> W657 L129 vs random) while held-out optimal-move
            # accuracy went 35.1% -> 32.9% and bpb climbed 13.9 -> 34.4.
            #
            # A per-position argmax test cannot see multi-ply consequences, so
            # it is flat-to-declining exactly when the model is getting better
            # at playing the game. Promoting on it discarded 6,000 steps of
            # genuine improvement: best_hard.pt/best_soft.pt were last written
            # at 08:49 (step ~1500) and never again through step 8500.
            #
            # Why the ladder score must be computed BEFORE this line, always:
            # `opp_results` is only populated when `step % save_every == 0`,
            # so on a cheap eval step it is `{}` and `score` would silently
            # read 0 -- making every non-save eval look like a catastrophic
            # regression.
            #
            # SCALE CONSISTENCY (fixed 2026-10-23). The fallback must be the
            # LAST MEASURED LADDER score, not `score`. `score` is the
            # vs_ref net-wins against the *centre column* reference -- a much
            # weaker opponent than `heuristic` -- so the two numbers are not
            # on the same scale. Mixing them made the promotion gate compare
            # "net wins vs heuristic" against "net wins vs centre" and
            # promote on the OPTIMISTIC one. Concretely, in V21 the gate
            # picked step 2900 because a cheap eval step returned +1 against
            # the centre rule, which outranked every real ladder
            # measurement (-288, -324, -325, -330). The chosen checkpoint
            # turned out to be genuinely the strongest (the independent
            # 2,000-game ladder confirmed it), but that was luck: the gate
            # was ranking a centre-rule score against heuristic scores.
            #
            # The fix keeps ONE scale (net wins vs `heuristic`) for the
            # primary key at every step, so a cheap eval step inherits the
            # last measured ladder value instead of a different opponent's
            # score. Holdout then acts as the tie-break between ladder
            # measurements, which is the intended design.
            _ladder_net = _last_ladder_net
            if opp_results:
                _he = opp_results.get("heuristic")
                if _he is not None:
                    _ladder_net = _he["wins"] - _he["losses"]
                    _last_ladder_net = _ladder_net

            # PRIMARY KEY: per-ply LABEL AGREEMENT when a per-ply sidecar is
            # loaded.  This is the quantity the loss actually optimises, and
            # it is measured on the shipped (hard) export with the shipped
            # legal mask.  It REPLACES the byte-bpb fallback that used to
            # promote the least label-faithful checkpoint.  The ladder
            # score and holdout remain as tie-breaks in the same 4-tuple
            # shape, so every seeded/comparison site stays consistent.
            if ppa is not None and ppa.get("perply_n", 0) > 0:
                _primary = float(ppa["perply_agreement_pct"])
            else:
                _primary = float(_ladder_net)
            sel = (_primary, ho_pct, -hard_bpb, float(best_distinct))

            # --- BOARD-SENSITIVITY GATE -------------------------------
            # The artifact that ships is `export_hard()` -- the one-hot
            # forward. A checkpoint whose exported model does not react to a
            # change of the BOARD is a constant-column policy wearing the
            # costume of a learner: it scores on weak opponents and loses
            # every real game. Measured over V23's checkpoints, the exported
            # model was board-blind (logit delta exactly 0.0000) for every
            # step before --hard-start, and only read the board afterwards.
            #
            # So: measure it, and refuse to promote a blind checkpoint. A
            # blind checkpoint is worse than no checkpoint -- it looks
            # plausible in every metric that does not require seeing the
            # stones.
            _board_sens = _board_sensitivity(model, device, version,
                                             corpus_rows=examples)
            if _board_sens is not None and _board_sens < 1e-3:
                print(f"[board-gate] step {step}: exported model is "
                      f"BOARD-BLIND (sensitivity {_board_sens:.6f}) -- "
                      f"refusing to promote this checkpoint.", flush=True)
                # Push the primary slot below anything measurable so the
                # comparison below cannot select it.
                sel = (-float("inf"), ho_pct, -hard_bpb, float(best_distinct))
            elif _board_sens is not None:
                print(f"[board-gate] step {step}: board sensitivity "
                      f"{_board_sens:.4f} -- model reads the board.", flush=True)

            log_metrics({
                "step": step, "elapsed_s": round(elapsed, 1),
                "soft_bpb": soft_bpb, "hard_bpb": hard_bpb,
                "soft_acc": soft_acc, "hard_acc": hard_acc,
                "legal_acc": legal_acc,
                **(ho or {}),
                **(ppa or {}),
                **({f"holdout_{k}": v for k, v in (hppa or {}).items()}),
                "vs_ref": play, "vs_opp": opp_results,
                "corpus_size": corpus_size, "temp": temp,
            })
            log_eval_panel(step, soft_bpb, hard_bpb, soft_acc,
                                       hard_acc, legal_acc, play, opp_results, ho,
                                       ppa, hppa)

            # ---------------- ABORT RULES ----------------------------------
            # These stop a run that is provably not learning, so nobody
            # waits 18 hours for a checkpoint that was never going to
            # improve.  They are deliberately CONSERVATIVE, because a
            # WRONG abort on an 18-hour run is far more expensive than a
            # run that goes on for an extra two hours:
            #
            #   * they only arm after a warm-up (step >= --abort-warmup),
            #     so the first hours of a run can never trigger one;
            #   * the flat rule needs SEVERAL consecutive flat evaluations,
            #     not two, and a low absolute score;
            #   * the dead-column rule needs the column to be genuinely
            #     present in the labels AND never picked AT ALL.
            #
            # Rule 1 (holdout flat): the DISJOINT holdout label agreement
            #   has not improved over `--abort-flat-evals` consecutive
            #   evaluations.  This is the generalisation signal, so it is
            #   the one that matters.  Guard: only after the warm-up, and
            #   only if the BEST holdout value so far is still low -- a run
            #   that has already reached a good score and merely plateaued
            #   is a "converged" signal, not a failure.
            # Rule 2 (dead column): a column that IS optimal in the
            #   holdout labels (so the data contains it) is NEVER picked
            #   across the whole holdout evaluation.  That is the
            #   "c1/c2 picks at 0" failure.  Guard: the column must have
            #   >= 50 optimal opportunities in the holdout labels, so a
            #   column that is merely rare cannot trip it.
            #
            # A column that is ABSENT from the labels is a DATA problem,
            # not a model problem, and rule 2 must not fire on it -- that
            # is exactly why the label histogram is needed.
            _abort_reasons = []
            if hppa and holdout_perply_scorer is not None:
                _h_agree = hppa.get("perply_agreement_pct", 0.0)
                _h_hist = _holdout_label_history
                if _h_hist is None:
                    _h_hist = []
                _h_hist.append((step, _h_agree))
                _flat_needed = max(2, int(getattr(args, "abort_flat_evals", 4)))
                _warmup = int(getattr(args, "abort_warmup", 20000))
                _armed = step >= _warmup
                # Rule 1
                if _armed and len(_h_hist) >= _flat_needed:
                    _best_h = max(v for _, v in _h_hist)
                    _recent = [v for _, v in _h_hist[-_flat_needed:]]
                    _flat = (max(_recent) - min(_recent)) < 0.5
                    if _flat and _best_h < float(
                            getattr(args, "abort_flat_below", 55.0)):
                        _abort_reasons.append(
                            f"holdout label agreement flat at "
                            f"{_recent[-1]:.2f}% for {_flat_needed} consecutive "
                            f"evaluations (best {_best_h:.2f}%, "
                            f"n={len(_h_hist)})")
                # Rule 2
                # The same warm-up guard applies.  At step 0 the model is
                # RANDOM, so of course it never picks c4 -- an unguarded
                # rule-2 abort fired immediately at step 0 on a randomly
                # initialised model and killed the run before it trained a
                # single step.  A dead-column finding is only meaningful
                # once the model has had time to learn the other columns.
                _lc = hppa.get("perply_label_cols") or []
                _lh = hppa.get("perply_label_col_hit") or []
                _pk = hppa.get("perply_picks") or []
                for c in range(7):
                    if not _armed:
                        break
                    if _lc and c < len(_lc) and _lc[c] >= 50:
                        if _lh[c] == 0 and (_pk[c] if c < len(_pk) else 0) == 0:
                            _abort_reasons.append(
                                f"column c{c} is optimal in {_lc[c]} holdout "
                                f"labels but was NEVER picked")
            if _abort_reasons:
                _msg = ("; ".join(_abort_reasons))
                print(f"[ABORT] step {step}: {_msg}", flush=True)
                try:
                    (run_dir / "ABORT.txt").write_text(
                        f"step {step}\n{_msg}\n")
                except Exception:
                    pass
                if getattr(args, "abort_on", "warn") == "abort":
                    raise SystemExit(f"[ABORT] step {step}: {_msg}")
                # Default is "warn": the finding is recorded and the run
                # CONTINUES.  An abort is a hard stop of an 18-hour run,
                # so it must be explicitly opted into -- a false positive
                # (e.g. the step-0 random model that never picks c4) is
                # far more expensive than a run that continues past a
                # recoverable plateau.
                print("[ABORT] not stopping (--abort-on warn); run continues",
                      flush=True)

            # --- periodic weight save, INDEPENDENT of the promotion gate ---
            # `--save-every` only used to trigger the ladder eval and the text
            # snapshot; it never wrote weights, so stopping a run destroyed
            # every step since the last promotion. V17 lost its best measured
            # model (step 11000, 86.1% vs random) exactly this way.
            if save_every > 0 and step % save_every == 0:
                try:
                    save_snapshot(model, run_dir, device, f"step{step}",
                                      {"step": step, "score": score,
                                       "ladder_net": _ladder_net,
                                       "ladder_opp": LADDER_PRIMARY_OPP,
                                       "ladder_greedy": True,
                                       "ladder_protocol": LADDER_PROTOCOL_VERSION,
                                   "hard_acc": hard_acc,
                                   "legal_acc": legal_acc,
                                   "hard_bpb": hard_bpb,
                                   "holdout_optimal_pct": ho_pct,
                                   "transcript": version,
                                   "d": d, "layers": layers,
                                   "fan_in": fan_in,
                                   "lags": list(model.lags),
                                   "coef_set": list(getattr(model, "coef_set", [])),
                                   "table_bias": float(args.table_bias),
                                   "model_seed": seed,
                                   "seed": args.seed},
                                   capture_train_state(opt, rng, nprng))
                    print(f"[snapshot saved -> step{step}]", flush=True)
                    _prune_checkpoints(run_dir, args)
                except Exception as e:  # pragma: no cover
                    print(f"[weight-snapshot error at step {step}: {e}]",
                          flush=True)

            # --- snapshot one eval board to a text log so you can see games ---
            if ((save_every > 0 and step % save_every == 0)
                    or step == args.steps) and opp_results:
                try:
                    snap = _play_one_snapshot(model, "heuristic", step, device, rng,
                                                    version=version)
                    (run_dir / "snapshots.txt").open("a").write(snap + "\n")
                except Exception as e:  # pragma: no cover
                    print(f"[snapshot error at step {step}: {e}]", flush=True)

            # Promote on move-byte accuracy (illegal columns masked out),
            # tie-broken by lower hard bpb and then by the gameplay score.
            # See the `sel` comment above: the old score-only gate discarded
            # the 54% and 60% models because a weak reference bot's W/L dipped.
            if sel > best_sel:
                best_sel = sel
                # Move to CPU for saving; restored right after so the
                # training loop stays on the accelerator.
                model.to("cpu")
                atomic_torch_save(export_hard_with_meta(model,
                                                meta={"step": step, "score": score,
                                     "ladder_net": _ladder_net,
                                     "ladder_opp": LADDER_PRIMARY_OPP,
                                     "ladder_greedy": True,
                                     "ladder_protocol": LADDER_PROTOCOL_VERSION,
                                     "hard_acc": hard_acc,
                                     "legal_acc": legal_acc,
                                     "hard_bpb": hard_bpb,
                                     "holdout_optimal_pct": ho_pct,
                                     "transcript": version,
                                     "d": d, "layers": layers,
                                     "fan_in": fan_in,
                                     "lags": list(model.lags),
                                     "coef_set": list(getattr(model, "coef_set", [])),
                                     "table_bias": float(args.table_bias),
                                     "model_seed": seed}),
                           run_dir / "checkpoints" / "best_hard.pt")
                # Also save the soft/float state alongside it, so the UI
                # and downstream tooling can run the soft path directly.
                # Before this, ``best_hard.pt`` was integer-only and
                # ``_load_agent`` had to rebuild the soft nn.Parameters.
                atomic_torch_save({"state": model.state_dict(),
                                            "meta": {"step": step, "score": score,
                                                     "ladder_net": _ladder_net,
                                                     "ladder_opp": LADDER_PRIMARY_OPP,
                                                     "ladder_greedy": True,
                                                     "ladder_protocol": LADDER_PROTOCOL_VERSION,
                                                     "hard_acc": hard_acc,
                                                     "legal_acc": legal_acc,
                                                     "hard_bpb": hard_bpb,
                                                     "holdout_optimal_pct": ho_pct,
                                                     "transcript": version,
                                                     "d": d, "layers": layers,
                                                     "fan_in": fan_in,
                                                     "lags": list(model.lags),
                                                     "coef_set": list(getattr(model, "coef_set", [])),
                                                     "table_bias": float(args.table_bias),
                                                     "model_seed": seed,
                                                     "seed": args.seed},
                                            **capture_train_state(opt, rng, nprng)},
                           run_dir / "checkpoints" / "best_soft.pt")
                # Put the model back on the accelerator. The `model.to("cpu")`
                # above only keeps the saved tensors small; without this restore
                # the next forward pass indexes `emb_logits` (now on cpu) with
                # cuda tokens and dies with "indices should be either on cpu or
                # on the same device". The comment above claimed this happened
                # right after the save -- it did not, and V17 never hit it only
                # because that process was running the pre-gate code.
                model.to(device)

        if step == args.steps:
            break

        # ---------------- LR drop at STE phase ----------------
        # This was `step == args.hard_start`, which can NEVER fire in a fresh
        # run: the step-0 evaluation runs `break` before this line is reached,
        # and with the usual --hard-start 0 the transition step is consumed by
        # that eval. A latch fixes it -- fire once on the first step at or
        # past hard_start, rather than on an exact equality that nothing hits.
        # The resume case still drops only when it actually crossed.
        #
        # With --hard-start 0 (the new default: train the shipped path from
        # step 0) there IS no separate STE phase, so there is nothing to
        # drop the learning rate for. Dropping it at step 1 would halve the
        # effective learning rate for the whole run. The drop only makes
        # sense when the hard phase starts LATE, i.e. hard_start > 0.
        if (not _ste_lr_dropped and args.hard_start > 0
                and step >= args.hard_start
                and (step > args.hard_start or start_step <= args.hard_start)):
            _ste_lr_dropped = True
            if start_step > args.hard_start:
                # Resuming past the STE transition: the original run already
                # dropped the LR once, so apply it here to stay on the same
                # schedule rather than training at 2x the intended LR.
                for pg in opt.param_groups:
                    pg["lr"] *= args.lr_drop
            print(f"\n--- step {step}: gentle STE phase ---", flush=True)

        # ---------------- corpus growth ----------------
        # V11: the data source is the negamax solver / heuristic teacher, NOT
        # the model's own self-play. Self-play from a model that loses 47/48
        # to a 2-ply heuristic contributed move-byte gradients for decisions
        # that were closer to random than to strategy. `--corpus-source selfplay`
        # restores the old behaviour for A/B comparison.
        if step % args.selfplay_every == 0 and args.corpus_source != "selfplay":
            new = teacher_batch_with_roles(args.teacher_games, step, args.seed)
            decided = winners_only([(m, w) for m, w, _ in new])
            roles = [r for (m, w, r) in new if w != 0]
            # Teacher games are strong, but "won" is not "played well", and
            # the heuristic seat wins about half of solver-vs-heuristic games
            # (measured W10 L10). Winner-only tagging would therefore label
            # the HEURISTIC's moves as the demonstration -- reproducing the
            # exact defect the legacy corpus has, but freshly generated.
            #
            # So teacher rows are trusted by ROLE (solver/lookahead seat) via
            # `action_consistent_examples(role_masks=...)`, not by seat-1-wins.
            # The behaviour gate is deliberately NOT applied here: these games
            # are generated BY the reference policies, so scoring them against
            # `lookahead` would reject every solver-played game by
            # construction.
            if decided:
                corpus.extend(decided)
                corpus_size += len(decided)
                n_t = sum(is_teacher)
                n_l = len(is_teacher) - n_t
                print(f"mix: teacher={n_t} legacy={n_l} "
                      f"({100.0 * n_t / max(1, len(is_teacher)):.1f}% teacher "
                      f"resident, sampling cap "
                      f"{100.0 * args.max_legacy_frac:.0f}% legacy/batch)",
                      flush=True)
                new_examples = games_to_examples(decided, version)
                examples.extend(new_examples)
                is_teacher.extend([True] * len(new_examples))
                outcome_tags.extend(
                    action_consistent_examples(decided, role_masks=roles,
                                                        version=version))
                new_lines = encode_games(decided, "dense", version=version)
                with corpus_path.open("ab") as f:
                    f.write(b"\n".join(new_lines) + b"\n")
                consumed_lines += len(new_lines)
                n_teacher_plies = sum(sum(1 for r in rs if r) for rs in roles)
                print(f"[teacher +{len(decided)} games] corpus={corpus_size} "
                      f"examples={len(examples)} "
                      f"(role-tagged, {n_teacher_plies} teacher plies)", flush=True)

        elif step % args.selfplay_every == 0:
            frac = step / max(1, args.steps)
            temp = args.temp_start + (args.temp_end - args.temp_start) * frac
            if args.corpus_mix == "mixed":
                new = mixed_opponent_games(model, device, args.selfplay_games,
                                           rng, temperature=temp)
            else:
                new = self_play_games(model, device, args.selfplay_games,
                                      rng, temperature=temp)
            decided = winners_only(new)
            if decided:
                corpus.extend(decided)
                corpus_size += len(decided)
                new_examples = games_to_examples(decided, version)
                examples.extend(new_examples)
                is_teacher.extend([False] * len(new_examples))
                outcome_tags.extend(action_consistent_examples(decided, version=version))
                new_lines = encode_games(decided, "dense", version=version)
                with corpus_path.open("ab") as f:
                    f.write(b"\n".join(new_lines) + b"\n")
                consumed_lines += len(new_lines)

        # ---------------- training step ----------------
        # snap_to_end pins the window to the END of the transcript. It is
        # required whenever a SIDECAR is loaded, because the sidecar's
        # labels are anchored to specific plies and a random window could
        # miss them. The per-ply sidecar must get the same treatment: its
        # labels are the LAST plies, so a random window would usually
        # contain none of them.
        snap = (soft_bank is not None) or (perply is not None)
        x, y, chosen, starts = build_batch(examples, args.batch, device,
                                    nprng, args.ctx,
                                    is_teacher=is_teacher,
                                    max_legacy_frac=args.max_legacy_frac,
                                    snap_to_end=snap)
        # Same windowing, tag slab (T-1 long). Resumed rows are 0.5
        # neutral; live self-play rows are 0 (loser) or 1 (winner).
        outcome = build_outcome_batch(outcome_tags, chosen,
                                      args.batch, args.ctx, device,
                                      starts=starts)
        move_mask = _move_byte_mask(y, starts)       # (B,T-1) bool: positional, where y is a column decision
        # The trained weight at each position:
        #   resume rows -> 0.5 neutral (combined with move_mask)
        #   loser rows  -> 0 (move_mask already True; weight = 0 -> no grad)
        #   winner rows -> 1.0
        # Off-move positions stay at 0 (move_mask False anyway).
        weight = (outcome * move_mask.float()).clamp(min=0.0)
        nmove = weight.sum().clamp(min=1.0)

        # Move positions get the MASKED 7-WAY loss; every other position
        # keeps the plain 256-way byte loss. Rationale: at a move byte only 7
        # of the 256 outcomes are legal, so the old loss spent 249/256 of its
        # gradient teaching the byte alphabet instead of the game. The mask
        # comes from the v2 header's own height bits -- it is NOT a constant
        # 7-way softmax, because a full column must stay illegal or the model
        # learns to play moves the server has to discard at serve time.
        use_7way = (args.move_loss == "7way" and args.transcript == "v2")
        keep = move_col = legal_flat = dist = None
        # `col_prior` was computed once at corpus-load time (it depends
        # only on the sidecar, not on the batch). Do NOT rebind it here --
        # a local assignment would shadow it with None.
        if use_7way:
            keep, move_col = build_move_targets(y, move_mask)
            keep = keep.reshape(-1)
            move_col = move_col.reshape(-1)
            hdr_index = _header_index_from_starts(starts, y.shape[1],
                                                  device=device)
            legal3 = header_legality_from_x(x, move_mask, hdr_index)
            # A window can start mid-ply, leaving the first ply's header
            # OUTSIDE the window. Legality is unreadable there, so drop those
            # positions from the masked loss instead of guessing "all 7 legal"
            # (which would train the model to pick a full column).
            visible = header_visibility(starts, y.shape[1], device=device)
            keep = keep & visible.reshape(-1)
            legal_flat = legal3.reshape(-1, 7)

            # Imported rows carry a soft target over the equally-optimal
            # columns, but ONLY on their final ply -- the position the
            # dataset actually labelled. Every earlier ply of the recovered
            # history has no Pons label, so it keeps the one-hot byte
            # target. Applying the sidecar everywhere would teach the final
            # position's answer onto every earlier ply of its history.
        if soft_bank is not None and use_7way:
            # row_soft is (B, W): sidecar row id per TOKEN, -1 where the
            # window does not end on the labelled ply. `keep` is flat
            # (B*W,). The previous code did `keep & row_soft`, mixing a
            # per-token bool with a per-token long grid whose shape had not
            # been flattened -> "size of tensor a (49152) must match tensor
            # b (191)". Flatten both, then scatter per valid token.
            row_soft = _soft_rows_for_batch(
                examples, chosen, starts, soft_bank, y.shape[1] + 1, PLY_LEN)
            dist = None
            if row_soft is not None and keep is not None:
                if row_soft.numel() != keep.numel():
                    raise SystemExit(
                        f"soft-target grid {tuple(row_soft.shape)} does not "
                        f"match the loss mask {tuple(keep.shape)}; refusing "
                        f"to train on misaligned targets.")
                flat_rows = row_soft.reshape(-1).to(device)
                # An imported row carries a real target ONLY on its final
                # ply; every earlier ply of the recovered history has no
                # Pons label. Those plies used to fall back to a one-hot
                # target on the dataset's argmax move -- i.e. a fabricated
                # label for a position nothing labelled. Measured on a
                # 400-row batch: 7,600 supervised tokens but only 400
                # (5.3%) had a real target, so 94.7% of the move-loss
                # gradient was noise. V19 collapsed onto col5 (90.8% of
                # all picks) for exactly this reason, and the collapse was
                # invisible because a holdout average reads as a smooth
                # curve whether the model learned or froze.
                #
                # Fix: when a sidecar is loaded, SUPPRESS unlabeled plies
                # entirely instead of inventing a target for them. Only
                # plies the dataset actually labelled contribute gradient,
                # so real supervision goes from 5.3% of the batch to ~100%.
                hit = keep.reshape(-1) & (flat_rows >= 0)
                keep = hit
                if bool(hit.any()):
                    dist = torch.zeros(keep.numel(), 7, device=device)
                    ridx = flat_rows[hit].cpu()   # soft_bank stays on CPU
                    dist[hit] = (
                        args.soft_w * soft_bank[ridx].to(device)
                        + (1.0 - args.soft_w)
                        * F.one_hot(move_col.reshape(-1)[hit].clamp(min=0),
                                    7).float())
            else:
                dist = torch.zeros(keep.numel(), 7, device=device)

        elif perply is not None and use_7way:
            # PER-PLY path. The single-label sidecar supervises exactly ONE
            # move per 193-byte window (the transcript's final ply) -- 16
            # real labels per batch of 16, i.e. 6.9% of the move positions
            # that COULD be supervised. The per-ply sidecar carries a target
            # for every ply of the transcript, so the SAME window now yields
            # up to 19 supervised decisions: ~16x more real gradient per
            # byte of corpus, and no extra corpus to gather.
            _pt, _po, _np_ = perply
            slot = _perply_rows_for_batch(
                examples, chosen, starts, _np_, _po, y.shape[1] + 1, PLY_LEN)
            dist = None
            if slot is not None and keep is not None:
                if slot.numel() != keep.numel():
                    raise SystemExit(
                        f"per-ply grid {tuple(slot.shape)} does not match the "
                        f"loss mask {tuple(keep.shape)}; refusing to train on "
                        f"misaligned targets.")
                flat_slot = slot.reshape(-1).to(device)
                # A per-ply slot exists for every labelled ply of the
                # transcript, so nearly every move position in the window
                # now has a real target. Positions with no slot (plys the
                # relabeller could not solve) are DROPPED, never faked.
                hit = keep.reshape(-1) & (flat_slot >= 0)
                keep = hit
                if bool(hit.any()):
                    dist = torch.zeros(keep.numel(), 7, device=device)
                    # The sidecar row for a position is the CORPUS ROW it was
                    # sampled from -- `chosen[k]` -- repeated across that
                    # row's W positions. It is NOT the position inside the
                    # batch.
                    #
                    # This used to be
                    #     b_idx = arange(keep.numel()) // (y.shape[1] + 1)
                    # which yields 0..B-1: the BATCH INDEX. The trainer then
                    # read the teacher answers of sidecar rows 0..15 of the
                    # 320,000-row corpus, for every step of every run. The
                    # slot index came from the right row (via `chosen`), so
                    # the mismatch was invisible: the target was a real,
                    # well-formed label -- of a DIFFERENT GAME.
                    #
                    # Measured over 13,041 supervised positions on the
                    # production corpus + sidecar:
                    #   trainer target == correct target :  9.59 %
                    #   distinct sidecar rows read        :  16 of 320,000
                    #   all-zero target (=> the loss
                    #   silently falls back to a one-hot
                    #   on the PLAYED move)              : 41.27 %
                    #   with the correct row index        :  0.00 %
                    # That is why fd_a learned the PLAYED move (59 % LEGAL)
                    # while agreeing with the teacher labels on only ~20 %.
                    #
                    # The single-label path already does this correctly via
                    # `_soft_rows_for_batch`, which returns `chosen[k]`.
                    corpus_row = _perply_corpus_rows(chosen, y.shape[1], device)
                    rows = corpus_row[hit].cpu()
                    s_idx = flat_slot[hit].cpu()
                    tgt = _pt[rows, s_idx].to(device)
                    # A target with no mass on the legal columns would make
                    # _weighted_move_loss fall back to a one-hot on the
                    # PLAYED move -- a fabricated label for a position the
                    # teacher never answered. Drop those positions instead.
                    # (With the row index fixed this is 0 %, but the guard
                    # keeps a future sidecar from silently reintroducing it.)
                    _leg = legal_flat[hit]
                    _mass = (tgt.to(device) * _leg.float()).sum(dim=1)
                    _alive = _mass > 1e-6
                    if not bool(_alive.all()):
                        n_drop = int((~_alive).sum())
                        keep = keep.clone()
                        keep[torch.nonzero(hit).squeeze(1)[~_alive]] = False
                        hit = keep.reshape(-1) & (flat_slot >= 0)
                        dist = torch.zeros(keep.numel(), 7, device=device)
                        rows = corpus_row[hit].cpu()
                        s_idx = flat_slot[hit].cpu()
                        tgt = _pt[rows, s_idx].to(device)
                        print(f"[perply] dropped {n_drop} target(s) with no mass "
                              f"on the legal columns (was: silent played-move "
                              f"fallback)", flush=True)
                    dist[hit] = (
                        args.soft_w * tgt
                        + (1.0 - args.soft_w)
                        * F.one_hot(move_col.reshape(-1)[hit].clamp(min=0),
                                    7).float())
            else:
                dist = torch.zeros(keep.numel(), 7, device=device) if keep is not None else None

        model.set_ste(False)
        logits_soft = model(x)
        flat_soft = logits_soft.reshape(-1, 256)
        w_flat = weight.reshape(-1)
        y_flat = y.reshape(-1)
        if use_7way:
            # weighted masked 7-way over moves
            ce_move = _weighted_move_loss(flat_soft, keep, move_col, legal_flat,
                                         w_flat, dist=dist,
                                         col_prior=col_prior)
            # 256-way for everything that is NOT a move byte
            other = (~move_mask.reshape(-1)).float()
            n_other = (other * 1.0).sum().clamp(min=1.0)
            lp_other = F.log_softmax(flat_soft, dim=-1)
            nll_o = -lp_other.gather(1, y_flat.unsqueeze(1)).squeeze(1)
            ce_other = (nll_o * other).sum() / n_other
            ce_soft = args.move_w * ce_move + args.other_w * ce_other
        else:
            lp_soft = F.log_softmax(flat_soft, dim=-1)
            nll_soft = -lp_soft.gather(1, y_flat.unsqueeze(1)).squeeze(1)
            ce_soft = (nll_soft * w_flat).sum() / nmove

        hard_ce = torch.tensor(0.0, device=device)
        hard_w = 0.0
        if step >= args.hard_start and args.hard_w > 0:
            hard_w = args.hard_w * min(1.0, (step - args.hard_start + 1) /
                                       max(1, args.hard_ramp))
            model.set_ste(True)
            logits_hard = model(x)
            flat_hard = logits_hard.reshape(-1, 256)
            if use_7way:
                hard_ce = _weighted_move_loss(flat_hard, keep, move_col,
                                              legal_flat, w_flat, dist=dist)
            else:
                lp_hard = F.log_softmax(flat_hard, dim=-1)
                nll_hard = -lp_hard.gather(1, y_flat.unsqueeze(1)).squeeze(1)
                hard_ce = (nll_hard * w_flat).sum() / nmove
            model.set_ste(False)

        # ------------------------------------------------------------------
        # MIRROR-EQUIVARIANCE PENALTY
        # ------------------------------------------------------------------
        #
        # Measured on the authoritative 1500-position holdout (hard export):
        #     mirror-consistent     941 / 1500  (62.73%)
        #     mirror-INCONSISTENT   559 / 1500  (37.27%)
        #
        # The mirrored corpus helps (29.53% -> 62.73%) but a symmetric
        # DATASET is not a CONSTRAINT on the solution. Nothing in the
        # cross-entropy says "your answer must reflect". The centre-column
        # shortcut survives because "always c3" is mirror-consistent --
        # it is a degenerate but symmetry-PRESERVING solution, so a model
        # that has not understood the position can adopt it and still look
        # consistent from both sides.
        #
        # This term puts the constraint in the objective: evaluate the
        # model at the position AND at its reflection, and penalise the
        # Jeffreys divergence between the mirrored prediction and the
        # reflection of the original prediction. Zero iff the model is
        # exactly equivariant.
        #
        # The mirrored stream is built with the CANONICAL mirror helpers
        # (src/c4/symmetry.py) and asserted to reproduce mirror_board, so
        # a hand-rolled mirror cannot silently reintroduce the bug that
        # produced a false "the model always plays c0" result earlier.
        sym_loss = torch.tensor(0.0, device=device)
        sym_hard = None
        if args.sym_w > 0.0 and args.sym_positions > 0 and use_7way:
            sym_loss, sym_hard = _symmetry_penalty(
                model, chosen, examples, keep, args.ctx, device, version,
                args.sym_positions, args.sym_hard)
            if sym_hard is not None:
                last_sym_hard = sym_hard
            # --- GUARD: the penalty must stay BOUNDED --------------------
            # A Jeffreys divergence between two distributions over the same
            # legal support is bounded by 2*log(7) ~ 3.9.  A value far
            # above that means the reflected target was built from an
            # already-masked distribution, so the KL term degenerates to
            # p*1e9 and the "regulariser" is a constant pull, not a
            # regulariser.  This shipped once at 4.25e7 and the run
            # trained on a garbage gradient without ever erroring.
            sv = float(sym_loss.detach().item())
            if not math.isfinite(sv) or sv > SYM_PENALTY_MAX:
                raise RuntimeError(
                    f"mirror-equivariance penalty out of range: {sv:.6e} "
                    f"(must be finite and <= {SYM_PENALTY_MAX}). "
                    f"Refusing to train: the reflected target is probably "
                    f"built from a masked log-softmax.")
            # Log it every step: a regulariser whose value is never printed
            # is a regulariser whose failure is never noticed.
            if step % 50 == 0 or step == 1:
                print(f"[sym] step {step:5d} penalty={sv:.6e} "
                      f"hard={sym_hard if sym_hard is None else f'{sym_hard:.6e}'} "
                      f"w={args.sym_w}", flush=True)

        ent, perm_pen = model.regularisers()
        lam = args.ent_max * min(1.0, (step + 1) / max(1, args.ent_warmup))
        loss = (ce_soft + hard_w * hard_ce + lam * ent
                + args.perm_w * perm_pen + args.sym_w * sym_loss)

        # Cosine LR decay, keyed on the ABSOLUTE step so a resumed run
        # stays on the same curve it would have followed without the
        # restart.  Set before the step so the gradient is taken at the
        # LR that is actually used.
        if _lr_sched is not None:
            _lr_now = _lr_sched(step)
            for _pg in opt.param_groups:
                _pg["lr"] = _lr_now

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        # EMA update AFTER the optimiser step, so the shadow tracks the
        # weights that were just produced.
        if _ema_decay > 0:
            # Warm up the decay so the EMA does not lag so far behind that
            # it never catches up in a short run: start from a low decay
            # (fast tracking) and ramp to the configured value.
            _d = min(_ema_decay,
                     _ema_decay * (step + 1) / max(1, _ema_warmup))
            _ema_update(_d)

    # ---- final save ----
    # Two artifacts, because the two inference paths need different things:
    #   hard_model_final.pt  -> export_hard() integer tables (the trained
    #                           hardened artifact; usable with --hard)
    #   final_float.pt       -> plain state_dict, so play.py can also run the
    #                           soft float path for comparison
    # The transcript stamp is NOT optional: `_load_agent` reads it to pick
    # the byte alphabet, and a checkpoint without it is treated as v1 BY
    # DEFINITION. A v2 model served as v1 reads a different alphabet -- no
    # crash, just a model that plays badly. `best_hard.pt` already carries
    # it; the final export must too, or the "final" artifact is the one
    # model that cannot be served correctly.
    atomic_torch_save(export_hard_with_meta(model.cpu(), {"step": step,
                                              "seed": args.seed,
                                              "transcript": version,
                                              "d": d, "layers": layers,
                                              "fan_in": fan_in,
                                              "lags": list(model.lags),
                                              "coef_set": list(getattr(model, "coef_set", [])),
                                              "table_bias": float(args.table_bias),
                                              "ladder_opp": LADDER_PRIMARY_OPP,
                                              "ladder_greedy": True,
                                              "ladder_protocol": LADDER_PROTOCOL_VERSION}),
               run_dir / "hard_model_final.pt")
    print(f"saved -> {run_dir / 'hard_model_final.pt'}", flush=True)
    atomic_torch_save({"state": model.state_dict(),
                "meta": {"step": args.steps, "d": 32, "layers": 3,
                         "fan_in": 3, "seed": args.seed,
                         "transcript": version,
                         "lags": list(model.lags),
                         "table_bias": float(args.table_bias),
                         "ladder_opp": LADDER_PRIMARY_OPP,
                         "ladder_greedy": True,
                         "ladder_protocol": LADDER_PROTOCOL_VERSION},
                **capture_train_state(opt, rng, nprng)},
               run_dir / "final_float.pt")
    print(f"saved -> {run_dir / 'final_float.pt'}", flush=True)
    print(f"corpus: {corpus_size} decided games, "
          f"{corpus_path.stat().st_size} bytes", flush=True)
    print(f"elapsed: {time.time() - t0:.0f}s", flush=True)

    # The 4-tuple is (ladder_net, holdout_optimal_pct, -hard_bpb, distinct).
    # The old keys below read the wrong slots -- `best_sel[2]` is
    # -hard_bpb, not a "vs_ref score", and `best_sel[0]` is the ladder net,
    # not a hard accuracy. Names now match the slots.
    (run_dir / "final.json").write_text(json.dumps({
        "n_params": n_par,
        "corpus_games": corpus_size,
        "corpus_bytes": corpus_path.stat().st_size,
        "best_ladder_net": best_sel[0],
        "best_ladder_opp": LADDER_PRIMARY_OPP,
        "best_holdout_optimal_pct": best_sel[1],
        "best_hard_bpb": -best_sel[2],
        "best_distinct_games": best_sel[3],
        "transcript": version,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/c4/selfplay")
    ap.add_argument("--arch", choices=["v6", "v7"], default="v6",
                    help="v6 = gf256_markov_lm_v6 (d32/L3/lags<=64); "
                         "v7 = gf256_markov_lm (d64/L6/lags<=256, n_coef=16)")
    ap.add_argument("--lags", default="",
                    help="override the architecture's lag set, e.g. "
                         "'0,1,2,3,4,5,6,7,8,9,10,20'. The 9-byte V2 header "
                         "sits at stream offsets -8..-1 from the read "
                         "position, so lags 1..8 make every column byte "
                         "ONE-hop visible and the model's view of the board "
                         "becomes left/right symmetric. Does NOT change ctx. "
                         "Empty = architecture default.")
    ap.add_argument("--transcript", default=fmt.DEFAULT_VERSION,
                    choices=["v1", "v2"],
                    help="byte encoding of the training stream. v2 encodes "
                         "per-cell ownership in the column header (v1 stored "
                         "heights only and lost which player owned what -- "
                         "tg_hdrproof.py measured 1662 colliding headers per "
                         "30k positions). The format is stamped into every "
                         "checkpoint so the loader can never feed a model the "
                         "wrong bytes.")
    ap.add_argument("--holdout-corpus", default="",
                    help="held-out V2 sequences; the honest metric is the "
                         "share of these where the model's argmax is an "
                         "optimal move (bpb is retired for this work)")
    ap.add_argument("--holdout-targets", default="",
                    help="sidecar targets aligned to --holdout-corpus")
    ap.add_argument("--soft-w", type=float, default=0.0,
                    help="weight on the corpus soft target (0 = ignore the "
                         "sidecar and train one-hot on the move byte, the "
                         "old behaviour; 1.0 = pure soft target)")
    ap.add_argument("--soft-targets", default="",
                    help="path to *_targets.npz from tools/import_hf_c4.py; "
                         "enables soft 7-way targets over tied optimal moves")
    ap.add_argument("--perply-targets", dest="perply_targets", default="",
                    help="path to a PER-PLY sidecar (target (N,P,7), ply_of "
                         "(N,P), n_plies (N,)) as written by "
                         "tools/relabel_perply.py. When set, every labelled "
                         "ply inside the window contributes a move target "
                         "instead of only the transcript's final ply -- ~16x "
                         "more real supervision per batch from the same "
                         "corpus bytes. Mutually exclusive with "
                         "--soft-targets.")
    ap.add_argument("--move-loss", dest="move_loss", choices=["byte", "7way"],
                    default="7way",
                    help="7way = masked legal-column softmax at move bytes "
                         "(v2 only); byte = the old 256-way byte loss.")
    ap.add_argument("--move-w", type=float, default=1.0,
                    help="weight of the 7-way move loss in the total.")
    ap.add_argument("--other-w", type=float, default=0.1,
                    help="weight of the 256-way loss on non-move bytes.")
    # Seed 32 is the paper's canonical value (paper.md section 7.2) and the
    # seed of the run that produced Hard BPC 3.071. It is NOT cosmetic: the
    # seed fixes the wiring (which channel each path reads, which lag) and the
    # initialisation of the S-box permutation and readouts. paper.md section
    # 8.3 states plainly that "different random wirings create fundamentally
    # different optimization landscapes" -- a bad wiring is a genuinely
    # different model, not a different random draw of the same one.
    # Every V7-V17 C4 run silently used the old default of 0 instead, because
    # none of the run scripts passed --seed at all.
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # --- locked V6 recipe ---
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--fan-in", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    # The hard (one-hot / STE) forward is what `export_hard()` SHIPS. If it
    # is not trained from the start, the shipping artifact for the first
    # `--hard-start` steps is an UNTRAINED approximation of the soft
    # function, and it is provably board-blind: measured over V23's
    # checkpoints, changing the board moves the exported logits by exactly
    # 0.0000 up to step 11500, and only from step 12000 (the old default)
    # does it read the board at all. The default is therefore 0 -- train the
    # path you ship, from step 0.
    ap.add_argument("--hard-start", type=int, default=0)
    ap.add_argument("--hard-ramp", type=int, default=4000)
    ap.add_argument("--hard-w", type=float, default=0.5)
    ap.add_argument("--ent-max", type=float, default=2.0)
    ap.add_argument("--perm-w", type=float, default=1.0)
    # --- mirror-equivariance regulariser ----------------------------------
    # The authoritative 1500-position holdout shows 37.27% of positions
    # get a DIFFERENT answer when the board is merely reflected.  A
    # mirrored corpus helps (29.53% -> 62.73% consistency) but a symmetric
    # DATASET is not a CONSTRAINT on the solution.  This penalty puts the
    # constraint in the objective: the Jeffreys divergence between the
    # model's prediction on the reflected board and the reflection of its
    # prediction on the original board.  Zero iff exactly equivariant.
    #
    # The centre-column shortcut survives every data-side fix partly
    # because "always c3" is mirror-CONSISTENT: it is a degenerate but
    # symmetry-preserving solution.
    #
    # sym-w = 0.0 (the default) keeps historical behaviour EXACTLY as it
    # was, so every existing run remains reproducible.
    ap.add_argument("--sym-w", type=float, default=0.0)
    ap.add_argument("--sym-positions", type=int, default=8)
    ap.add_argument("--sym-hard", action="store_true",
                    help="also report the penalty on the hard export path")
    # --- table init bias (THE board-blindness fix) --------------------------
    # The exported artifact is the ARGMAX of emb_logits / sbox_logits /
    # coef_logits. At init those carry a large positive bonus toward the
    # identity/one-hot solution (emb +10/+8, sbox +8, coef +6/+4), which
    # creates a top1-top2 margin of ~7.7 logits.
    #
    # Adam moves a parameter by at most `lr` per step, so N steps can move
    # it by at most lr*N -- 1.8 logits at lr=0.0006 over 3,000 steps, and
    # 7.2 over 12,000. The margin that must be crossed is 7.33. So on any
    # realistic budget the argmax CANNOT flip, the exported integer table
    # stays frozen, and the hard export is board-blind NO MATTER how the
    # STE schedule is set. (Measured: emb/sbox/coef table-hash identical at
    # steps 0..2500; grad-norm on the table logits is exactly 0.0 because
    # the zero-initialised readout multiplies their output by zero.)
    #
    # The entropy regulariser makes it worse: minimising entropy pushes
    # the identity logit UP, so the margin GROWS during training (sbox
    # 7.72 -> 9.94 between step 0 and step 3000).
    #
    # --table-bias 0.0 removes the bonus so the argmax is reachable and the
    # identity solution must be LEARNED instead of handed over. The RNG
    # draw order is unchanged, so coef_set is byte-identical and existing
    # checkpoints remain loadable.
    #
    # DEFAULT IS NOW 0.0. The controlled A/B (runs/_probe/tb10 vs tb00,
    # 3000 steps, identical in every other respect) showed that under
    # 1.0 the exported tables are FROZEN (argmax identical at every
    # checkpoint, table gradient exactly zero, board sensitivity 0.0000
    # until step 3000), while under 0.0 they learn at every checkpoint
    # and the hard export reads the board from step 500 (sensitivity
    # 0.5444 -> 1.6794). The soft_loader defaults OLD checkpoints to 1.0
    # when they carry no `table_bias` in their metadata, so every existing
    # artifact still reconstructs exactly as it was trained.
    ap.add_argument("--col-prior", dest="col_prior", type=float, default=0.0,
                    help="Logit-adjustment strength tau: subtract "
                         "tau*log p_teacher(c) from column c's logit during "
                         "the 7-way move loss. 0 = historical behaviour. "
                         "1 = fully removes the teacher's marginal column "
                         "prior, so the model must learn the board to "
                         "reduce the loss. Measured need: the hard export's "
                         "column offset gap (2.58) dwarfs its board signal "
                         "(0.47), which collapses the policy onto c3.")
    ap.add_argument("--table-bias", type=float, default=0.0,
                    help="scale of the identity/one-hot init bonus on the "
                         "table logits; 0.0 = unbiased (tables can learn, "
                         "DEFAULT); 1.0 = historical (frozen tables)")
    # --- optimisation ---
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lr-drop", type=float, default=0.3)
    # --- LR schedule (Claude's review, confirmed gap) ----------------------
    # A constant LR lets Adam oscillate around the optimum late in a long
    # run.  Cosine decay removes that without touching the objective.
    # "none" is the DEFAULT so every historical run reproduces exactly.
    ap.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="'cosine' decays the LR from --lr to half of it over "
                         "the run (absolute step, so a resume stays on the "
                         "same curve); 'none' keeps the historical constant LR")
    # --- EMA weights (Claude's review, confirmed gap) ---------------------
    # The EMA copy is flatter around the optimum than the raw weights and is
    # the copy that should be hardened and shipped when --ema-export is set.
    # 0.0 disables it (default) so historical runs are unaffected.
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA decay for the shadow weights; 0.0 = disabled")
    ap.add_argument("--ema-warmup", type=int, default=1000,
                    help="steps before the EMA is allowed to be exported")
    ap.add_argument("--ema-export", action="store_true",
                    help="harden and promote the EMA weights instead of the "
                         "raw weights")
    # --- checkpoint retention ---------------------------------------------
    # A full float checkpoint (weights + Adam moments) is ~203 MB and a hard
    # export is ~34 MB.  Keeping every snapshot of a 100k-step run would
    # need ~10 GB, which does not fit on a nearly-full disk.  The trainer
    # keeps the most recent --keep-last float checkpoints (the ones a
    # resume could use) and prunes the older ones, keeping their HARD
    # exports (34 MB) as the historical artifact.
    ap.add_argument("--keep-last", type=int, default=3,
                    help="how many recent FLOAT checkpoints to keep "
                         "(default 3, enough for a clean resume)")
    ap.add_argument("--keep-hard-all", action="store_true",
                    help="keep every hard export; otherwise hard exports "
                         "older than --keep-last are pruned too")
    # --- abort rules --------------------------------------------------------
    # An abort on an 18-hour run is expensive, so the rules arm only after a
    # warm-up and need several consecutive flat evaluations.
    ap.add_argument("--abort-warmup", type=int, default=20000,
                    help="no abort rule can fire before this step "
                         "(default 20000 = 1 epoch)")
    ap.add_argument("--abort-flat-evals", type=int, default=4,
                    help="consecutive flat holdout evaluations that trip the "
                         "flat rule (default 4)")
    ap.add_argument("--abort-flat-below", type=float, default=55.0,
                    help="the flat rule only fires while the best holdout "
                         "agreement is below this (default 55%)")
    ap.add_argument("--abort-on", choices=["warn", "abort"], default="warn",
                    help="what to do when an abort rule fires: 'warn' "
                         "records the finding and continues (default), "
                         "'abort' stops the run. Aborting an 18-hour run on "
                         "a false positive is expensive, so it is opt-in.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ent-warmup", type=int, default=2000)
    # --- self-play ---
    ap.add_argument("--seed-source", choices=["teacher", "random"],
                    default="teacher",
                    help="teacher = seed the corpus with solver games "
                         "(competent demonstrations); random = the old "
                         "random-vs-random winner games, which teach nothing.")
    ap.add_argument("--seed-games", type=int, default=2000)
    ap.add_argument("--selfplay-games", type=int, default=32)
    ap.add_argument("--selfplay-every", type=int, default=20)
    ap.add_argument("--temp-start", type=float, default=1.2)
    # --- V11: teacher data source ---
    ap.add_argument("--corpus-source", choices=["teacher", "selfplay"],
                    default="teacher",
                    help="'teacher' (default): grow the corpus with negamax/"
                         "lookahead/heuristic teacher games. 'selfplay': the old "
                         "V9/V10 behaviour, model plays itself -- kept only for "
                         "A/B comparison.")
    ap.add_argument("--teacher-games", type=int, default=48,
                    help="teacher games per growth cycle (replaces --selfplay-games "
                         "when --corpus-source teacher)")
    # --- V11.2: corpus quality gate ---
    ap.add_argument("--keep-low-quality", action="store_true",
                    help="keep grader-BLUNDER rows (games the winner won while "
                         "dropping a winning move). Off by default: outcome "
                         "weighting would happily teach those bad moves. The "
                         "corpus file is never modified either way -- exclusion "
                         "happens when training examples are built.")
    ap.add_argument("--gate-report", default="",
                    help="write the quality-gate audit (rows in/out, grades, "
                         "excluded counts) to this JSON path")
    # --- V12: teacher-dominant sampling mix ---
    ap.add_argument("--max-legacy-frac", type=float, default=0.20,
                    help="cap the share of FILTERED LEGACY rows in the live "
                         "training buffer. Teacher rows are appended every "
                         "growth cycle, but the gated legacy block is resident "
                         "in `examples` for the entire run, so without a cap "
                         "legacy stays the majority of every batch forever. "
                         "Measured on V11.1: 2,037 legacy vs 288 teacher = "
                         "12.4%% teacher / 87.6%% legacy, which is why hard-acc "
                         "climbed while the model kept losing ~48/48 to the "
                         "heuristic. Default 0.20 gives the intended ~80%% "
                         "teacher mix. corpus.txt is NOT modified -- this only "
                         "governs which rows are SAMPLED into each batch.")
    # --- V12: behaviour gate ---
    ap.add_argument("--behavior-ref", choices=["heuristic", "lookahead"],
                    default="lookahead",
                    help="reference policy a row's winner must agree with to "
                         "count as a demonstration. 'lookahead' (2-ply) is the "
                         "default: unbeaten by heuristic at ~150x solver speed. "
                         "The negamax solver is deliberately NOT offered -- at "
                         "~0.3 s/game a full sweep would take minutes and it is "
                         "not dominant over the heuristic (W10 L10).")
    ap.add_argument("--min-agreement", type=float, default=0.60,
                    help="fraction of the WINNER's plies that must match the "
                         "reference policy's top-1 for a row to be kept. "
                         "Measured on this corpus: teacher games score 0.74, "
                         "random games 0.19, legacy rows average 0.41 -- so "
                         "0.60 keeps the well-played games and rejects the "
                         "random-grade majority.")
    ap.add_argument("--corpus-mix", choices=["random", "mixed"],
                    default="random",
                    help="'random' = self-play vs self (current). "
                         "'mixed' = model vs random/heuristic/lookahead/minimax "
                         "with weights 0.15/0.30/0.30/0.25. Mixed produces more "
                         "decisive game trajectories that the model can learn "
                         "strong play from.")
    ap.add_argument("--temp-end", type=float, default=0.4)
    # Evaluation plays at this temperature. Kept separate from the
    # sampling schedule above on purpose: sampling temperature is a data
    # -generation knob, eval temperature is a measurement knob, and
    # conflating them meant we scored the model on its stumbles.
    #
    # DEFAULT CHANGED 0.2 -> 0.0 (2026-10-23). 0.0 means GREEDY: the
    # argmax over the legal columns, which is exactly what the UI serves
    # and exactly what `Agent(greedy=True)` does. At 0.2 the ladder graded
    # a STOCHASTIC policy, so every "vs_opp" number in every log from V15
    # onward was a measurement of the sampler rather than of the shipped
    # agent. Do not raise this above 1e-3 without saying so in the log:
    # the strength numbers stop being comparable to the shipping policy.
    # The opponent is explored separately (per-ply epsilon in
    # `_eval_opponents`), which is where the variation belongs.
    ap.add_argument("--eval-temp", type=float, default=0.0)
    # --- eval / logging ---
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000,
                    help="steps between full opponent-ladder evals + snapshots")
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--resume", default=None,
                    help="path to a best_hard.pt export to resume weights+step from")
    ap.add_argument("--load-corpus", action="store_true",
                    help="train on an existing corpus.txt in --run-dir "
                         "instead of seeding one (for imported datasets)")
    ap.add_argument("--no-ckpt", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())