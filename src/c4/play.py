"""Play Connect Four against the trained GF(256) Lagrange Field Transformer.

You are always P1 and move first -- the harder side. The model plays P2
through the hardened integer forward (`export_hard` -> `GFCipher` tables),
i.e. the same GF(256) body it was trained with, with no float matmuls in the
nonlinear path.

Connect Four is a solved game: P1 wins against perfect play, starting in the
middle column. The model does not know that at the start. What you are
actually measuring is how far pure self-play -- winners-only corpus, no
solver, no search, no human demonstrations -- gets from nothing.

  python -m src.c4.play --ckpt runs/c4/final.pt            # soft float path
  python -m src.c4.play --ckpt runs/c4/final.pt --hard     # hardened path
  python -m src.c4.play --ckpt runs/c4/final.pt --temp 0.3
  python -m src.c4.play --ckpt runs/c4/final.pt --games 5   # auto-play, no input
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.c4.agent import Agent, hard_to_cpu      # noqa: E402
from src.c4.board import Board, COLS, P1, P2, is_terminal  # noqa: E402
from src.gf256_markov_lm_v6 import GFMarkovLM     # noqa: E402

RED = "R"
YEL = "Y"


def render(board: Board, moves: list[int], next_to_move: int) -> str:
    """Print the board, bracketing the most recently played stone."""
    out = ["".join(f"{c + 1:>3}" for c in range(COLS)),
           "+" + "+".join("-" * 2 for _ in range(COLS)) + "+"]
    last = moves[-1] if moves else None
    for r in range(6):
        row = []
        for c in range(COLS):
            cell = board.cell(r, c)
            if cell == 0:
                row.append(" .")
                continue
            mark = RED if cell == P1 else YEL
            # the newest stone is the topmost filled cell of its column
            if c == last and board.heights[c] - 1 == r:
                row.append(f"[{mark}]")
            else:
                row.append(f" {mark}")
        out.append("".join(row))
    out += ["+" + "+".join("-" * 2 for _ in range(COLS)) + "+",
            f"[R]=you  [Y]=model   to move: "
            f"{'you' if next_to_move == P1 else 'model'}"]
    return "\n".join(out)


def load_model(ckpt_path: str, device: str) -> tuple[GFMarkovLM, dict]:
    """Rebuild the V6 architecture and load the trained weights.

    Two checkpoint flavours exist after a train run:
      final_float.pt      -> {"state": state_dict, "meta": {...}}
      hard_model_final.pt -> raw export_hard() integer tables

    weights_only=False is required because the hard export is an
    OrderedDict of plain ints / lists / tensors with no classes or code
    objects in it, and weights_only=True cannot consume export_hard
    payloads. The float-statedict path is then handled normally.
    """
    model = GFMarkovLM(d=32, layers=3, fan_in=3, seed=0).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta: dict = {}
    state = None
    if isinstance(ck, dict):
        meta = ck.get("meta", {}) or {}
        state = ck.get("state")
    if state is None:
        raise SystemExit(
            f"{ckpt_path} holds only an export_hard() payload (no float\n"
            "  state), so the soft path cannot run. Point --ckpt at\n"
            "  final_float.pt, or train with the state sidecar enabled.")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, meta


def model_move(agent: Agent, board: Board, moves: list[int],
               hard=None) -> tuple[int, bool, torch.Tensor]:
    """Ask the model for a column. Returns (col, from_hard_path, logits)."""
    if hard is not None:
        logits = agent.hard_column_logits(moves, hard)
    else:
        logits = agent.column_logits(moves)
    legal = agent.legal_mask(board)
    scores = logits + legal
    col = int(scores.argmax())
    return col, hard is not None, logits


def one_game(agent: Agent, hard, temperature: float, interactive: bool,
             rng: random.Random, human_first: bool = True) -> dict:
    """Play one game. Returns a result dict.

    Human is always P1 (board.turn == 1) and moves first. The model is P2.
    Result is reported via board.is_win() (which returns the winning player
    1 or 2 directly) -- never via board.turn, which has already flipped to
    the loser by the time we look.
    """
    board = Board()
    moves: list[int] = []
    human = P1 if human_first else P2
    print("\n" + "=" * 36)
    print(f"NEW GAME  (you are {'R' if human == P1 else 'Y'})")
    print("=" * 36)

    while True:
        if is_terminal(board):
            break

        if board.turn == human:
            if interactive:
                print(render(board, moves, board.turn))
                legal = sorted(board.valid_actions())
                while True:
                    raw = input(
                        f"your move (1-{COLS}, legal "
                        f"{[c + 1 for c in legal]}): ").strip()
                    if raw.lower() in {"q", "quit", "exit"}:
                        return {"result": "quit", "plies": len(moves),
                                "moves": moves}
                    try:
                        col = int(raw) - 1
                    except ValueError:
                        print("  not a number.")
                        continue
                    if col in legal:
                        break
                    print(f"  column {col + 1} is not playable.")
            else:
                col = rng.choice(sorted(board.valid_actions()))
        else:
            # model plays via Agent.choose, which already handles
            # legal masking, temperature sampling, and use_hard on/off
            if hard is not None:
                agent.hard = hard
                col = agent.choose(board, moves, use_hard=True)
            else:
                col = agent.choose(board, moves, use_hard=False)

        board.apply_move(col)
        moves.append(col)
        if interactive and board.turn != human:
            print(render(board, moves, board.turn))
            print(f"  model played column {col + 1}")

    print()
    print(render(board, moves, 0))
    win = board.is_win()
    if win:
        # is_win() returns the winning player directly. Do NOT infer the
        # winner from board.turn -- that has already flipped to the loser.
        you_won = (win == human)
        print(f"\n  {'YOU WIN' if you_won else 'MODEL WINS'} (P{win})")
        result = "you" if you_won else "model"
    else:
        print("\n  DRAW -- board full, nobody got four")
        result = "draw"
    print(f"  {len(moves)} plies: {' '.join(str(m + 1) for m in moves)}")

    return {"result": result, "plies": len(moves), "moves": moves}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/c4/run1/final_float.pt")
    ap.add_argument("--hard", action="store_true",
                    help="use the hardened integer forward "
                         "(GF(256) tables)")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy, >0 samples the model")
    ap.add_argument("--games", type=int, default=0,
                    help="play N games automatically, no input")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"checkpoint not found: {args.ckpt}")
        print("train one first:  python -m src.c4.train "
              "--run-dir runs/c4/run1")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} on {device}")
    if meta:
        print(f"  trained steps: {meta.get('step', '?')}")

    agent = Agent(model, device,
                  greedy=args.temp <= 0.0,
                  temperature=max(args.temp, 1e-6),
                  rng=random.Random(args.seed))

    hard = None
    if args.hard:
        # Export from the live float model: identical to the on-disk
        # hard_model_final.pt, and guarantees hard and soft see the same
        # weights. hard_to_cpu() is required -- GFCipher tables live on CPU.
        hard = hard_to_cpu(model.export_hard())
        print("  path: HARDENED integer forward (GF(256) tables)")
    else:
        print("  path: soft float forward")

    rng = random.Random(args.seed + 1)
    interactive = args.games == 0
    tally = {"you": 0, "model": 0, "draw": 0}

    if interactive:
        print("\nYou are R. You move first. Columns are 1-7.")
        print("Connect Four is solved -- P1 wins with best play, "
              "starting centre.\nType a column number and hit enter. "
              "q quits.")
        while True:
            r = one_game(agent, hard, args.temp, True, rng)
            if r["result"] == "quit":
                break
            tally[r["result"]] = tally.get(r["result"], 0) + 1
            again = input("\nanother game? [y/N] ").strip().lower()
            if again not in {"y", "yes"}:
                break
    else:
        for i in range(args.games):
            r = one_game(agent, hard, args.temp, False, rng)
            tally[r["result"]] = tally.get(r["result"], 0) + 1
            print(f"game {i + 1}/{args.games}: {r['result']} "
                  f"in {r['plies']} plies")

    total = sum(tally.values())
    if total:
        print("\n" + "=" * 36)
        print(f"  you {tally['you']}   model {tally['model']}   "
              f"draw {tally['draw']}")
        print("=" * 36)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())