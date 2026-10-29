"""Connect-Four GF(256) Markov-LM agent.

Public API:
  * `Agent` -- wraps a V6 `GFMarkovLM`, exposes choose/choose_batch.
  * `moves_to_tokens` -- make one training row from a game.
  * `start_of_game_tokens` / `prompt_tokens` -- the model prefix used by
    choose() and the chat loop.
  * `hard_to_cpu` -- move every tensor in `export_hard()` to CPU. Required
    because `model.export_hard()` returns buffers still on the source
    device, and `EXP` / `LOG` tables in `gf256_markov_lm_v6.gfcpu` are CPU.
  * `agent_agreement` -- fraction of positions where two copies of the same
    agent pick the same column (sanity check on determinism / sampling).
"""

from __future__ import annotations

import random
from typing import List, Sequence

import torch

from . import COLS, P1, P2
from .board import Board, is_terminal
from . import fmt

# V2 is the format this tree speaks by default; V1 stays available so an old
# checkpoint keeps decoding. The Agent is constructed with the version read
# from the checkpoint meta, so a model is never fed the wrong format.
_transcript = fmt.get()
PLY_LEN = _transcript.PLY_LEN
# COL_BYTE_OFFSET stays the RANGE the model must output for a move. It is
# deliberately 0x41..0x47 in BOTH formats: that range is disjoint from the
# header bytes (v1 heights 0x30..0x36, v2 columns 0x81..0xFF), which is what
# makes COL_SLICE safe to argmax over.
COL_BYTE_OFFSET = _transcript.COL_BYTE_OFFSET
N_COL_BYTES = _transcript.N_COL_BYTES
COL_SLICE = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + N_COL_BYTES)

# Padding for short prompts. 0x00 is benign -- no live board state uses it.
PAD_BYTE = 0x00
# The padding byte the TRAINER actually uses. build_batch() does
# `window + b"\n" * (L - len(window))`, i.e. RIGHT-padding with byte 10.
# Inference must use the same byte on the same side or the model reads a
# token sequence it was never trained on.
RIGHT_PAD_BYTE = 0x0A
# Hardened GFCipher needs a fixed prompt length; the model is built with a
# 512-byte context window so left-padding is fine. The floor must be >= the
# model's max lag, otherwise hard_forward's lagged slice src[:, :-l] returns a
# 0-width tensor and the cat pads to l, which no longer matches T.
HARD_MIN_LEN = 96


def hard_min_len_for(lags) -> int:
    """Pad floor for a hardened model with the given lag set.

    V6's max lag is 64 (<= the legacy 96 floor), but V7 reaches 256, so the
    floor has to track the model or hard_forward raises a shape error.
    """
    return max(HARD_MIN_LEN, (max(int(l) for l in lags) + 1) if len(lags) else 0)


def moves_to_tokens(moves: Sequence[int],
                    version: str | None = None) -> torch.Tensor:
    """One game's plies as a 1D int64 tensor of byte ids (skip the newlines)."""
    mod = fmt.get(version)
    raw = mod.encode_game(moves)
    return torch.tensor(list(raw.replace(b"\n", b"")), dtype=torch.long)


def start_of_game_tokens(version: str | None = None) -> torch.Tensor:
    """The opening ply's header with last=NO_LAST, turn=P1."""
    b = Board()
    h = fmt.get(version).encode_state(b, last=None, side_to_move=P1)
    return torch.tensor(list(h), dtype=torch.long)


def prompt_tokens(moves: Sequence[int],
                  version: str | None = None) -> torch.Tensor:
    """The full prompt right before the move byte for the next ply.

    The model owns one (T,256) forward per byte position; we feed it the
    header bytes of the NEXT ply, ending on its TURN byte. That makes the
    logits at position T-1 predict the next byte the model needs to write,
    which IS that ply's move byte.

    The next ply's header is NOT in the stream of already-played plies -- it
    must be encoded from the board. Truncating the played stream instead
    (the old `flat[:len(flat)-2]`) lands the read position on the previous
    ply's TURN byte, whose target the trainer masks out of the loss, so the
    agent reads column logits from a never-trained slot.

    Trainer parity (train.py `_move_byte_mask` / `action_consistent_examples`):
    with k = len(moves), dense index k*PLY_LEN + (PLY_LEN-2) = k*10+8 must be
    the last input position, predicting target k*10+9 = the move byte.
    """
    mod = fmt.get(version)
    if not moves:
        return start_of_game_tokens(version)
    raw = mod.encode_game(moves)
    # strip the newlines; the model predicts bytes
    flat = raw.replace(b"\n", b"")
    k = len(moves)
    # `flat` holds k complete plies (k*PLY_LEN bytes). Append the next ply's
    # header, then cut so the read position lands on that header's TURN byte.
    board = board_from_moves(moves)
    hdr = list(mod.encode_state(board, last=moves[-1],
                                side_to_move=board.turn))
    pre = flat + bytes(hdr)
    # k complete plies (k*10 bytes) + next ply's 9-byte header = k*10+9 bytes,
    # i.e. last position k*10+8 = that header's TURN byte. Its next-byte
    # target is index k*10+9 = the move byte. Exactly the trainer's slot.
    assert len(pre) == k * PLY_LEN + (PLY_LEN - 1), (len(pre), k)
    return torch.tensor(list(pre), dtype=torch.long)


def board_from_moves(moves: Sequence[int]) -> Board:
    """Replay `moves` into a Board. Turns/legality come from apply_move."""
    b = Board()
    for c in moves:
        b.apply_move(c)
    return b


def hard_to_cpu(hard):
    """Recursively move every tensor inside an export_hard() payload to CPU.

    The hardened forward in `gf256_markov_lm_v6.gfcpu` runs entirely on CPU
    tables (EXP / LOG). If any tensor inside the exported dict is still on
    CUDA, `to(torch.int64)` falls back to a copy that leaves it on CUDA, and
    the GFCipher kernel crashes with a device mismatch. This helper walks
    the export and `.cpu()`s every leaf tensor.
    """
    if isinstance(hard, torch.Tensor):
        return hard.cpu()
    if isinstance(hard, dict):
        return {k: hard_to_cpu(v) for k, v in hard.items()}
    if isinstance(hard, (list, tuple)):
        return type(hard)(hard_to_cpu(v) for v in hard)
    return hard


def trainer_window(stream: Sequence[int], ctx: int):
    """Slice an unpadded byte stream exactly the way train.py's build_batch does.

    build_batch() (train.py:456-465) draws windows of length L = ctx+1 with
    `start = rng.integers(0, len(seq) - L + 1)` -- always a full window ending
    at or before the sequence end -- right-pads with b"\\n" for the rare short
    sequence, and splits
        x = window[:-1]   (ctx inputs)
        y = window[1:]    (ctx targets)
    x[j] predicts window[j+1]. So a target at absolute index m is read at
    j = m - start - 1, and 0 <= j <= L-2 = ctx-1. The single target slot that
    makes L = ctx+1 is NEVER an input -- y[-1] has no x counterpart.

    To predict the byte at absolute index n (= len(stream), the next move
    byte), x must be the ctx real bytes stream[n-ctx : n] and we read at
    ctx-1. That is exactly the trainer's last legal window: start = n - L + 1,
    i.e. seq[start:start+L] = seq[n-ctx : n+1] which is ctx real bytes plus
    the single pad standing in for the unseen target.

    Returns `(x, read_idx)`: the ctx-token input window and the position whose
    logits predict byte `n`. The trailing pad is causally invisible to
    read_idx, but read_idx itself matters because positional encodings are
    window-relative -- reading one slot early predicts the previous byte.

    Every inference path MUST use this. Reading `logits[-1]` off an unpadded
    prompt, left-padding, padding with byte 0, or an off-by-one start all give
    the model an input it never saw during training.
    """
    L = ctx + 1
    n = len(stream)
    start = max(0, n - ctx)                 # == n - L + 1
    x = list(stream[start:n])                # <= ctx real bytes
    read_idx = len(x) - 1                    # == n - start - 1
    x += [RIGHT_PAD_BYTE] * (ctx - len(x))   # trainer pads on the right
    assert len(x) == ctx and 0 <= read_idx <= ctx - 1, (n, ctx, read_idx)
    return x, read_idx


class Agent:
    """Wrap a GFMarkovLM for connect-four inference.

    Args:
        model: a fully-built V6 GFMarkovLM (already loaded via .to(device)).
        device: torch device the model lives on. Hardened path uses CPU.
        greedy: if True, argmax the column logits; else sample by softmax.
        temperature: sampling temperature (only used when not greedy).
        rng: a random.Random / torch.Generator instance.
    """

    def __init__(self, model, device, greedy: bool = True,
                 temperature: float = 1.0,
                 rng: random.Random | None = None,
                 ctx: int = 192,
                 version: str | None = None):
        self.model = model
        self.device = device
        self.greedy = greedy
        self.temperature = temperature
        self.rng = rng or random.Random(0xC4C4_C4C4)
        # Trained context length (train.py --ctx). The inference window must
        # match it exactly; do NOT infer this from model.lags, whose largest
        # entry (256) is a positional-lag spacing, not the context length.
        self.ctx = ctx
        # Transcript format this agent ENCODES in. Must match the format the
        # weights were trained on (read from the checkpoint meta by the
        # loader). Feeding a v2-trained model a v1 prompt reads pure noise
        # and fails silently, so this is never inferred from byte values.
        self.version = fmt.normalize(version)
        self.hard = None          # lazily built integer export, hard path only

    # ------------------------------------------------------------------ utils

    def legal_mask(self, board: Board) -> torch.Tensor:
        """(COLS,) bool tensor, True where the column is playable."""
        h = board.heights
        return torch.tensor([c < 6 for c in h], dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ soft

    def column_logits(self, board: Board, moves: Sequence[int]) -> torch.Tensor:
        """(COLS,) un-normalised column preferences for the side-to-move.

        Builds the trainer-exact window (right-padded with byte 10, read at
        the last real token) -- see trainer_window().
        """
        x, read_idx = trainer_window(
            prompt_tokens(moves, self.version).tolist(), self.ctx)
        with torch.no_grad():
            logits = self.model(torch.tensor([x], dtype=torch.long,
                                            device=self.device))   # (1,ctx,256)
        return logits[0, read_idx, COL_SLICE]                      # (7,)

    def batch_column_logits(self, batch_prompts: Sequence[Sequence[int]],
                            device: torch.device) -> torch.Tensor:
        """(B, COLS) column logits for a padded batch of prompts.

        Each prompt is windowed independently with trainer_window() so the
        read index is per-row, then right-padded to a common ctx length for
        the batched forward.
        """
        rows = []
        for p in batch_prompts:
            x, _ = trainer_window(list(p), self.ctx)
            rows.append(x)
        T = max(len(r) for r in rows)
        padded = [r + [RIGHT_PAD_BYTE] * (T - len(r)) for r in rows]
        t = torch.tensor(padded, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model(t)                                # (B,T,256)
        # all rows are padded to the same T; re-derive each read index
        out = []
        for b, p in enumerate(batch_prompts):
            _, read_idx = trainer_window(list(p), self.ctx)
            out.append(logits[b, read_idx, COL_SLICE])
        return torch.stack(out)                                   # (B,7)

    # ------------------------------------------------------------------ hard

    def hard_column_logits(self, board: Board, moves: Sequence[int],
                           hard=None) -> torch.Tensor:
        """Same as column_logits but routes through `model.hard_forward`.

        `hard_forward(tokens, hard)` is the integer forward: argmax-ed
        embedding tables, GF table lookups, float readout. It runs on CPU,
        so the payload must already be CPU tensors (see hard_to_cpu).

        The window MUST be byte-identical to what build_batch() fed the
        trainer, otherwise the readout is meaningless:

          * padding byte is `\\n` (10) -- NOT PAD_BYTE (0);
          * padding goes on the RIGHT, never the left;
          * the read position is the LAST REAL TOKEN (index n-1 of the
            unpadded stream), because `logits[-1]` after right-padding
            reads a PAD slot the trainer never used as an input.

        `hard_min_len_for(lags)` only sizes the lag mix; feeding lag-256
        context to a ctx=192 model is out-of-distribution.
        """
        if hard is None:
            hard = self.hard
        if hard is None:
            hard = hard_to_cpu(self.model.export_hard())
            self.hard = hard
        stream = prompt_tokens(moves, self.version).tolist()   # unpadded, len = n
        x, read_idx = trainer_window(stream, self.ctx)  # right-padded, ctx long
        with torch.no_grad():
            logits = self.model.hard_forward(
                torch.tensor([x], dtype=torch.long), hard)   # (1,ctx,256)
        return logits[0, read_idx, COL_SLICE]

    # ------------------------------------------------------------------ choose

    def choose(self, board: Board, moves: Sequence[int], use_hard: bool = False) -> int:
        """Pick a column index [0..6]. Greedy or sampled. Honours full columns."""
        if use_hard:
            logits = self.hard_column_logits(board, moves)
        else:
            logits = self.column_logits(board, moves)
        mask = self.legal_mask(board).to(logits.device)
        # set illegal columns to a very negative score so they are never picked
        NEG = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        logits = torch.where(mask, logits, NEG)
        if self.greedy:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
        return idx

    def choose_batch(self, boards: Sequence[Board],
                     games: Sequence[Sequence[int]],
                     use_hard: bool = False) -> List[int]:
        """Pick columns for a list of positions independently."""
        return [self.choose(b, list(m), use_hard=use_hard)
                for b, m in zip(boards, games)]


def agent_agreement(a: Agent, b: Agent, n: int = 200) -> float:
    """Fraction of positions where `a` and `b` (same weights, different rngs)
    pick the same column. ~1.0 with greedy=True. Used to check determinism.
    """
    import collections
    wins = 0
    rng = random.Random(0xA11C_E)
    for _ in range(n):
        board = Board()
        moves = []
        while True:
            ca = a.choose(board, moves)
            cb = b.choose(board, moves)
            if ca == cb:
                wins += 1
            m = rng.choice(board.valid_actions())
            board.apply_move(m)
            moves.append(m)
            winner = board.is_win()
            if is_terminal(board):
                break
    return wins / n
