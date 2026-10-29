"""Tiny HTTP server that serves the connect-four UI and lets the browser
call the V6 GF(256) Markov-LM agent.

Endpoints
---------
GET  /        -> serves ui.html from the same directory as this file
GET  /reset   -> start a new game (the user is always P1, model is P2)
POST /move    -> JSON {"col": int}; applies the user's move, asks the agent
                 for a reply, returns {cells, outcome, your_move, ai_move}.

We use ``http.server.ThreadingHTTPServer`` so a long-ish model call doesn't
block other requests (the model is small and CPU-only so this rarely matters,
but the page also polls and pre-loads).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: E402  (for legal-action softmax distribution)

HERE = Path(__file__).resolve().parent

# ---- thread budget ----------------------------------------------------
# The server answers every /state with a REAL forward pass (float path
# ~93 ms + hard path ~12 ms on this box).  By default torch grabs all
# 32 cores for EACH inference, so a server thread and a client thread
# that both infer at the same time oversubscribe the machine: the
# measured /state latency jumps from ~105 ms to >20 s, and the live
# parity tests (timeout 20 s) fail with a read timeout even though the
# server answered.
#
# The server only ever needs ONE inference at a time -- the game state
# is serialised behind `_lock` -- so a small thread budget is both
# correct and much friendlier to a co-located test client.
_TORCH_THREADS = int(os.environ.get("TG_UI_TORCH_THREADS", "4"))
try:
    torch.set_num_threads(max(1, _TORCH_THREADS))
except Exception:  # pragma: no cover - torch always has this
    pass

# Make the project's `src/` importable so this script also runs as
# `python -m src.c4.ui_server` from the project root.
_ROOT = HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.c4 import COLS, ROWS, P1, P2  # noqa: E402
from src.c4 import fmt  # noqa: E402
from src.c4.agent import Agent, hard_to_cpu, prompt_tokens  # noqa: E402
from src.c4.board import Board, Outcome  # noqa: E402
from src.c4.patterns import pattern_report  # noqa: E402
from src.c4.sessions import (  # noqa: E402
    SessionStore,
    clear_session_cookie,
    new_session_id,
    parse_cookies,
    session_cookie,
    valid_session_id,
)
from src.gf256_markov_lm_v6 import GFMarkovLM as GFMarkovLM_V6  # noqa: E402
from src.gf256_markov_lm import GFMarkovLM as GFMarkovLM_V7  # noqa: E402


# Outcome names are RELATIVE TO THE HUMAN, so they depend on the seat:
# with the human as P2, a WIN_P1 is the MODEL's win. The seat-aware
# mapping lives in Game._outcome_name().
_OUTCOME_NAMES = {
    Outcome.ONGOING: "ongoing",
    Outcome.WIN_P1: "you_win",
    Outcome.WIN_P2: "ai_win",
    Outcome.DRAW: "draw",
}

# --------------------------------------------------------------- hot reload
# The whole point of a long training run is that you can play the model
# WHILE it improves. The trainer rewrites `best_hard.pt` in place (atomically,
# see `src.c4.train.atomic_torch_save`), but this server loaded the agent
# exactly once at startup, so a running server would serve a frozen model
# forever.
#
# These globals let `/reload` swap in a freshly written checkpoint without
# restarting the server (and without dropping an in-progress game).
_MODEL_PATH: Path | None = None   # the checkpoint file we watch
_MODEL_DEVICE: str = "cpu"
_MODEL_NAME: str = ""
_MODEL_MTIME: float = 0.0         # mtime of the checkpoint we last loaded
_MODEL_SIZE: int = -1             # size of the checkpoint we last loaded
_MODEL_SHA: str = ""              # sha256 of the checkpoint we last loaded
_MODEL_STEP: int | None = None    # training step of the checkpoint we last loaded
_RELOAD_LOCK = threading.Lock()


def _fingerprint(path: Path) -> tuple[int, float]:
    """(size, mtime) of a checkpoint file, or (-1, 0.0) if unreadable."""
    try:
        st = path.stat()
        return int(st.st_size), float(st.st_mtime)
    except OSError:
        return -1, 0.0


def _sha256(path: Path) -> str:
    """sha256 of a file, or '' if it cannot be read.

    The size+mtime pair is the cheap change detector, but it is NOT
    sufficient on its own: some copy paths preserve mtime, so a
    checkpoint can change content while keeping both size and mtime.
    The hash is the authoritative identity of what is being served.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _seat_outcome_name(outcome: int, user_seat: int) -> str:
    """Outcome name relative to the human, given the human's seat."""
    if outcome == Outcome.ONGOING:
        return "ongoing"
    if outcome == Outcome.DRAW:
        return "draw"
    winner = Outcome.WIN_P1 if outcome == Outcome.WIN_P1 else Outcome.WIN_P2
    return "you_win" if winner == user_seat else "ai_win"


def _cells(board: Board) -> list[list[int]]:
    """Return the 7x6 cell matrix as row 0 = bottom .. row ROWS-1 = top.

    Front-end renders the matrix top-down; we put bottom (gravity-wise) at
    index 0 so the rendered board matches real connect-four orientation.
    """
    rows = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = board.cell(r, c)
            row.append(int(cell))
        rows.append(row)
    return rows


class Game:
    """One in-progress game of human-vs-model on the server side."""

    def __init__(self, agent: Agent, model_name: str):
        self.agent = agent
        self.model_name = model_name
        self.lock = threading.Lock()
        self.board = Board()
        self.moves: list[int] = []
        self.last_ai_move: int | None = None
        self.last_user_move: int | None = None
        self.last_gen_ms: float = 0.0
        self.turn_label = "user"  # user always moves first (P1)
        # Which colour the human plays. 1 = P1 (moves first), 2 = P2 (moves
        # second). The model takes the other seat. The UI had no way to
        # choose, so the human was always forced onto the harder P1 side.
        self.user_seat: int = P1
        # The EXACT distribution the model sampled from when it chose its
        # last move, captured at the DECISION position (before that move was
        # applied). Without this the UI can only show the distribution of the
        # CURRENT board, which is a different position and a different
        # answer -- the classic "the panel said A but it played B" confusion.
        self.last_decision_probs: dict[str, Any] | None = None
        # The same, but for the position where the USER is to move: this is
        # the model's advice for the human's next column.
        self.user_advice_probs: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            o = self.board.outcome()
            # NOTE: probs are computed AFTER the lock is released -- see
            # _probs_locked() vs column_probs(). Calling column_probs() here
            # would deadlock on this non-reentrant lock.
            snap = {
                "cells": _cells(self.board),
                "outcome": _seat_outcome_name(o, self.user_seat),
                "turn": "user" if self.turn_label == "user" else "ai",
                "your_move": self.last_user_move,
                "ai_move": self.last_ai_move,
                "moves": list(self.moves),
                "valid_actions": self.board.valid_actions(),
                "model": self.model_name,
                "last_gen_ms": round(self.last_gen_ms, 1),
                "probs": None,
                "user_seat": self.user_seat,
                "model_seat": (P2 if self.user_seat == P1 else P1),
                "patterns": pattern_report(self.board),
                "choice_for": (
                    P2 if self.turn_label == "user" else P1
                ) if o == Outcome.ONGOING else None,
            }
            ongoing = o == Outcome.ONGOING
            moves_copy = list(self.moves)
            board_copy = self.board
            last_dec = self.last_decision_probs
            advice = self.user_advice_probs
        if ongoing:
            snap["probs"] = self.column_probs(board=board_copy, moves=moves_copy)
        # The two distributions the UI needs, kept SEPARATE and each labelled
        # with the position it was measured on:
        #   decision  -- what the model saw when it chose its last move
        #   advice    -- what the model would play from the CURRENT board
        # The UI must never present `advice` as "the move it played".
        snap["last_decision"] = last_dec
        snap["user_advice"] = advice
        return snap

    # -------------------------------------------------------- probability
    def column_probs(self, board: Board | None = None,
                     moves: list[int] | None = None) -> dict[str, Any]:
        """Return softmax over legal columns for the side-to-move.

        Takes the same ``column_logits`` the agent's ``choose()`` consumes, so
        the UI shows exactly the distribution a move would be sampled from.
        Illegal columns are masked to -1e9 and renormalised out.

        Callers that already hold ``self.lock`` should use the optional
        ``board``/``moves`` arguments (which skip re-locking); the default
        path grabs the lock itself for the GET /probs route.
        """
        with self.lock:
            if board is None:
                board = self.board
            if moves is None:
                moves = list(self.moves)
            return self._probs_locked(board, moves)

    def _probs_locked(self, board: Board, moves: list[int]) -> dict[str, Any]:
        """Compute the legal-column distribution for `board`/`moves`.

        Must be called with ``self.lock`` held (or via ``column_probs``,
        which takes the lock). Illegal columns are masked to -1e9 and
        renormalised out, matching ``Agent.choose``.

        Returns ``{"cols", "raw", "legal", "picked", "chosen_prob",
        "hard_cols"}``.
        """
        if board.outcome() != Outcome.ONGOING:
            return {"cols": [0.0] * 7, "raw": [0.0] * 7, "hard_cols": None,
                    "legal": [False] * 7, "picked": None,
                    "chosen_prob": None, "soft_picked": None}
        legal = torch.tensor([c < 6 for c in board.heights],
                             dtype=torch.bool)
        logits = self.agent.column_logits(board, list(moves))
        # Same masking as agent.choose: -1e9 for illegal columns. After
        # _load_agent() pins the model to the user's device (typically
        # CUDA), every tensor in this path MUST be on that same device --
        # mixed CPU/CUDA `torch.where` raises "two devices, cuda and cpu".
        legal_dev = legal.to(logits.device)
        NEG = torch.tensor(-1e9, dtype=logits.dtype, device=logits.device)
        masked = torch.where(legal_dev, logits, NEG)
        probs = F.softmax(masked, dim=-1)
        cpu = probs.detach().cpu().numpy().tolist()
        # Hard integer-table path: same logits, exported tables, CPU.
        try:
            hard = self.agent.hard
        except AttributeError:
            hard = None
        if hard is None:
            hard = hard_to_cpu(self.agent.model.export_hard())
            self.agent.hard = hard
        hard_logits = self.agent.hard_column_logits(board, list(moves))
        # hard_column_logits() returns CPU (hard tables are CPU by design).
        # Use the CPU `legal` mask here, but build the NEG on the hard path's
        # device too so `torch.where` does not see a device mismatch.
        hard_NEG = torch.tensor(-1e9, dtype=hard_logits.dtype)
        hard_masked = torch.where(legal, hard_logits, hard_NEG)
        hard_probs = F.softmax(hard_masked, dim=-1)
        raw = logits.detach().cpu().numpy().tolist()
        # `picked` / `chosen_prob` MUST describe the move the model actually
        # played. The live game calls choose(..., use_hard=True), so those
        # come from the HARD row. Taking them from the soft row is wrong on
        # any V6-class checkpoint: import_hard() does not exist for V6, so
        # the float parameters were never populated, column_logits() returns
        # a flat 1/7, and the panel reported "picked column A" for a move the
        # model actually played in column B. On V7 the two paths agree, so
        # this is a no-op there.
        picked = int(torch.argmax(hard_masked).item())
        chosen_prob = float(hard_probs[picked].item())
        soft_picked = int(torch.argmax(masked).item())
        # Is the FLOAT path actually trained, or is it a dead uniform row?
        # On a V6 hard export the float parameters were never populated
        # (import_hard does not exist for V6), so column_logits() returns
        # exactly 0.0 for every column. Comparing its argmax with the hard
        # argmax and reporting "argmax agrees: no (column A vs F)" is then
        # meaningless -- it compares two different models, one of which does
        # not exist. Report the flatness explicitly instead.
        soft_range = max(raw) - min(raw) if raw else 0.0
        soft_is_flat = soft_range < 1e-6
        return {
            "cols": cpu,                 # soft (float) forward -- the
                                        # probabilities choose() samples from
            "hard_cols": hard_probs.detach().cpu().numpy().tolist(),
            "raw": raw,                   # pre-softmax raw logits
            "legal": [bool(x) for x in legal.tolist()],
            "picked": picked,             # argmax over masked HARD columns
                                        #   == the move actually played
            "soft_picked": soft_picked,   # argmax over masked soft columns
            "soft_is_flat": soft_is_flat, # float path carries no signal
            "soft_range": float(soft_range),
            "chosen_prob": chosen_prob,   # hard probability of `picked`
            "model": "float column_logits + integer hard_forward",
        }

    def user_move(self, col: int) -> dict[str, Any]:
        with self.lock:
            if self.turn_label != "user":
                raise ValueError("not your turn")
            if col not in self.board.valid_actions():
                raise ValueError(f"col={col} is not playable")
            self.board.apply_move(col)
            self.moves.append(col)
            self.last_user_move = col
            self.turn_label = "ai"
            self._maybe_flip_after_user()

    def _apply_model_move(self) -> None:
        """Let the model play one move from the current position.

        The distribution the model ACTUALLY chose from is captured here, at
        the decision position (before the move is applied). Capturing it
        after applying the move would measure a different board and report a
        different column -- the bug that made the panel say "column A" while
        the model played "column B".
        """
        o = self.board.outcome()
        if o != Outcome.ONGOING:
            self.last_decision_probs = None
            self.user_advice_probs = None
            return
        t0 = time.perf_counter()
        # Measure the decision distribution BEFORE the move is applied.
        decision = self._probs_locked(self.board, list(self.moves))
        col = self.agent.choose(self.board, list(self.moves), use_hard=True)
        self.last_gen_ms = (time.perf_counter() - t0) * 1000.0
        # paranoia: legality check
        if col not in self.board.valid_actions():
            col = self.board.valid_actions()[0]
        # The model's move IS the argmax of the decision distribution. Record
        # it explicitly so the UI can show "played X" and the distribution it
        # came from, with no chance of the two drifting apart.
        decision["played"] = int(col)
        decision["played_prob"] = float(decision["hard_cols"][col]) \
            if decision.get("hard_cols") else None
        self.last_decision_probs = decision
        self.board.apply_move(col)
        self.moves.append(col)
        self.last_ai_move = col
        self.turn_label = "user"
        # The advice for the HUMAN's next move, measured on the board as it
        # stands now. This is a DIFFERENT position from the decision above
        # and must never be labelled as the move the model played.
        self.user_advice_probs = self._probs_locked(self.board, list(self.moves))

    def _maybe_flip_after_user(self) -> None:
        """After the user moved, check terminal and (if ongoing) get the
        model's reply in-place."""
        self._apply_model_move()

    def _user_played(self, index: int) -> bool:
        """True if the move at `index` in self.moves was played by the human.

        Move order depends on the seat:
          user is P1 -> [user, model, user, model, ...]  (even = human)
          user is P2 -> [model, user, model, user, ...]  (odd  = human)
        """
        return (index % 2 == 0) if self.user_seat == P1 else (index % 2 == 1)

    def undo_user(self) -> bool:
        """Take back the last human move AND the model's reply to it.

        Returns True if something was removed. The board, the move list,
        the turn and the captured distributions are all restored together,
        so the next move continues from exactly the position that was on
        screen before the undo.

        The previous implementation removed only ONE move (the model's
        reply) and then left the turn on the model, so the human's next
        click was rejected with "not your turn" -- the button looked
        broken. Undo must always hand the turn back to the human, and must
        work identically for both seats.
        """
        with self.lock:
            if not self.moves:
                return False
            if self.board.outcome() != Outcome.ONGOING:
                # Finished game: start over rather than leaving a dead
                # board on screen.
                self._reset_locked()
                return False

            # Drop the trailing moves until the last HUMAN move is gone.
            # Typically that removes exactly two moves (model reply +
            # human move); for a human move that ended the game it removes
            # one.
            while self.moves:
                last = len(self.moves) - 1
                if self._user_played(last):
                    self.moves.pop()
                    break
                self.moves.pop()   # model's reply

            # Rebuild the board exactly from the surviving move list.
            self.board = Board()
            for m in self.moves:
                if m not in self.board.valid_actions():
                    self._reset_locked()
                    return False
                self.board.apply_move(m)
            if self.board.outcome() != Outcome.ONGOING:
                # Undoing into a finished position is not useful; reset.
                self._reset_locked()
                return False
            # Restore the bookkeeping to match the replayed board.
            self.last_user_move = None
            self.last_ai_move = None
            for i, m in enumerate(self.moves):
                if self._user_played(i):
                    self.last_user_move = m
                else:
                    self.last_ai_move = m
            # The human always gets the turn back after an undo.
            self.turn_label = "user"
            self.last_gen_ms = 0.0
            # The captured decision distribution described the undone
            # position -- it must not be shown next to a different board.
            self.last_decision_probs = None
            self.user_advice_probs = self._probs_locked(
                self.board, list(self.moves))
            return True

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        """Reset the game. MUST be called with ``self.lock`` held.

        ``threading.Lock`` is NOT reentrant, so every path that resets from
        inside a locked section has to use this, not ``reset()`` -- calling
        ``reset()`` from inside the lock deadlocks the server.
        """
        self.board = Board()
        self.moves = []
        self.last_user_move = None
        self.last_ai_move = None
        self.turn_label = "user"
        self.last_gen_ms = 0.0
        self.last_decision_probs = None
        self.user_advice_probs = None
        # If the human plays P2, the model owns P1 and must open the
        # game. The model is greedy, so its first move is deterministic.
        if self.user_seat == P2:
            self.board.turn = P1
            self._apply_model_move()

    def set_seat(self, seat: int) -> None:
        """Choose the human's colour: 1 = P1 (moves first), 2 = P2.

        Changing seats starts a fresh game -- the two seats are different
        games, so continuing a P1 game as P2 would be incoherent.
        """
        seat = int(seat)
        if seat not in (P1, P2):
            raise ValueError("seat must be 1 (P1) or 2 (P2)")
        with self.lock:
            self.user_seat = seat
            self._reset_locked()


def _torch_load_with_retry(ckpt_path, attempts: int = 40, delay_s: float = 0.25):
    """`torch.load` with a short retry for transient Windows file locks.

    The trainer publishes checkpoints atomically (temp file + rename), so a
    checkpoint on disk is never half-written.  But on Windows, a rename
    over a file that this process has open fails, and a reader can hit
    `PermissionError: [Errno 13]` while the writer swaps the file.

    A transient lock is not a corrupt file, so it must not be reported as
    one.  Retrying for up to ~10 s turns a momentary lock into a normal
    load; a genuinely broken file still fails, just a moment later.
    """
    last_exc = None
    for i in range(attempts):
        try:
            # weights_only=False is required for this repo's own hard
            # export payloads: they are dicts of LISTS of tensors plus
            # (channel, lag) int tuples, not a plain tensor dict.
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except PermissionError as e:
            last_exc = e
            time.sleep(delay_s)
        except FileNotFoundError:
            # The trainer may be mid-swap: the rename briefly removes the
            # target from the directory view.  That is transient too.
            last_exc = None
            time.sleep(delay_s)
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"checkpoint not found after {attempts} attempts: {ckpt_path}")


def _load_agent(ckpt_path: str, device: str = "cpu") -> Agent:
    """Rebuild the exact ``GFMarkovLM`` that produced an ``export_hard()``
    payload, then load the integer weights.

    The payload is self-describing, so no arch guess is needed:
      d      = emb.shape[1]
      layers = len(coef)
      lags   = payload["lags"]
    V6 and V7 differ only in those numbers (d32/L3/lags<=64 vs
    d64/L6/lags<=256), and both classes take the same kwargs.

    WINDOWS FILE LOCKING
    --------------------
    The trainer publishes checkpoints by writing a temp file and renaming
    it over the target, which is atomic.  But on Windows a rename over a
    file that a reader currently has open fails, and (more visibly here)
    the READER can get `PermissionError: [Errno 13]` while the writer is
    in the middle of the swap.

    So a load can fail for a perfectly good file.  This helper retries
    briefly instead of propagating the error, because the caller
    (`_reload_agent`) treats a failed load as "keep serving the old
    model", and a transient lock would otherwise mean the UI silently
    stays on a stale checkpoint for a whole training run.
    """
    # weights_only=False is required: an export_hard() payload is a dict of
    # LISTS of tensors (coef/sbox/wiring/readout), not a single tensor dict,
    # and the wiring entries are (channel, lag) int tuples. The file is
    # produced by this repo's own training run, so it is trusted input.
    raw = _torch_load_with_retry(ckpt_path)
    payload = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

    d = int(payload["emb"].shape[1])
    layers = len(payload["coef"])
    lags = tuple(int(x) for x in payload["lags"])
    fan_in = int(payload["coef"][0].shape[1])

    cls = GFMarkovLM_V7 if d >= 64 else GFMarkovLM_V6
    model = cls(d=d, layers=layers, fan_in=fan_in, lags=lags)

    hard = hard_to_cpu(payload)
    # The transcript format the WEIGHTS were trained on travels in the
    # checkpoint meta. Serving must encode in exactly that format: a v1 model
    # fed v2 bytes is reading a different alphabet, which does not crash --
    # it plays badly and looks like a weak model. Checkpoints written before
    # the v2 migration carry no version and are v1 by definition.
    #
    # export_hard() flattens the checkpoint: the metadata is promoted to the
    # TOP level of the payload ("transcript", "step", "ladder_net", ...),
    # NOT nested under a "meta" key. A hard export therefore has no
    # raw["meta"] at all. Looking only at raw["meta"] silently fell back to
    # v1 for every hard export and made the served model play the WRONG
    # alphabet -- no crash, just a bad player. Read the top level first,
    # then fall back to the nested meta for soft checkpoints.
    meta = raw.get("meta") if isinstance(raw, dict) else None
    if not isinstance(meta, dict):
        meta = raw if isinstance(raw, dict) else {}
    ckpt_ver = meta.get("transcript")
    # A checkpoint with no recorded version predates the v2 migration, so it
    # is v1 BY DEFINITION. Defaulting it to v2 would feed those weights a
    # different byte alphabet: no crash, just a model that plays badly and
    # looks undertrained. Only v2 checkpoints opt in, by writing the meta.
    version = fmt.normalize(ckpt_ver if ckpt_ver is not None else "v1")
    if ckpt_ver is None:
        print(f"ui_server: WARNING checkpoint declares no transcript version; "
              f"assuming {version}. A v2 model served as v1 plays badly.")
    else:
        print(f"ui_server: checkpoint transcript={version}")
    agent = Agent(model, torch.device(device), greedy=True, version=version)
    agent.hard = hard
    # import_hard() also copies every parameter (emb/coef/sbox/wires/bias/
    # readout[i]/readout_bias) from the integer export into the float
    # nn.Parameter buffers. Without this, model.readout[0..N] are still
    # random-init zeros and the soft column_logits() path collapses to
    # uniform garbage. call it BEFORE the device move so subsequent
    # ``model.to(device)`` ships the populated tensors.
    # We keep ``agent.hard`` populated too -- ``choose(use_hard=True)`` is
    # the live-game inference path.
    #
    # V6-class payloads (d < 64) predate import_hard -- that inverse was
    # added in V7, so calling it unconditionally raises AttributeError on an
    # otherwise perfectly loadable V6 model. It only feeds the SOFT
    # column_logits() path; ``choose(use_hard=True)`` -- what the live game
    # calls -- reads ``agent.hard`` directly and never touches the float
    # buffers. Skipping it is correct V6 serving, not a degraded fallback.
    if hasattr(agent.model, "import_hard"):
        agent.model.import_hard(hard)
    # Serving the float column_logits() path requires the parameters on
    # the same device as the prompt tokens, so move it explicitly.
    want = torch.device(device)
    agent.model.to(want)
    # Serving must be inference-only: training=True enables checkpointing and
    # STE paths that only the trainer wants, and leaves dropout/BN-type state
    # active. hard_forward() is unaffected (it reads only `hard`).
    agent.model.eval()
    torch.set_grad_enabled(False)
    return agent


def _checkpoint_step(ckpt_path: Path) -> int | None:
    """Best-effort read of the `step` recorded in a checkpoint.

    Used only for reporting ("we are now serving step 12345"), never for
    correctness. A failure here must not stop a reload.
    """
    try:
        raw = _torch_load_with_retry(ckpt_path)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    meta = raw.get("meta")
    if isinstance(meta, dict) and meta.get("step") is not None:
        try:
            return int(meta["step"])
        except (TypeError, ValueError):
            pass
    for key in ("step",):
        if raw.get(key) is not None:
            try:
                return int(raw[key])
            except (TypeError, ValueError):
                pass
    return None


def _reload_agent(reason: str = "manual") -> dict[str, Any]:
    """Re-read the watched checkpoint and swap it into the live game(s).

    Returns a status dict for the HTTP layer. The in-progress board and
    move history are PRESERVED: the human keeps playing the same game,
    but the model behind it is the newest checkpoint.

    With per-session games there is no single `_GAME` to swap, so the
    new agent is pushed into EVERY live session. A visitor mid-game
    keeps their board and starts playing against the improved model.
    """
    global _MODEL_MTIME, _MODEL_SIZE, _MODEL_SHA, _MODEL_STEP
    global _CURRENT_AGENT
    with _RELOAD_LOCK:
        if _MODEL_PATH is None:
            return {"ok": False, "error": "no checkpoint is being watched"}
        ckpt = _MODEL_PATH
        if not ckpt.exists():
            return {"ok": False, "error": f"checkpoint not found: {ckpt}"}
        try:
            size, mtime = _fingerprint(ckpt)
            agent = _load_agent(str(ckpt), device=_MODEL_DEVICE)
        except Exception as e:
            # A reload must never take the server down: a half-written or
            # corrupt file is a recoverable condition. The trainer writes
            # atomically, so this is rare, but a failed reload keeps the
            # previous (working) model in place.
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        n_swapped = 0
        # Publish the new agent BEFORE swapping the sessions, so a
        # session created concurrently picks up the new model rather
        # than the old one.
        _CURRENT_AGENT = agent
        if _SESSIONS is not None:
            for g in _SESSIONS.all_games():
                # Swap the model only; keep board/moves/turn so the game
                # continues seamlessly.
                try:
                    with g.lock:
                        g.agent = agent
                        g.model_name = _MODEL_NAME
                    n_swapped += 1
                except Exception:
                    # One broken session must not abort the swap for the
                    # others.
                    continue
        if _GAME is not None:
            with _GAME.lock:
                _GAME.agent = agent
                _GAME.model_name = _MODEL_NAME
            n_swapped += 1
        _MODEL_MTIME = mtime
        _MODEL_SIZE = size
        _MODEL_SHA = _sha256(ckpt)
        _MODEL_STEP = _checkpoint_step(ckpt)
        return {
            "ok": True,
            "reason": reason,
            "checkpoint": str(ckpt),
            "step": _MODEL_STEP,
            "mtime": mtime,
            "size": size,
            "sha256": _MODEL_SHA[:16],
            "sessions_swapped": n_swapped,
        }


def _watched_checkpoint_changed() -> bool:
    """True if the watched checkpoint differs from the one we last loaded.

    Uses size+mtime as the fast path, and falls back to a content hash
    when the cheap check says "unchanged" but the caller wants proof.
    `changed_on_disk` reports the cheap check; `changed_strict` reports
    the hash-verified answer.
    """
    if _MODEL_PATH is None or not _MODEL_PATH.exists():
        return False
    size, mtime = _fingerprint(_MODEL_PATH)
    if size != _MODEL_SIZE or mtime > _MODEL_MTIME:
        return True
    # Cheap check says unchanged. mtime can be preserved by copy paths,
    # so verify by content before claiming nothing changed.
    sha = _sha256(_MODEL_PATH)
    return bool(sha) and bool(_MODEL_SHA) and sha != _MODEL_SHA


_GAME: Game | None = None
_HTML_PATH: Path = HERE / "ui.html"

# ------------------------------------------------------------- sessions
# PUBLIC DEPLOYMENT REQUIREMENT
# ---------------------------
# The original server kept ONE game in the module-level `_GAME` global.
# Every visitor shared one board: a move by one person moved everyone
# else's game, and the board state leaked across visitors. That is fine
# for a single user on a laptop and completely wrong for a public URL.
#
# The fix is per-visitor state:
#   * each visitor gets an opaque session id in an HttpOnly cookie;
#   * each session id maps to its OWN Game instance;
#   * the MODEL stays shared (it is read-only during inference), so one
#     copy of the 32 MB checkpoint serves every visitor.
#
# `_SESSIONS` is the store. It is created in `main()` once the agent is
# loaded, because the factory needs the agent.
_SESSIONS: SessionStore | None = None
# Set by main(): when True the session cookie carries the `Secure`
# attribute, because the site is then served over HTTPS.
_COOKIE_SECURE = False
# Set by main(): the set of origins allowed to send state-changing
# requests. It contains the site's own origin plus every allowlisted
# embedding origin (a framed game is a cross-origin request by
# definition, so the embedding parents have to be trusted too).
_ALLOWED_ORIGINS: set[str] = set()
# Set by main(): the SameSite policy of the session cookie. "Lax" is the
# default; it becomes "None" when the game is embedded in a frame on
# another site, because a Lax cookie is NOT sent on a cross-site
# subresource request and every framed request would start a new game.
_COOKIE_SAMESITE = "Lax"
# Set by main(): when True, visitors are rate limited per session.
_RATE_LIMIT = True
# Set by main(): a background sweeper drops idle sessions.
_SESSION_SWEEP_INTERVAL_S = 60.0
# Set by main(): when True, the operational routes (/modelinfo, /reload)
# are reachable. They expose the checkpoint path and allow swapping the
# model, so they are LOCAL-ONLY by default and must be turned off for a
# public deployment.
_ALLOW_ADMIN = True
# Set by main(): the list of origins allowed to EMBED this server in an
# iframe, e.g. ["https://chris4k-connect-four-gf256.static.hf.space"].
#
# Empty (the default) means "nobody may frame this": the response then
# carries `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'`, which
# is the correct posture for a site that is meant to be visited directly.
# This is an explicit ALLOWLIST, never a wildcard -- an open
# `frame-ancestors *` would let any site clickjack the game.
_FRAME_ANCESTORS: list[str] = []
# The agent the server is currently serving. A hot reload replaces it.
# The session factory reads THIS global (not a closure over the startup
# agent) so a session created after a reload plays against the new
# checkpoint.
_CURRENT_AGENT: Any = None
# Idle lifetime of a session, in seconds.
DEFAULT_SESSION_TTL_S = 2 * 3600


class Handler(BaseHTTPRequestHandler):
    # quiet the default access log; the model's load time matters and stdout
    # noise is misleading.
    def log_message(self, format, *args):  # noqa: A003 - stdlib override
        sys.stderr.write("ui_server: " + (format % args) + "\n")

    # ------------------------------------------------------------- session
    def _session(self, *, create: bool = True):
        """Resolve this request's session, creating one if needed.

        Returns ``(sid, game, is_new)``. `game` is None only when the
        session store is unavailable (the agent failed to load).

        The session id comes from the `tg_sid` cookie. A cookie that is
        absent, malformed, or points at an expired session yields a NEW
        session -- never an error, and never a shared fallback game.
        """
        if _SESSIONS is None:
            return None, None, False
        cookies = parse_cookies(self.headers.get("Cookie"))
        sid = cookies.get("tg_sid")
        if sid is not None and not valid_session_id(sid):
            # A malformed id is refused outright rather than repaired:
            # silently repairing it would let two different malformed
            # ids collapse onto one session.
            sid = None
        game = _SESSIONS.get(sid) if sid else None
        if game is not None:
            return sid, game, False
        if not create:
            return sid, None, False
        new_sid, game = _SESSIONS.create(None)
        # Remember it so `_send_json` can emit the Set-Cookie header.
        self._new_session_sid = new_sid
        return new_sid, game, True

    def _rate_limited(self) -> bool:
        """True when this request should be rejected as over budget."""
        if not _RATE_LIMIT or _SESSIONS is None:
            return False
        cookies = parse_cookies(self.headers.get("Cookie"))
        sid = cookies.get("tg_sid")
        if sid is not None and not valid_session_id(sid):
            sid = None
        return not _SESSIONS.allow(sid)

    def _csp(self) -> str:
        """The Content-Security-Policy for this response.

        The policy is strict by default: nothing is loadable except the
        server's own inline script and styles. `frame-ancestors` names the
        origins allowed to embed the page, or is 'none' when nobody may.

        `frame-ancestors` replaces `X-Frame-Options` when an allowlist is
        in force, because XFO can only allow or deny ONE origin while the
        allowlist may name several, and because every current browser
        understands CSP.
        """
        ancestors = " ".join(_FRAME_ANCESTORS) if _FRAME_ANCESTORS else "'none'"
        return (
            "default-src 'none'; script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'none'; "
            f"frame-ancestors {ancestors}")

    def _send_json(self, code: int, payload: dict[str, Any],
                   extra_headers: list[tuple[str, str]] | None = None) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Security headers for a PUBLIC deployment. The UI is entirely
        # self-hosted (no external scripts, no external fonts, no images
        # from other origins), so a strict CSP costs nothing and blocks
        # the classic injection paths.
        self.send_header("X-Content-Type-Options", "nosniff")
        if not _FRAME_ANCESTORS:
            # Legacy header, kept for the no-embed case: it covers very
            # old browsers that ignore CSP.
            self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", self._csp())
        for k, v in (extra_headers or []):
            self.send_header(k, v)
        # A brand-new session must be told its id. An existing session
        # keeps the same cookie, so it is NOT re-sent on every request.
        new_sid = getattr(self, "_new_session_sid", None)
        if new_sid:
            self.send_header("Set-Cookie", session_cookie(
                new_sid, secure=_COOKIE_SECURE,
                max_age=int(DEFAULT_SESSION_TTL_S),
                samesite=_COOKIE_SAMESITE))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        ctype = "text/html; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # The same security headers as the JSON routes, so the HTML
        # document itself is protected (X-Frame-Options matters most
        # here: it is what stops clickjacking the game).
        self.send_header("X-Content-Type-Options", "nosniff")
        if not _FRAME_ANCESTORS:
            self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", self._csp())
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if self.path == "/" or self.path.startswith("/index"):
            self._send_file(_HTML_PATH)
            return
        # /modelinfo and /reload are OPERATIONAL routes: they expose the
        # checkpoint path and allow swapping the model. They must never
        # be reachable from the public internet, so they are refused
        # unless the server was explicitly started in local mode.
        if self.path.startswith("/modelinfo"):
            if not _ALLOW_ADMIN:
                self._send_json(403, {"error": "forbidden"})
                return
            info = {
                "checkpoint": str(_MODEL_PATH) if _MODEL_PATH else None,
                "model_name": _MODEL_NAME,
                "device": _MODEL_DEVICE,
                "mtime": _MODEL_MTIME,
                "size": _MODEL_SIZE,
                "sha256": _MODEL_SHA[:16] if _MODEL_SHA else None,
                "changed_on_disk": _watched_checkpoint_changed(),
                # The training step of the served checkpoint.  The watcher
                # (`play_v24.sh`) reports this to the user, so it has to
                # come from the payload rather than from the filename --
                # the promoted artifact is `best_hard.pt` and carries no
                # step in its name.
                #
                # Cached at load time: reading it here would load a
                # 34 MB checkpoint on EVERY poll, and the watcher polls
                # every couple of seconds.  That starved the server's
                # request threads and made live requests time out.
                "step": _MODEL_STEP,
                "sessions": _SESSIONS.stats() if _SESSIONS else None,
            }
            self._send_json(200, info)
            return
        if self.path.startswith("/reload"):
            if not _ALLOW_ADMIN:
                self._send_json(403, {"error": "forbidden"})
                return
            # Hot-swap in a freshly written checkpoint. The in-progress
            # game is preserved; only the model behind it changes.
            status = _reload_agent(reason="http")
            self._send_json(200 if status.get("ok") else 409, status)
            return

        if self._rate_limited():
            self._send_json(429, {"error": "too many requests"})
            return

        sid, game, is_new = self._session(create=True)
        if game is None:
            _send_unavailable(self)
            return
        if self.path == "/reset":
            game.reset()
            self._send_json(200, game.snapshot())
            return
        if self.path == "/undo":
            ok = game.undo_user()
            self._send_json(200, {"undone": ok, "state": game.snapshot()})
            return
        if self.path == "/state":
            self._send_json(200, game.snapshot())
            return
        if self.path.startswith("/probs"):
            self._send_json(200, game.column_probs())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "unknown GET")

    def _origin_allowed(self) -> bool:
        """True when a state-changing request comes from a trusted origin.

        `SameSite=Lax` used to make a cross-site POST impossible: the
        browser withheld the session cookie. When the game is framed on
        another site the cookie must be `SameSite=None`, which removes
        that implicit protection -- so the check has to be explicit.

        A request WITHOUT an `Origin` header is allowed: that is a
        non-browser client (curl, the test-suite, a server-side call),
        which cannot be ridden by a hostile page in the first place.
        A request WITH an `Origin` must name either the site itself or
        one of the explicitly allowlisted embedding origins.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        if origin in _ALLOWED_ORIGINS:
            return True
        return False

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send_json(403, {"error": "origin not allowed"})
            return
        if self.path.startswith("/reload") or self.path.startswith("/modelinfo"):
            if not _ALLOW_ADMIN:
                self._send_json(403, {"error": "forbidden"})
                return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            req = json.loads(raw or b"{}")
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        if self._rate_limited():
            self._send_json(429, {"error": "too many requests"})
            return

        sid, game, is_new = self._session(create=True)
        if game is None:
            _send_unavailable(self)
            return

        if self.path == "/seat":
            try:
                game.set_seat(int(req["seat"]))
            except (KeyError, ValueError) as e:
                self._send_json(400, {"error": f"bad seat: {e}"})
                return
            self._send_json(200, game.snapshot())
            return

        if self.path == "/undo":
            ok = game.undo_user()
            self._send_json(200, {"undone": ok, "state": game.snapshot()})
            return

        if self.path != "/move":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown POST")
            return
        try:
            col = int(req["col"])
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return
        try:
            game.user_move(col)
        except ValueError as e:
            self._send_json(400, {"error": str(e),
                                  "state": game.snapshot()})
            return
        self._send_json(200, game.snapshot())


def _send_unavailable(handler: Handler) -> None:
    handler._send_json(503, {"error": "agent not loaded - is the model path valid?"})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="path to a best_hard.pt (export_hard() payload)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # --- public-deployment switches ------------------------------------
    p.add_argument("--public", action="store_true",
                   help="serve a PUBLIC site: disable /modelinfo and "
                        "/reload, mark the session cookie Secure, and "
                        "keep the rate limit on")
    p.add_argument("--no-rate-limit", action="store_true",
                   help="disable the per-session request budget (local "
                        "debugging only)")
    p.add_argument("--session-ttl", type=float, default=2 * 3600.0,
                   help="idle lifetime of a session, in seconds")
    p.add_argument("--max-sessions", type=int, default=512,
                   help="maximum concurrent sessions (LRU eviction)")
    p.add_argument("--allow-embedding", action="append", default=None,
                   metavar="ORIGIN",
                   help="allow this origin to embed the game in an iframe. "
                        "Repeatable. Only needed for the Hugging Face Space "
                        "that frames the game. Without it the page is "
                        "framed by nobody (X-Frame-Options: DENY and CSP "
                        "frame-ancestors 'none').")
    p.add_argument("--cookie-secure", choices=["auto", "yes", "no"],
                   default="auto",
                   help="whether the session cookie carries the Secure "
                        "attribute. 'auto' means Secure in --public mode. "
                        "Use 'no' ONLY for a plain-HTTP test harness: a "
                        "browser refuses to store a Secure cookie over "
                        "HTTP, so the visitor would silently lose their "
                        "session on every request.")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"ui_server: model not found at {args.model}", file=sys.stderr)
        return 2

    global _COOKIE_SECURE, _RATE_LIMIT, _ALLOW_ADMIN, _SESSIONS
    global _FRAME_ANCESTORS, _COOKIE_SAMESITE, _ALLOWED_ORIGINS
    _COOKIE_SECURE = bool(args.public)
    if args.cookie_secure == "yes":
        _COOKIE_SECURE = True
    elif args.cookie_secure == "no":
        _COOKIE_SECURE = False
    _RATE_LIMIT = not args.no_rate_limit
    # The operational routes expose the checkpoint path and allow
    # swapping the model. They are useful locally (the watcher polls
    # /modelinfo) and dangerous in public, so `--public` turns them off.
    _ALLOW_ADMIN = not args.public

    # --- embedding allowlist ------------------------------------------
    # Only https origins are accepted: an http ancestor could be
    # intercepted, and a wildcard would let any site clickjack the game.
    origins = [o.strip() for o in (args.allow_embedding or []) if o.strip()]
    for o in origins:
        if not o.startswith("https://"):
            print(f"ui_server: refusing embedding origin without https: {o}",
                  file=sys.stderr)
            return 2
        if o == "https://*":
            print("ui_server: refusing a wildcard embedding origin",
                  file=sys.stderr)
            return 2
    _FRAME_ANCESTORS = origins
    # The site's own origin is always trusted, plus every embedding
    # origin (the framed parents that legitimately drive the game).
    _ALLOWED_ORIGINS.clear()
    _ALLOWED_ORIGINS.add("https://c4.tensorgarten.de")
    _ALLOWED_ORIGINS.update(origins)
    if _FRAME_ANCESTORS:
        print("ui_server: embedding allowed for: "
              + ", ".join(_FRAME_ANCESTORS), file=sys.stderr)

    # --- session cookie policy ----------------------------------------
    # A `SameSite=Lax` cookie is NOT sent on a cross-site subresource
    # request. In an iframe on another site (the Hugging Face Space, or
    # the HF viewer wrapping that Space) the browser would therefore drop
    # the session id on every request: the board would render, but each
    # move would arrive without a session and the game would reset.
    # `SameSite=None` is the only value that survives a cross-site
    # iframe, and browsers only accept it together with `Secure`.
    # So: only relax the policy when the site is actually framed, and
    # only when the cookie is already Secure (which `--public` sets).
    if _FRAME_ANCESTORS:
        if not _COOKIE_SECURE:
            print("ui_server: embedding requested but the cookie is not "
                  "Secure; keeping SameSite=Lax (a framed game would not "
                  "keep its session)", file=sys.stderr)
        else:
            _COOKIE_SAMESITE = "None"
    else:
        _COOKIE_SAMESITE = "Lax"
    print(f"ui_server: session cookie SameSite={_COOKIE_SAMESITE}",
          file=sys.stderr)

    # The default request_queue_size is 5. The live-parity sweep fires
    # requests back-to-back (GET /state, POST /move, GET /state, ...) with
    # a model call in the middle of each POST, so several connections are
    # pending in the accept queue at once. With a backlog of 5 the queue
    # overflows and the NEXT connect() times out:
    #
    #   urllib.error.URLError: <urlopen error [WinError 10060]>
    #   Ein Verbindungsversuch ist fehlgeschlagen, da die Gegenstelle
    #   nach einer bestimmten Zeitspanne nicht richtig reagiert hat
    #
    # That is a server-capacity symptom, not a model defect -- the same
    # test passes standalone. A backlog of 128 removes the race.
    class _Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True
        # NOTE: allow_reuse_address is deliberately LEFT AT ITS DEFAULT
        # (False on Windows).
        #
        # On Windows, SO_REUSEADDR does NOT mean "reuse an address that is
        # still bound" the way it does on Linux -- Windows honours it for
        # TIME_WAIT sockets but STILL lets a second live socket bind the
        # same address. The visible symptom was a STALE ui_server from an
        # earlier session silently shadowing the freshly started one:
        #
        #   * both processes printed "ui_server: serving http://127.0.0.1/"
        #   * netstat showed many ESTABLISHED/WARTEND (LISTENING) rows for
        #     the same port, each owned by a different PID
        #   * requests were served by the OLD process, so /modelinfo lacked
        #     the `step` field and the live-parity tests intermittently
        #     failed with connection resets
        #
        # The regression sweep then "fixed" itself at random, because the
        # shadowed server died with its session. The real defect was two
        # servers on one port, not a flaky test.
        #
        # The explicit pre-bind probe below makes the collision LOUD
        # instead of silent, so a stale server can never shadow a new one.
        allow_reuse_address = False

    # Claim the port BEFORE loading the 34 MB model.  Binding first means
    # the port is held for the whole startup, so a second server can never
    # slip in between the check and the bind -- and a stale server is
    # rejected in milliseconds instead of after a model load.
    try:
        srv = _Server((args.host, args.port), Handler)
    except OSError as exc:
        print(f"ui_server: cannot bind {args.host}:{args.port}: {exc}",
              file=sys.stderr)
        print("ui_server: refusing to start a second server on the same "
              "port (on Windows a second bind silently shadows the first). "
              "Stop the old server, or pass --port <other>.", file=sys.stderr)
        return 3

    try:
        print("ui_server: loading agent ...", flush=True)
        agent = _load_agent(args.model, device=args.device)
        model_name = os.path.basename(args.model).removesuffix(".pt")
        global _GAME, _MODEL_PATH, _MODEL_DEVICE, _MODEL_NAME, _MODEL_MTIME
        global _MODEL_SIZE, _MODEL_SHA, _MODEL_STEP
        global _CURRENT_AGENT
        # Remember what we are serving so /reload can pick up a newer
        # checkpoint written by a running trainer, without restarting.
        _MODEL_PATH = Path(args.model).resolve()
        _MODEL_DEVICE = args.device
        _MODEL_NAME = model_name
        _CURRENT_AGENT = agent
        _MODEL_SIZE, _MODEL_MTIME = _fingerprint(_MODEL_PATH)
        _MODEL_SHA = _sha256(_MODEL_PATH)
        # Read the step ONCE, here, while we are already loading the file.
        # /modelinfo reports this cached value; reading it per-request would
        # load a 34 MB checkpoint on every poll and starve the server.
        _MODEL_STEP = _checkpoint_step(_MODEL_PATH)

        # ---------------------------------------------------- sessions
        # Each visitor gets their OWN Game. The factory reads the
        # CURRENT agent from a module global rather than closing over
        # the startup one: a session created AFTER a hot reload must
        # play against the new checkpoint, not the one that was loaded
        # when the process started.
        def _new_game():
            return Game(_CURRENT_AGENT, model_name=model_name)

        _SESSIONS = SessionStore(
            factory=_new_game,
            ttl_s=float(args.session_ttl),
            max_sessions=int(args.max_sessions),
        )

        # Sweep idle sessions so an unattended public server does not
        # accumulate boards forever. A daemon thread, so it dies with the
        # process.
        store = _SESSIONS

        def _sweeper():
            while True:
                time.sleep(_SESSION_SWEEP_INTERVAL_S)
                try:
                    n = store.sweep()
                    if n:
                        print(f"ui_server: swept {n} idle session(s)",
                              flush=True)
                except Exception:
                    pass

        threading.Thread(target=_sweeper, daemon=True,
                         name="tg-session-sweeper").start()

        print(f"ui_server: serving http://{args.host}:{args.port}/ "
              f"(model={args.model})", flush=True)
        if _ALLOW_ADMIN:
            print("ui_server: hot reload enabled -- GET /reload swaps in a "
                  "newer checkpoint, GET /modelinfo shows what is served",
                  flush=True)
        else:
            print("ui_server: PUBLIC mode -- /modelinfo and /reload are "
                  "disabled; per-session games are isolated", flush=True)
        print(f"ui_server: session isolation on (ttl={args.session_ttl:.0f}s, "
              f"max={args.max_sessions}, rate_limit={_RATE_LIMIT})",
              flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
