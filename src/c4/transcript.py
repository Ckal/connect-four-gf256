"""Byte encoding for Connect-Four transcripts (10 bytes per ply).

A ply is

    [h0..h6]  [last]  [turn]  [move]                 (PLY_LEN = 10 bytes)

where h0..h6 are the seven column heights in the position BEFORE the move
is made (offset by 0x30), `last` is the column just played (NO_LAST=0 on
ply 0), `turn` identifies the side to move next (TOK_X=0x58 for P1,
'O' 0x4F for P2), and `move` is the column the side on `turn` actually
picked (0x41..0x47 = 'A'..'G').

Every ply is followed by a newline (0x0A). A full 42-ply game is

    42 * PLY_LEN           = 420 bytes of payload
    42 newlines            = 42 bytes
    -------------------------------
    462 bytes              (MAX_LINE)

The model context window is 512, comfortably larger.

Why height-based plies and not a raw 42-byte board snapshot per ply:
  * Sequence length scales with game length, not board area.
  * An opening ply is 10 bytes; the constant-empty 42-byte grid would be
    51 bytes per ply, ~90% constant noise.
  * The full position is still preserved: each ply records every column's
    height, which exactly reconstructs the masks given the `turn` byte
    and the placement order encoded in the sequence.
  * Crucial: every move byte is a well-posed prediction target given the
    heights/last/turn prefix -- the loss mask only counts the `move` slot.

Why one newline per ply (and not one per game):
  * Predicting the next byte is then "predict the next column", even when
    training on a single game. Otherwise the byte after the last `move`
    of a game would be a column-height header byte, not a move, and the
    model would be told the move wrong.
"""

from __future__ import annotations

from . import P1, P2, TOK_X, ROWS, COLS
from .board import Board

COL_BYTE_OFFSET = 0x41                # 'A'
N_COL_BYTES = 7                       # 'A'..'G'
COL_BYTES = b"ABCDEFG"                # 0x41..0x47
NO_LAST = 0x00                        # ply 0 sentinel for the `last` slot
TOK_TURN_P1 = TOK_X                   # 0x58 -- P1 to move
TOK_TURN_P2 = 0x4F                    # 'O'   -- P2 to move
PLY_LEN = 10                          # bytes per ply record (9 header + 1 move)
HEADER_LEN = PLY_LEN - 1              # 9 header bytes before the move byte
MOVE_OFFSET = 9                       # index of the move byte in a ply
MAX_PLIES = 42                        # full connect-four game
MAX_LINE = MAX_PLIES * PLY_LEN + MAX_PLIES  # 420 + 42 newline = 462

# Height encoding: heights fit in 0..ROWS (6), so we offset by 0x30 ('0').
# Gives every height a printable byte and keeps the stream ASCII-friendly.
HGT_ZERO = 0x30                       # b'0'   height 0
HGT_TOP = HGT_ZERO + ROWS             # 0x36   height 6 (full column)
COL_SLICE = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + N_COL_BYTES)


def encode_state(board, last, side_to_move):
    """One 9-byte header [h0..h6, last, turn] for the position BEFORE the move."""
    out = bytearray(HEADER_LEN)
    for c in range(COLS):
        n = board.heights[c]
        if not 0 <= n <= ROWS:
            raise ValueError(f"bad column height {n} for col {c}")
        out[c] = HGT_ZERO + n
    out[COLS] = NO_LAST if last is None else COL_BYTE_OFFSET + last
    out[COLS + 1] = TOK_TURN_P1 if side_to_move == P1 else TOK_TURN_P2
    return bytes(out)


def decode_state(header):
    """Inverse of encode_state. Returns (heights, last_col, side_to_move)."""
    if len(header) != HEADER_LEN:
        raise ValueError(f"header must be {HEADER_LEN} bytes, got {len(header)}")
    heights = [header[c] - HGT_ZERO for c in range(COLS)]
    for n in heights:
        if not 0 <= n <= ROWS:
            raise ValueError(f"bad decoded height {n}")
    last_byte = header[COLS]
    last = None if last_byte == NO_LAST else last_byte - COL_BYTE_OFFSET
    if last is not None and not (0 <= last < COLS):
        raise ValueError(f"bad last column {last}")
    turn = P1 if header[COLS + 1] == TOK_TURN_P1 else P2
    return heights, last, turn


def encode_game(moves):
    """Encode a full game as a sequence of newline-terminated ply records.

    Each ply's [h0..h6, last, turn] is computed against the board state
    AFTER replaying the prior moves. Ply 0's `last` is NO_LAST (no move yet).
    """
    if len(moves) > MAX_PLIES:
        raise ValueError(f"game too long: {len(moves)} plies (max {MAX_PLIES})")
    b = Board()
    side = P1
    last = None
    out = bytearray()
    for m in moves:
        header = encode_state(b, last, side)
        out.extend(header)
        out.append(COL_BYTE_OFFSET + m)
        out.append(0x0A)               # newline
        b.apply_move(m)
        last = m
        side = P2 if side == P1 else P1
    return bytes(out)


def plies_of(line):
    """Number of plies encoded in `line`.

    There is one newline per ply (interleaved), not one trailing newline,
    so the ply count IS the newline count. The remaining bytes are the
    payload, which must be exactly PLY_LEN per ply.
    """
    if isinstance(line, str):
        line = line.encode("ascii")
    line = bytes(line)
    n = line.count(b"\n")
    payload = len(line) - n
    if payload % PLY_LEN:
        raise ValueError(f"payload {payload}B not a multiple of {PLY_LEN}")
    if payload // PLY_LEN != n:
        raise ValueError(f"{payload // PLY_LEN} plies of payload vs {n} newlines")
    return n


def decode_game(line):
    """Inverse of encode_game. Returns the move list."""
    if isinstance(line, str):
        line = line.encode("ascii")
    line = bytes(line)
    # strip the interleaved newlines -> a dense 10-bytes-per-ply payload
    flat = line.replace(b"\n", b"")
    plies = len(flat) // PLY_LEN
    if plies * PLY_LEN != len(flat):
        raise ValueError(f"payload {len(flat)}B not a multiple of {PLY_LEN}")
    moves = []
    b = Board()
    side = P1
    for i in range(plies):
        rec = flat[i * PLY_LEN:(i + 1) * PLY_LEN]
        header, mbyte = rec[:HEADER_LEN], rec[-1]
        _heights, last, turn = decode_state(header)
        if turn != side:
            raise ValueError(f"ply {i}: expected side {side}, got {turn}")
        if last is not None and moves and last != moves[-1]:
            raise ValueError(f"ply {i}: header last={last} but prior move={moves[-1]}")
        col = mbyte - COL_BYTE_OFFSET
        if not (0 <= col < COLS):
            raise ValueError(f"bad move byte {mbyte:#x}")
        if b.heights[col] >= ROWS:
            raise ValueError(f"ply {i}: col {col} already full")
        moves.append(col)
        b.apply_move(col)
        side = P2 if side == P1 else P1
    return moves


def load_corpus_lines(path):
    """Read a corpus file one line at a time. Empty lines skipped."""
    lines = []
    with open(path, "rb") as f:
        for ln in f:
            s = ln.strip()
            if s:
                lines.append(s)
    return lines

