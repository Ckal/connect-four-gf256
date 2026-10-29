"""V2 transcript encoding: per-cell ownership instead of bare column heights.

WHY THIS EXISTS
---------------
V1 headers store only [h0..h6, last, turn] -- the seven column HEIGHTS. The
V1 docstring asserts those "exactly reconstruct the masks". They do not.
tg_hdrproof.py found 1748 headers, in 30k sampled positions, that encode
two DIFFERENT boards, e.g. heights [1,0,1,1,1,1,0] last=3 turn=P2 matches
both:

    |X.OXOX.|      and      |O.XXXO.|

Same byte stream, different board, different winner. No model capacity can
recover what the encoder discarded. That is the measured 16% win-in-1.

THE FIX
-------
Replace each column's height byte with a single byte that also encodes WHICH
player owns each of the h cells:

    bits = sum(1 << r for r in range(h) if cell(r, c) == P1)   # bit r = row r
    byte = 0x80 | (1 << h) | bits

  * bit 7 (0x80) set  -> marks this as a column byte (no other slot uses 0x80)
  * bit h set          -> the height marker; h = that bit's index
  * bits 0..h-1       -> P1 owns row r iff bit r set; P2 owns the rest

Range is 0x81..0xFF and the map is a BIJECTION onto column states, so
decode_state_v2 reconstructs both masks exactly -- the property V1 only
claimed. A 6-stone column needs 1 height bit + 6 ownership bits = 7 bits, so
one byte per column is information-theoretically tight.

COST: zero extra bytes. PLY_LEN stays 10 (7 column + last + turn + move), so
the corpus size, context window, and stride arithmetic are all unchanged.

This module is deliberately SEPARATE from transcript.py: V1 is frozen so
existing corpus.txt still decodes and so the old format stays available as
the negative control in the A/B.
"""
from __future__ import annotations

from . import P1, P2, ROWS, COLS
from .board import Board

COL_BYTE_OFFSET = 0x41                # 'A'..'G'  (unchanged from V1)
N_COL_BYTES = 7
COL_BYTES = b"ABCDEFG"
NO_LAST = 0x00
TOK_TURN_P1 = 0x58                   # 'X'  (unchanged)
TOK_TURN_P2 = 0x4F                   # 'O'
PLY_LEN = 10                          # IDENTICAL to V1 -- see COST above
HEADER_LEN = PLY_LEN - 1              # 9
MOVE_OFFSET = 9                       # index of the move byte in a ply
MAX_PLIES = 42
MAX_LINE = MAX_PLIES * PLY_LEN + MAX_PLIES

COLUMN_FLAG = 0x80                    # marks a column byte
COL_SENTINEL = 0x00                   # not used by columns; reserved


def encode_column(board, c: int) -> int:
    """One byte = height marker + P1 ownership bits for column `c`."""
    h = board.heights[c]
    if not 0 <= h <= ROWS:
        raise ValueError(f"bad column height {h} for col {c}")
    bits = 0
    for r in range(h):
        if board.cell(r, c) == P1:
            bits |= 1 << r
    return COLUMN_FLAG | (1 << h) | bits


def decode_column(byte: int) -> tuple[int, list[int]]:
    """Inverse of encode_column -> (height, per-row owner list, bottom-up)."""
    if not byte & COLUMN_FLAG:
        raise ValueError(f"byte {byte:#04x} is not a V2 column byte")
    v = byte & ~COLUMN_FLAG
    h = v.bit_length() - 1
    if not 0 <= h <= ROWS:
        raise ValueError(f"decoded height {h} out of range (byte {byte:#04x})")
    owners = [P1 if (v >> r) & 1 else P2 for r in range(h)]
    return h, owners


def encode_state(board, last, side_to_move):
    """One 9-byte header [col0..col6, last, turn] for the position BEFORE
    the move. Occupancy AND ownership, unlike V1 which stored heights only.
    """
    out = bytearray(HEADER_LEN)
    for c in range(COLS):
        out[c] = encode_column(board, c)
    out[COLS] = NO_LAST if last is None else COL_BYTE_OFFSET + last
    out[COLS + 1] = TOK_TURN_P1 if side_to_move == P1 else TOK_TURN_P2
    return bytes(out)


def decode_state(header):
    """Inverse of encode_state -> (columns, last_col, side_to_move).

    `columns` is [(height, owners_bottom_up), ...] for all 7 columns, which
    is the information V1 threw away.
    """
    if len(header) != HEADER_LEN:
        raise ValueError(f"header must be {HEADER_LEN} bytes, got {len(header)}")
    columns = [decode_column(header[c]) for c in range(COLS)]
    last_byte = header[COLS]
    last = None if last_byte == NO_LAST else last_byte - COL_BYTE_OFFSET
    if last is not None and not (0 <= last < COLS):
        raise ValueError(f"bad last column {last}")
    turn = P1 if header[COLS + 1] == TOK_TURN_P1 else P2
    return columns, last, turn


def encode_game(moves):
    """Encode a full game: one newline-terminated record per ply.

    Identical framing to V1 (PLY_LEN payload + 0x0A newline per ply) so the
    newline stays the next-byte prediction target after every move byte.
    """
    if len(moves) > MAX_PLIES:
        raise ValueError(f"game too long: {len(moves)} plies (max {MAX_PLIES})")
    b = Board()
    side = P1
    last = None
    out = bytearray()
    for m in moves:
        out.extend(encode_state(b, last, side))
        out.append(COL_BYTE_OFFSET + m)
        out.append(0x0A)
        b.apply_move(m)
        last = m
        side = P2 if side == P1 else P1
    return bytes(out)


def encode_game_dense(moves):
    """Same as encode_game with the newlines stripped: stride-PLY_LEN."""
    return encode_game(moves).replace(b"\n", b"")


def decode_game(line):
    """Inverse of encode_game. Returns the move list.

    Validates ownership consistency as it goes: ownership must alternate with
    the turn, which is a much stronger check than V1's height-only replay.
    """
    if isinstance(line, str):
        line = line.encode("ascii")
    flat = bytes(line).replace(b"\n", b"")
    plies = len(flat) // PLY_LEN
    if plies * PLY_LEN != len(flat):
        raise ValueError(f"payload {len(flat)}B not a multiple of {PLY_LEN}")

    moves = []
    b = Board()
    side = P1
    for i in range(plies):
        rec = flat[i * PLY_LEN:(i + 1) * PLY_LEN]
        header, mbyte = rec[:HEADER_LEN], rec[-1]
        columns, last, turn = decode_state(header)
        if turn != side:
            raise ValueError(f"ply {i}: expected side {side}, got {turn}")

        # V2's real gain: verify the encoded ownership matches the replay.
        for c in range(COLS):
            h, owners = columns[c]
            if h != b.heights[c]:
                raise ValueError(
                    f"ply {i} col {c}: header height {h} != replay {b.heights[c]}")
            for r in range(h):
                if owners[r] != b.cell(r, c):
                    raise ValueError(
                        f"ply {i} col {c} row {r}: header says {owners[r]}, "
                        f"replay says {b.cell(r, c)}")

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


def moves_of_dense(line):
    """Inverse of encode_game_dense."""
    return decode_game(bytes(line))


def plies_of(line):
    """Ply count of a spaced (newline-terminated) game."""
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


__all__ = [
    "encode_column", "decode_column", "encode_state", "decode_state",
    "encode_game", "encode_game_dense", "decode_game", "moves_of_dense",
    "plies_of", "COL_BYTE_OFFSET", "COL_BYTES", "NO_LAST", "TOK_TURN_P1",
    "TOK_TURN_P2", "PLY_LEN", "HEADER_LEN", "MOVE_OFFSET", "MAX_PLIES",
    "MAX_LINE", "COLUMN_FLAG", "N_COL_BYTES",
]