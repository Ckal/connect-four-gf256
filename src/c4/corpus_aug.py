"""Append games to corpus.txt in the format the trainer's splitlines() reads.

Why this module exists
----------------------
`transcript.encode_game()` emits ONE NEWLINE PER PLY, interleaved:

    [h0..h6, last, turn, move] \\n [h0..h6, ...] \\n ...

That is the correct format for `decode_game()` -- the ply count IS the
newline count, and a training window can straddle ply boundaries.

But the trainer persists the corpus with

    f.write(b"\\n".join(games_to_lines(decided)) + b"\\n")

which joins ALREADY-interleaved sequences with a further newline. On read
back, `corpus_path.read_bytes().splitlines()` therefore yields one line per
PLY (10 bytes) rather than one line per GAME. `examples` is then a list of
single-ply records, and `build_batch` pads each 10-byte record up to
ctx+1 = 193 bytes -- so ~95% of every batch's supervised positions are
newline padding, and the model never sees more than one ply of board
context.

The fix keeps both worlds working:

  * GAME_DENSE  -- one game as one dense payload, plies concatenated with
    NO interleaved newlines, terminated by a single \\n. splitlines() yields
    the whole game as one training sequence. This is what we append.
  * GAME_SPACED -- the interleaved format, kept so decode_game() still
    round-trips any legacy bytes already on disk.

Every writer goes through `write_games()` so the on-disk layout has exactly
one definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .corpus import Game
from .transcript import HEADER_LEN, MOVE_OFFSET, PLY_LEN

# One game = concatenated 10-byte ply payloads, then a single b"\n".
# length == nplies * PLY_LEN, no interior newlines.
GAME_DENSE = "dense"
GAME_SPACED = "spaced"


def game_dense(moves: Sequence[int], version: str | None = None) -> bytes:
    """Encode one game into the dense single-line format."""
    from .board import Board, P1, P2
    from . import fmt

    mod = fmt.get(version)
    b = Board()
    side = P1
    last = None
    out = bytearray()
    for m in moves:
        out.extend(mod.encode_state(b, last, side))
        out.append(mod.COL_BYTE_OFFSET + m)
        b.apply_move(m)
        last = m
        side = P2 if side == P1 else P1
    return bytes(out)


def game_spaced(moves: Sequence[int], version: str | None = None) -> bytes:
    """Encode one game into the legacy interleaved-newline format."""
    from . import fmt

    return fmt.get(version).encode_game(moves)


def encode_games(games: Iterable[Game], layout: str = GAME_DENSE,
                 version: str | None = None) -> List[bytes]:
    if layout == GAME_DENSE:
        return [game_dense(moves, version) for moves, _w in games]
    return [game_spaced(moves, version) for moves, _w in games]


def write_games(path: str | Path,
                games: Iterable[Game],
                layout: str = GAME_DENSE,
                append: bool = True) -> int:
    """Append `games` to `path`. Returns bytes written.

    Never truncates. Existing content is preserved byte-for-byte.
    """
    lines = encode_games(games, layout)
    if not lines:
        return 0
    blob = b"\n".join(lines) + b"\n"
    mode = "ab" if append else "wb"
    with open(path, mode) as f:
        f.write(blob)
        f.flush()
    return len(blob)


def validate_file(path: str | Path) -> dict:
    """Structural check of a corpus file. Returns stats or raises ValueError."""
    p = Path(path)
    raw = p.read_bytes()
    lines = [ln for ln in raw.splitlines() if ln]
    good = bad = 0
    for ln in lines:
        if len(ln) % PLY_LEN == 0 and len(ln) >= PLY_LEN:
            good += 1
        else:
            bad += 1
    return {
        "bytes": len(raw),
        "lines": len(lines),
        "well_formed": good,
        "malformed": bad,
        "mean_bytes_per_line": (sum(len(x) for x in lines) / max(1, len(lines))),
        "plies_total": sum(len(x) // PLY_LEN for x in lines if len(x) % PLY_LEN == 0),
    }


def plies_of_dense(line: bytes) -> int:
    """Ply count of a dense (newline-free) game line."""
    if b"\n" in line:
        raise ValueError("dense line contains a newline")
    if len(line) % PLY_LEN:
        raise ValueError(f"{len(line)} bytes is not a multiple of {PLY_LEN}")
    return len(line) // PLY_LEN


def moves_of_dense(line: bytes) -> List[int]:
    """Recover the move list (column indices 0..6) from a dense line.

    The move slot holds COL_BYTE_OFFSET + col (0x41..0x47), so the offset
    must be subtracted -- the raw byte is not the column index.
    """
    from .transcript import COL_BYTE_OFFSET
    n = plies_of_dense(line)
    raw = [line[i * PLY_LEN + MOVE_OFFSET] for i in range(n)]
    for b in raw:
        if not COL_BYTE_OFFSET <= b <= COL_BYTE_OFFSET + 6:
            raise ValueError(f"bad move byte 0x{b:02x} in dense line")
    return [b - COL_BYTE_OFFSET for b in raw]


__all__ = [
    "GAME_DENSE", "GAME_SPACED",
    "game_dense", "game_spaced", "encode_games", "write_games",
    "validate_file", "plies_of_dense", "moves_of_dense",
]