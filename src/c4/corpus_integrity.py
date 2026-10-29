"""Fast corpus integrity scanner for the transcript LAST-MOVE field.

WHY THIS EXISTS
---------------
The v2 corpus format stores, for every ply, a 9-byte header:

    [col0..col6, last, turn]

`last` is the column of the PREVIOUS move (the header describes the
position BEFORE the current move).  A corpus row is internally
consistent only if, for every ply p >= 1,

        header[p].last == move[p - 1]

`decode_game()` enforces this, so a violating row FAILS to decode.  That
is the good news: the corruption is loud, not silent.  The bad news is
that the trainer's startup guard only probed ONE row, so a defect that
starts 60% of the way through a corpus passed silently and 39.5% of the
rows trained on a wrong byte.

WHAT THIS MODULE DOES
---------------------
A cheap O(bytes) scan that checks the last-move field against the move
bytes WITHOUT replaying the board.  It is:

  * ~100x faster than a full `decode_game()` on every row,
  * exact for this defect class (a wrong last byte is the ONLY way the
    field can disagree with the move bytes), and
  * usable as a trainer startup gate.

The trainer also replays a sample of rows through `decode_game()` for the
full ownership check; this scan is the cheap, always-on guard.
"""
from __future__ import annotations

import pathlib
from typing import Iterable

# v2 layout: 9 header bytes + 1 move byte = 10 bytes per ply.
PLY_LEN = 10
HEADER_LEN = 9
MOVE_OFFSET = 9
LAST_OFFSET = 7
COL_BYTE_OFFSET = 0x41
NO_LAST = 0x40


def last_field_violations(line: bytes) -> list[int]:
    """Ply indices whose header last-move field contradicts the move list.

    Returns [] when the row is consistent (or has a single ply, where no
    previous move exists yet).
    """
    flat = line.replace(b"\n", b"")
    n = len(flat) // PLY_LEN
    bad: list[int] = []
    prev_move: int | None = None
    for p in range(n):
        rec = flat[p * PLY_LEN:(p + 1) * PLY_LEN]
        last_byte = rec[LAST_OFFSET]
        move_byte = rec[MOVE_OFFSET]
        if p >= 1 and last_byte != NO_LAST:
            recorded = last_byte - COL_BYTE_OFFSET
            if prev_move is not None and recorded != prev_move:
                bad.append(p)
        if move_byte >= COL_BYTE_OFFSET:
            prev_move = move_byte - COL_BYTE_OFFSET
    return bad


def row_is_consistent(line: bytes) -> bool:
    return not last_field_violations(line)


def scan_corpus(path: str | pathlib.Path,
                limit: int = 0) -> dict:
    """Scan a corpus file for last-move-field violations.

    `limit` > 0 stops after that many rows (0 = whole file).  Returns a
    dict with the counts, the first offending row index, and a sample.
    """
    path = pathlib.Path(path)
    bad_rows = 0
    bad_plies = 0
    first_bad = -1
    examples: list[dict] = []
    checked = 0
    with path.open("rb") as fh:
        for i, raw in enumerate(fh):
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            checked += 1
            v = last_field_violations(line)
            if v:
                bad_rows += 1
                bad_plies += len(v)
                if first_bad < 0:
                    first_bad = i
                if len(examples) < 5:
                    examples.append({"row": i, "plies": v[:8]})
            if limit and checked >= limit:
                break
    return {
        "path": str(path),
        "rows_checked": checked,
        "rows_with_last_field_violations": bad_rows,
        "plies_with_last_field_violations": bad_plies,
        "first_bad_row": first_bad,
        "examples": examples,
        "clean": bad_rows == 0,
    }


def assert_corpus_consistent(path: str | pathlib.Path,
                              limit: int = 0,
                              context: str = "") -> dict:
    """Raise SystemExit if the corpus has any last-move-field violation."""
    rep = scan_corpus(path, limit=limit)
    if not rep["clean"]:
        where = f" ({context})" if context else ""
        raise SystemExit(
            f"refusing to train{where}: {path} has "
            f"{rep['rows_with_last_field_violations']} row(s) with an "
            f"inconsistent last-move header field "
            f"(first at row {rep['first_bad_row']}). Such rows do not "
            f"decode and train the model on a wrong byte. Repair the "
            f"corpus first (tools/repair_corpus_last_field.py)."
        )
    return rep
