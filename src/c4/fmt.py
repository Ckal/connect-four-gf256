"""Single source of truth for WHICH transcript format a component speaks.

Why a resolver instead of swapping imports
------------------------------------------
V1 (`transcript.py`) and V2 (`transcript_v2.py`) are byte-INCOMPATIBLE: V2's
header column bytes are 0x81..0xFF, V1's are 0x30..0x36. A model trained on
one and fed the other is not "slightly worse" -- it is reading pure noise, and
it will fail silently, because every byte is still a legal byte.

So the version must travel WITH the weights. The checkpoint's `meta` records
`transcript`, the loader reads it, and the Agent encodes in exactly that
format. Nothing infers the version from the bytes themselves.

  * `get("v2") -> transcript_v2 module`
  * `normalize(None) -> DEFAULT_VERSION`
  * `resolve(ref) -> (version, module)` accepting a str or a module.

Two invariants that hold for BOTH formats and must never be broken:

  1. PLY_LEN == 10. The move byte sits at offset MOVE_OFFSET (9) of each ply.
  2. The move byte's VALUE range is COL_BYTE_OFFSET..COL_BYTE_OFFSET+6
     (0x41..0x47) and this range must stay disjoint from every header byte,
     because COL_SLICE is used to argmax the column logits. V2 satisfies this:
     its columns are 0x81..0xFF, its `last` byte is 0x00 or 0x41..0x47.
     (Note `last` deliberately still overlaps the move range -- see
     positional_move_mask() for why value tests cannot find the move slot.)
"""
from __future__ import annotations

import importlib

DEFAULT_VERSION = "v2"

_REGISTRY = {
    "v1": "src.c4.transcript",
    "v2": "src.c4.transcript_v2",
}

# PLY_LEN is IDENTICAL (10) in both formats, and so are MOVE_OFFSET, COL_BYTE_OFFSET
# and COL_SLICE: v2 only changed what the 7 HEADER column bytes mean. Callers that do
# pure stride arithmetic (corpus windowing, move-mask positions, tag slab lengths) can
# import these from here and stay format-agnostic. Anything that encodes or decodes a
# board MUST go through get()/resolve() instead.
PLY_LEN = 10
MOVE_OFFSET = 9
COL_BYTE_OFFSET = 0x41
COL_SLICE = slice(COL_BYTE_OFFSET, COL_BYTE_OFFSET + 7)

# V2-only: the flag bit that distinguishes a v2 column byte (0x80 | height |
# ownership bits) from every other byte. V1 has no equivalent, so this is None
# for v1 -- which is exactly why legality-at-a-mask is a v2-only capability.
COLUMN_FLAG = 0x80


def normalize(version: str | None) -> str:
    """Accept None / 'V2' / 'transcript_v2' and return the canonical 'v2'."""
    if version is None:
        return DEFAULT_VERSION
    v = str(version).strip().lower()
    if v.startswith("src.c4."):
        v = v.rsplit(".", 1)[-1]
    if v.startswith("transcript"):
        v = v[len("transcript"):]
    if v in ("", "_v2", "2"):
        v = "v2"
    if v not in _REGISTRY:
        raise ValueError(
            f"unknown transcript version {version!r}; expected one of "
            f"{sorted(_REGISTRY)}")
    return v


def get(version: str | None = None):
    """Return the transcript module for `version` (default DEFAULT_VERSION)."""
    return importlib.import_module(_REGISTRY[normalize(version)])


def resolve(ref):
    """(version, module) from either a version string or a module object."""
    if hasattr(ref, "encode_game"):
        name = getattr(ref, "__name__", "")
        for v, path in _REGISTRY.items():
            if name == path or name.endswith(path.rsplit(".", 1)[-1]):
                return v, ref
        raise ValueError(f"unrecognised transcript module {name}")
    v = normalize(ref)
    return v, get(v)


def same_format(other: str | None) -> bool:
    """True when `other` is the default format this tree speaks."""
    return normalize(other) == DEFAULT_VERSION


__all__ = ["DEFAULT_VERSION", "normalize", "get", "resolve", "same_format",
           "PLY_LEN", "MOVE_OFFSET", "COL_BYTE_OFFSET", "COL_SLICE",
           "COLUMN_FLAG"]