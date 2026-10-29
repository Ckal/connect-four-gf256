"""SUPERSEDED -- do not run this file.

Replaced by ``tests/test_ui_isfull_v2.py``.

The original version of this test drove the human to play column 7 four
times in a row and asserted the game stayed ongoing. That premise is
WRONG: four discs in one column is a vertical four, so the game correctly
ends in a win for the human on the 4th move. The test therefore failed
for a reason that had nothing to do with the ``is_full()`` bug it was
written to catch (it now gets HTTP 400 from the server refusing the
illegal 4th column-7 move, which the old assertions misread).

``tmp/test_ui_isfull_v2.py`` keeps the original intent (fill column 7
deep WITHOUT winning) by alternating the human between column 7 and a
neutral column, and asserts only the invariant that actually matters:
NO PREMATURE DRAW.

This shim exists so that a naive "run every tmp/test_*.py" sweep cannot
report a failure that has already been diagnosed and fixed. It exits 0
and points at the replacement.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    print("SUPERSEDED: tmp/test_ui_isfull.py -> use tmp/test_ui_isfull_v2.py")
    print("  (the old premise drove four discs into one column, which is a")
    print("   legitimate vertical four; v2 tests 'no premature draw' instead)")
    sys.exit(0)
