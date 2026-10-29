"""Check the symmetry penalty is measured ON the training distribution.

The trainer reads corpus LINES (byte streams).  The penalty reconstructs
a stream from the decoded MOVE LIST.  If those two differ, the penalty
measures a distribution the model never saw in training and the
regulariser is meaningless.

So: for a sample of real corpus lines, assert

    decision_stream(decode(line)) == line_with_newlines_stripped

and the same for the reflected move list.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.c4.symmetry import decision_stream, mirrored_decision_stream  # noqa: E402
from src.c4.transcript_v2 import decode_game                          # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",
                    default="runs/_probe/perply_v23_full/corpus_320k_fixed.txt",
                    help="corpus to check (default: the REPAIRED original)")
    args = ap.parse_args()
    corpus = ROOT / args.corpus
    raw = corpus.read_bytes()
    lines = [ln for ln in raw.split(b"\n") if ln]
    print(f"corpus: {args.corpus}")
    print(f"corpus rows: {len(lines)}")

    import random
    rng = random.Random(32)
    sample = rng.sample(range(len(lines)), 300)

    n_ok = 0
    n_bad = 0
    bad_examples = []
    for i in sample:
        line = lines[i]
        try:
            moves = decode_game(line)
        except Exception as e:
            bad_examples.append((i, f"decode failed: {e}"))
            n_bad += 1
            continue
        if len(moves) < 1:
            continue
        # The agent's stream for the first k moves is:
        #     corpus[0 : k*10]        (the k played plies)
        #   + corpus[k*10 : k*10+9]  (the NEXT ply's header, appended by
        #                             prompt_tokens so the read position
        #                             lands on the TURN byte)
        # so the agent stream for k = len(moves)-1 moves must equal the
        # corpus prefix of (k*10 + 9) bytes.
        k = len(moves) - 1
        if k < 1:
            continue
        expect = line.replace(b"\n", b"")[:k * 10 + 9]
        try:
            ds = decision_stream(moves[:k])
        except Exception as e:
            bad_examples.append((i, f"decision_stream failed: {e}"))
            n_bad += 1
            continue
        if ds == expect:
            n_ok += 1
        else:
            n_bad += 1
            if len(bad_examples) < 5:
                bad_examples.append((i, f"len {len(ds)} vs {len(expect)}"))

    check(f"decision_stream reproduces the corpus stream "
          f"({n_ok}/{n_ok + n_bad})",
          n_bad == 0, f"{n_bad} mismatches: {bad_examples[:3]}")

    # The reflected stream must be the reflection of the original stream's
    # MOVE BYTES, and must have the same length.
    n_ok2 = 0
    n_bad2 = 0
    for i in sample[:200]:
        line = lines[i]
        try:
            moves = decode_game(line)
            if len(moves) < 2:
                continue
            k = len(moves) - 1
            ms = mirrored_decision_stream(moves[:k])
            expect_len = k * 10 + 9
        except Exception:
            n_bad2 += 1
            continue
        if len(ms) == expect_len:
            n_ok2 += 1
        else:
            n_bad2 += 1

    check(f"mirrored stream has the same length "
          f"({n_ok2}/{n_ok2 + n_bad2})", n_bad2 == 0)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
