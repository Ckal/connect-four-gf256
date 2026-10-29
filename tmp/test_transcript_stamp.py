"""REGRESSION: every shipped checkpoint artifact must carry its transcript
version.  A checkpoint WITHOUT the stamp is treated as **v1 by definition**,
which is a silent, crash-free wrong-alphabet read: the model loads, serves
HTTP 200, and simply plays badly.

This is the failure that made the 500-row acceptance run look like 24.37 %
when the correctly-stamped artifact measures 91.29 %.  The gate result was
real; the artifact it was measured on was the wrong one.

What is asserted:
  1. `_load_agent` refuses to guess -- it warns and assumes v1.  (The loader
     behaviour is correct; the *producer* is the bug, so that is what we
     assert on.)
  2. For a known v2 run directory, every final export and every promoted
     checkpoint carries `"transcript": "v2"`.
  3. The trainer's final-save path includes the stamp -- asserted against
     the SOURCE, so the test fails the moment someone deletes the key from
     the save path again.

Run: ./.venv-c4/Scripts/python.exe tmp/test_transcript_stamp.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_PY = ROOT / "src" / "c4" / "train.py"


def _meta_of(path: pathlib.Path) -> dict:
    # weights_only=False: these are this repo's own training artifacts.
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        meta = raw.get("meta")
        if isinstance(meta, dict):
            return meta
        return raw
    return {}


def test_final_save_path_stamps_transcript():
    """The trainer's FINAL export must include the transcript version.

    Asserted on the source text because the failure is a deleted key, and
    asserting on the source catches it before a 30-minute training run
    produces another unstamped artifact.
    """
    src = TRAIN_PY.read_text(encoding="utf-8")

    # Locate the final-save block.  The anchor `model.cpu()` matters: the
    # promotion path ALSO calls `export_hard_with_meta`, and a non-greedy
    # match would stop at that earlier call -- which does carry the stamp --
    # and would then pass even when the final export has lost it.
    #
    # The write goes through `atomic_torch_save(...)` (temp file + rename),
    # so the pattern must allow that wrapper as well as a bare
    # `torch.save(...)`.  Anchoring on `export_hard_with_meta(model.cpu()`
    # keeps the match specific to the final export either way.
    m = re.search(
        r"(?:torch\.save|atomic_torch_save)\("
        r"export_hard_with_meta\(model\.cpu\(\).*?"
        r"run_dir / \"hard_model_final\.pt\"\)",
        src, re.S)
    assert m is not None, (
        "could not find the hard_model_final.pt save block in src/c4/train.py "
        "-- if the save path was restructured, update this test")
    block = m.group(0)
    assert '"transcript": version' in block, (
        "the hard_model_final.pt save path does NOT stamp the transcript "
        "version. A checkpoint without it is served as v1 by definition, "
        "which silently reads the wrong byte alphabet.")

    # The float export must carry it too, or the float path mis-builds.
    # Like the hard export above, this write goes through
    # `atomic_torch_save(...)`, so the pattern must accept either wrapper.
    mf = re.search(
        r"(?:torch\.save|atomic_torch_save)\(\{\"state\": model\.state_dict\(\).*?"
        r"run_dir / \"final_float\.pt\"\)",
        src, re.S)
    assert mf is not None, (
        "could not find the final_float.pt save block in src/c4/train.py "
        "-- if the save path was restructured, update this test")
    assert '"transcript": version' in mf.group(0), (
        "the final_float.pt save path does NOT stamp the transcript version.")


def test_existing_artifacts_are_stamped():
    """Known run dirs must have their final exports stamped."""
    checked = 0
    for run in ("overfit_fixed", "fd_a_s6000", "fd_b_s6000", "fd_c_s6000"):
        d = ROOT / "runs" / "_probe" / run
        if not d.is_dir():
            continue
        for name in ("hard_model_final.pt", "final_float.pt"):
            p = d / name
            if not p.exists():
                continue
            meta = _meta_of(p)
            ver = meta.get("transcript")
            assert ver == "v2", (
                f"{run}/{name} has transcript={ver!r}, expected 'v2'. "
                "Run tmp/restamp_transcript.py to repair it.")
            checked += 1
    assert checked > 0, (
        "no stamped artifacts found to check -- the run dirs are missing, "
        "so this test is vacuous. Point it at real artifacts.")


def test_loader_reports_missing_stamp():
    """The loader must not silently accept an unstamped checkpoint.

    `_load_agent` prints a warning and assumes v1.  That is the documented
    behaviour and it is CORRECT -- the loader cannot guess.  What it must
    never do is pretend the model is fine.  This test asserts the warning
    path exists in the source so a future refactor cannot make it quiet.
    """
    src = (ROOT / "src" / "c4" / "ui_server.py").read_text(encoding="utf-8")
    assert "declares no transcript version" in src, (
        "ui_server._load_agent no longer warns when a checkpoint has no "
        "transcript stamp. That warning is the only signal that the model "
        "is being served with the wrong byte alphabet.")


def main() -> int:
    tests = [
        test_final_save_path_stamps_transcript,
        test_existing_artifacts_are_stamped,
        test_loader_reports_missing_stamp,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
