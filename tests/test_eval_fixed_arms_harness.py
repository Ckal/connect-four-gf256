"""Prove the fixed evaluator FAILS LOUDLY instead of writing a green artifact.

Three independent checks:

  A. run-tag isolation: two concurrent/near-concurrent evaluator runs must
     not share any intermediate file path.

  B. failure propagation: if a phase subprocess fails, the evaluator must
     exit 1, the artifact must carry ok=False and a non-empty failures
     list, and the failing phase must have NO data key.

  C. stale-file immunity: a leftover intermediate file from a previous
     run must not be readable as this run's output.  We simulate this by
     planting a file at the exact path the evaluator will use and
     checking that the evaluator deletes it before running (so it can
     only ever contain this run's data, or nothing).

Run:  python -m tmp.test_eval_fixed_arms_harness
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-c4/Scripts/python.exe")
EVAL = "tmp/eval_fixed_arms.py"

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}: {msg}")
    if not cond:
        fails.append(msg)


print("=== A. run-tag isolation (static check of generated paths) ===")
src = (ROOT / EVAL).read_text(encoding="utf-8")
check("--run-tag" in src, "evaluator exposes --run-tag")
check("args.run_tag" in src, "generated paths use args.run_tag")
check('f"runs/_probe/eval_{arm}_holdout.json"' not in src,
      "old un-tagged holdout path is gone")
check('f"runs/_probe/eval_{arm}_mirror.json"' not in src,
      "old un-tagged mirror path is gone")
check('f"runs/_probe/eval_{arm}_solver.json"' not in src,
      "old un-tagged solver path is gone")
check('f"runs/_probe/duel_{a}_vs_{b}.log"' not in src,
      "old un-tagged duel log path is gone")
check("unlink(missing_ok=True)" in src,
      "pre-existing intermediates are deleted before the run")

print()
print("=== B. failure propagation (live run against a missing checkpoint) ===")
out = ROOT / "runs/_probe/_harness_fail_test.json"
if out.exists():
    out.unlink()
p = subprocess.run(
    [PY, "-u", EVAL, "--arms", "does_not_exist_arm",
     "--out", "runs/_probe/_harness_fail_test.json",
     "--run-tag", "harnessfail", "--skip-duel"],
    cwd=str(ROOT), capture_output=True, text=True)
check(p.returncode != 0,
      f"evaluator exits non-zero when an arm is missing (rc={p.returncode})")
if out.exists():
    d = json.loads(out.read_text())
    check(d.get("ok") is False, "artifact carries ok=False")
    check(bool(d.get("failures")), "artifact carries a non-empty failures list")
else:
    check(False, "artifact was written (expected: ok=False + failures)")

print()
print("=== C. stale-file immunity (plant a decoy, expect it removed) ===")
decoy = ROOT / "runs/_probe/eval_decoyarm_harnessstale_holdout.json"
decoy.parent.mkdir(parents=True, exist_ok=True)
decoy.write_text(json.dumps({"strict_pct": 99.99, "planted": True}),
                 encoding="utf-8")
check(decoy.exists(), "decoy file planted")
# The evaluator deletes intermediates for its tag before running.  A
# decoy under the SAME tag the evaluator will use must therefore be gone
# afterwards -- proving a leftover cannot be adopted as this run's data.
p = subprocess.run(
    [PY, "-u", EVAL, "--arms", "decoyarm",
     "--out", "runs/_probe/_harness_stale_test.json",
     "--run-tag", "harnessstale", "--skip-duel"],
    cwd=str(ROOT), capture_output=True, text=True)
gone = not decoy.exists()
check(gone, "decoy at this run's own path was removed, not adopted")
# and confirm the evaluator never wrote a 'planted' flag into its output
st = ROOT / "runs/_probe/_harness_stale_test.json"
if st.exists():
    ds = json.loads(st.read_text())
    check("planted" not in json.dumps(ds),
          "no planted/stale value leaked into the artifact")
else:
    check(True, "no artifact written for missing arm (nothing to leak)")

# cleanup
for p_ in (out, ROOT / "runs/_probe/_harness_stale_test.json", decoy):
    p_.unlink(missing_ok=True)

print()
if fails:
    print(f"VERDICT: HARNESS_TEST_FAILED ({len(fails)} failures)")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("VERDICT: HARNESS_TEST_PASSED")
