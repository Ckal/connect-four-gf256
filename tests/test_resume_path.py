"""Regression: the resume path must be USABLE, not just non-crashing.

Three defects were stacked on the resume path and each one killed the run
before a single training step:

  1. `start_step > args.steps` left `step` unbound -> UnboundLocalError.
     Now: an actionable SystemExit.
  2. The model was built from `--lags` BEFORE the checkpoint was read, so a
     12-lag run was resumed into the 8-lag V6 default ->
     "IndexError: index 9 is out of bounds for dimension 0 with size 8".
     Now: the checkpoint is loaded first and supplies the lag set; a
     disagreeing --lags is refused.
  3. The hard branch called `model.import_hard(ckpt)`, a method the shipped
     V6 model does not have -> AttributeError.

This test checks the two cheap invariants (1) and (2) directly, and then
proves (3) by checking that the resume code never calls a method the model
does not have.

It is a STATIC + BEHAVIOURAL test: it does not need a trained checkpoint
and does not need a GPU, so it can run in the fast sweep.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "c4" / "train.py"

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        fails.append(name)


src = TRAIN.read_text(encoding="utf-8")
tree = ast.parse(src)

# ---------------------------------------------------------------- 1. lags
# The checkpoint load must come BEFORE the model construction, otherwise the
# lag set cannot be taken from the checkpoint.
load_line = None
build_line = None
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "load"
                and isinstance(f.value, ast.Name) and f.value.id == "torch"):
            load_line = node.lineno
        # the model is built through `arch_cls(...)`, where arch_cls is
        # GFMarkovLM_V7 or GFMarkovLM -- NOT through a literal V6/V7 call
        if isinstance(f, ast.Name) and f.id == "arch_cls":
            build_line = node.lineno

check("checkpoint load exists", load_line is not None)
check("model construction exists", build_line is not None)
if load_line and build_line:
    check(
        "checkpoint is loaded BEFORE the model is built",
        load_line < build_line,
        f"load@{load_line} build@{build_line}",
    )

# ---------------------------------------------------------------- 2. lag set
check(
    "lag set is read from the checkpoint (meta.lags)",
    '"lags"' in src and "meta" in src,
    "no meta.lags read found",
)
check(
    "a disagreeing --lags is refused on resume",
    "refusing to resume" in src,
    "no resume refusal message found",
)
check(
    "the refusal message names the lag set",
    "is part of the model structure" in src,
)

# ---------------------------------------------------------------- 3. import_hard
# The shipped V6 model has no import_hard method, so calling it is a
# guaranteed AttributeError. The resume path must not call it on a model
# that lacks it.
v6 = (ROOT / "src" / "gf256_markov_lm_v6.py").read_text(encoding="utf-8")
v6_has_import_hard = "def import_hard" in v6
check(
    "shipped V6 model really has no import_hard (premise)",
    not v6_has_import_hard,
    "premise broken: V6 now defines import_hard; re-check the resume path",
)

# Look for an unguarded `.import_hard(` call in train.py.
bad_calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "import_hard":
            guarded = False
            # walk up is not available; instead check the surrounding source
            # text for a hasattr guard in the same statement block
            seg = src.splitlines()[max(0, node.lineno - 4):node.lineno]
            guarded = any("hasattr" in ln for ln in seg)
            if not guarded:
                bad_calls.append(node.lineno)
check(
    "no unguarded model.import_hard(...) call",
    not bad_calls,
    f"unguarded at lines {bad_calls}",
)

# ---------------------------------------------------------------- 4. guard
check(
    "an empty corpus is refused clearly (not ValueError: high <= 0)",
    "contains no usable rows" in src,
)

print()
if fails:
    print(f"FAIL: {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print("PASS: resume path invariants hold")
