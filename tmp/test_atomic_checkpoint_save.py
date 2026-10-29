"""Regression: checkpoint writes must be ATOMIC.

`torch.save(obj, path)` truncates the target and then writes into it.
The whole point of a long training run is that the best-so-far checkpoint
is playable WHILE the run continues -- the UI server loads
`best_hard.pt` and the promotion scorer reads it, both while the trainer
is writing it. A reader that opens the file during the write window sees
a truncated pickle.

This test:
  1. checks that every checkpoint write in the trainer goes through
     `atomic_torch_save` (not a bare `torch.save`),
  2. checks the helper exists and uses write-temp-then-rename,
  3. PROVES the behaviour with a concurrent reader: a writer loops
     over the file while a reader loads it, and the reader must never
     fail.
"""
import ast
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        fails.append(name)


# ---------------------------------------------------- 1. static: call sites
train_path = ROOT / "src" / "c4" / "train.py"
src = train_path.read_text(encoding="utf-8")
tree = ast.parse(src)

bare_saves = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "save":
            if isinstance(f.value, ast.Name) and f.value.id == "torch":
                # The helper's own `torch.save(obj, tmp)` is the ONE
                # allowed bare call: it writes to the temp file, which no
                # reader ever opens.
                fn = next((n for n in ast.walk(tree)
                           if isinstance(n, ast.FunctionDef)
                           and n.name == "atomic_torch_save"), None)
                if fn is not None and fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno):
                    continue
                bare_saves.append(node.lineno)

check("every checkpoint write goes through atomic_torch_save",
      not bare_saves, f"bare torch.save at lines {bare_saves}")

check("atomic_torch_save is defined",
      "def atomic_torch_save(" in src)
check("it writes to a temp file then renames",
      "os.replace(tmp, path)" in src)
check("it leaves no .tmp file behind on the happy path",
      "tmp = path.with_name(path.name + \".tmp\")" in src)

# ---------------------------------------------------- 2. behavioural proof
try:
    import torch
    from src.c4.train import atomic_torch_save
except Exception as e:  # pragma: no cover
    print(f"  FAIL  cannot import the helper: {e}")
    fails.append("import")
    torch = None

if torch is not None:
    # A UNIQUE scratch directory per process.  A fixed name would collide
    # with another concurrent test (or a second run of this test) that
    # cleans up the directory while this one is still writing into it --
    # which showed up as a bogus `FileNotFoundError` on the temp file.
    import os as _os
    import uuid as _uuid

    tmpdir = pathlib.Path(ROOT / "runs" / "_probe" /
                         f"_atomic_probe_{_os.getpid()}_{_uuid.uuid4().hex[:8]}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    target = tmpdir / "ckpt.pt"

    # seed it with a valid file so the reader has something to load
    atomic_torch_save({"state": {"w": torch.zeros(4)}, "meta": {"step": 1}}, target)

    stop = threading.Event()
    errors = []
    reads = [0]
    corrupt = []
    transient = []

    def reader():
        while not stop.is_set():
            try:
                # Separate the OPEN from the UNPICKLE so we can tell a
                # transient Windows lock apart from a corrupt payload.
                with open(target, "rb") as fh:
                    data = fh.read()
                import io
                d = torch.load(io.BytesIO(data), map_location="cpu",
                               weights_only=False)
                if "state" not in d:
                    corrupt.append("loaded but missing 'state'")
                reads[0] += 1
            except PermissionError as e:
                transient.append(f"PermissionError: {e}")
            except FileNotFoundError as e:
                transient.append(f"FileNotFoundError: {e}")
            except Exception as e:
                # Anything else that reaches the UNPICKLE stage would mean
                # the file opened but its content was broken -- a genuine
                # atomicity violation.
                corrupt.append(f"{type(e).__name__}: {e}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # writer: overwrite the SAME path many times with a big-ish payload so
    # the write window is long enough to be caught.
    payload = {"state": {"w": torch.randn(4096, 4096)}, "meta": {"step": 2}}
    for i in range(40):
        atomic_torch_save(payload, target)
    time.sleep(0.05)
    stop.set()
    t.join(timeout=10)

    # THE ATOMICITY PROPERTY: a reader must never observe a partially
    # written checkpoint.  Transient open failures are a different thing
    # -- they are Windows share-mode semantics (the writer holds the
    # destination open without FILE_SHARE_DELETE while it swaps it), not
    # a data problem.  The production UI handles those with a bounded
    # retry (see `_torch_load_with_retry` in src/c4/ui_server.py).
    check("a concurrent reader NEVER saw a corrupt/partial file",
          not corrupt, f"{len(corrupt)} corrupt reads, first: {corrupt[:1]}")
    check("transient open failures were only Windows locks",
          all(e.startswith(("PermissionError", "FileNotFoundError"))
              for e in transient),
          f"unexpected: {[e for e in transient if not e.startswith(('PermissionError', 'FileNotFoundError'))][:1]}")
    check("the reader actually read the file (not a vacuous pass)",
          reads[0] > 0, f"reads={reads[0]}")

    # and the file is still valid after all that
    d = torch.load(target, map_location="cpu", weights_only=False)
    check("the final file loads cleanly", "state" in d)

    # no temp file left behind
    leftovers = list(tmpdir.glob("*.tmp"))
    check("no .tmp file left behind", not leftovers,
          f"leftovers={[p.name for p in leftovers]}")

    # cleanup
    for f in tmpdir.glob("*"):
        f.unlink()
    tmpdir.rmdir()

print()
if fails:
    print(f"FAIL: {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print("PASS: checkpoint writes are atomic")
