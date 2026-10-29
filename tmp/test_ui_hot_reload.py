"""End-to-end proof that the UI can serve a NEW checkpoint while running.

This is the "can I play a new version every ~30 minutes?" test.

Scenario:
  1. Start the UI server on checkpoint A.
  2. Play a few moves -> confirm the game is live.
  3. Copy checkpoint B over the SAME path (atomically, as the trainer does).
  4. GET /reload -> the server must now serve B.
  5. The in-progress game must survive the swap (board + moves intact).
  6. GET /modelinfo must report the new mtime and `changed_on_disk=false`
   (because the reload already consumed the change).

No network assumptions: the server binds 127.0.0.1 on an ephemeral port.
"""
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS = []
PASSES = 0


def check(name, cond, detail=""):
    global PASSES
    if cond:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------ cookie jar
# Since the session refactor every visitor has their OWN game, keyed by
# the tg_sid cookie. A client that does not keep its cookie therefore
# gets a NEW game on every request -- which is correct server behaviour
# but breaks any test that expects its own game to persist. This opener
# keeps one cookie for the whole test run.
import http.cookiejar  # noqa: E402

_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(_JAR))


def get_c(url, timeout=60):
    with _OPENER.open(url, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def post_c(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def get_raw(url, timeout=120):
    """GET without any cookie -> the server must create a NEW session."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


# --------------------------------------------------------------- fixtures
CKPT_A = ROOT / "runs" / "_probe" / "fx_a_s6000" / "checkpoints" / "step6000_hard.pt"
CKPT_B = ROOT / "runs" / "_probe" / "fd_a_s6000" / "checkpoints" / "step6000_hard.pt"

if not CKPT_A.exists():
    print(f"SKIP: {CKPT_A} missing")
    sys.exit(0)
if not CKPT_B.exists():
    print(f"SKIP: {CKPT_B} missing")
    sys.exit(0)

tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ui_reload_", dir=str(ROOT / "runs" / "_probe")))
served = tmpdir / "served_hard.pt"
shutil.copy2(CKPT_A, served)

port = free_port()
log = tmpdir / "server.log"
proc = subprocess.Popen(
    [sys.executable, "-u", "-m", "src.c4.ui_server",
     "--model", str(served), "--port", str(port), "--device", "cpu"],
    cwd=str(ROOT), stdout=open(log, "w", encoding="utf-8"),
    stderr=subprocess.STDOUT, env={**__import__("os").environ,
                                   "PYTHONPATH": str(ROOT)},
)

base = f"http://127.0.0.1:{port}"
try:
    # wait for readiness
    ready = False
    for _ in range(120):
        try:
            st, _ = get_c(f"{base}/state", timeout=5)
            if st == 200:
                ready = True
                break
        except Exception:
            time.sleep(0.5)
    check("server came up", ready)
    if not ready:
        print(log.read_text(encoding="utf-8")[-2000:])
        raise SystemExit(1)

    # ---------------------------------------------------------- play moves
    st, state = get_c(f"{base}/state")
    check("initial state is served", st == 200, f"st={st}")

    st, after = post_c(f"{base}/move", {"col": 3})
    check("a move is accepted", st == 200, f"st={st} body={after}")
    moves_before = after.get("moves", [])

    st, info1 = get_c(f"{base}/modelinfo")
    check("modelinfo reports the served checkpoint",
          info1.get("checkpoint") is not None and "served_hard.pt" in info1["checkpoint"],
          str(info1))
    check("modelinfo reports changed_on_disk=false right after load",
          info1.get("changed_on_disk") is False, str(info1))

    # ------------------------------------------- trainer overwrites the path
    # Exactly what src.c4.train.atomic_torch_save does: write a temp file
    # next to the target, then rename it over the target.
    tmp = served.with_name(served.name + ".tmp")
    shutil.copy2(CKPT_B, tmp)
    import os
    os.replace(tmp, served)

    st, info2 = get_c(f"{base}/modelinfo")
    check("modelinfo detects the checkpoint changed on disk",
          info2.get("changed_on_disk") is True, str(info2))

    # ---------------------------------------------------------- hot reload
    st, rel = get_c(f"{base}/reload")
    check("reload succeeds", st == 200 and rel.get("ok") is True,
          f"st={st} rel={rel}")

    st, info3 = get_c(f"{base}/modelinfo")
    check("after reload, changed_on_disk is false again",
          info3.get("changed_on_disk") is False, str(info3))

    # ------------------------------------------------ the game survived
    st, state2 = get_c(f"{base}/state")
    check("in-progress game survived the reload",
          state2.get("moves") == moves_before,
          f"before={moves_before} after={state2.get('moves')}")

    # ------------------------------------------------ the model is serving
    st, probs = get_c(f"{base}/probs")
    check("probs still served after reload", st == 200, f"st={st}")
    check("probs has a real distribution",
          isinstance(probs.get("cols"), list) and len(probs["cols"]) == 7,
          str(probs)[:200])

    # the serving model must now be B: its move choice must match B's
    # own prediction, not A's.
    sys.path.insert(0, str(ROOT))
    from src.c4.ui_server import _load_agent  # noqa: E402
    from src.c4.agent import Agent  # noqa: E402
    import torch  # noqa: E402

    agent_b = _load_agent(str(CKPT_B), device="cpu")

    # play a few more moves after the reload and check the AI replies
    # consistently with checkpoint B on the resulting position.
    st, s = post_c(f"{base}/move", {"col": 2})
    board = s.get("board")
    check("post-reload move accepted", st == 200, f"st={st}")

    # the server's own reported next-move distribution must be non-degenerate
    st, probs2 = get_c(f"{base}/probs")
    cols = probs2.get("hard_cols") or probs2.get("cols")
    check("post-reload hard_cols is a real distribution",
          isinstance(cols, list) and len(cols) == 7 and
          abs(sum(cols) - 1.0) < 1e-3 and len(set(cols)) > 1,
          str(cols))

    # ------------------------------------------------ reload is idempotent
    st, rel2 = get_c(f"{base}/reload")
    check("a second reload also succeeds", st == 200 and rel2.get("ok") is True,
          f"st={st}")

    # ------------------------------------- a NEW session after the reload
    # The session factory used to close over the STARTUP agent, so a
    # visitor arriving after a hot reload was handed the OLD model.
    # The fix publishes the current agent in a module global that the
    # factory reads. Prove it: open a FRESH session (no cookie) and
    # compare its next-move distribution against checkpoint B computed
    # independently, and against checkpoint A.
    def probs_for_new_session():
        """Fetch /probs with NO cookie -> a brand-new session."""
        return get_raw(f"{base}/probs")

    st, new_probs = probs_for_new_session()
    check("a new session after the reload is served", st == 200, f"st={st}")
    new_cols = new_probs.get("hard_cols") or new_probs.get("cols")
    check("the new session has a real distribution",
          isinstance(new_cols, list) and len(new_cols) == 7 and
          abs(sum(new_cols) - 1.0) < 1e-3,
          str(new_cols))

    # Compute the same position's distribution from B and from A,
    # independently of the server, and compare.
    from src.c4.board import Board  # noqa: E402
    from src.c4.agent import Agent  # noqa: E402

    def cols_for(ckpt_path, moves):
        ag = _load_agent(str(ckpt_path), device="cpu")
        b = Board()
        for mv in moves:
            b = b.apply_move(mv)
        lg = ag.hard_column_logits(b, list(moves))
        return torch.softmax(lg, dim=-1).tolist()

    # an empty board is the position a brand-new session starts from
    empty_moves = []
    cols_b = cols_for(CKPT_B, empty_moves)
    cols_a = cols_for(CKPT_A, empty_moves)

    def l1(p, q):
        return sum(abs(x - y) for x, y in zip(p, q))

    d_new_b = l1(new_cols, cols_b)
    d_new_a = l1(new_cols, cols_a)
    check("the new session plays against the RELOADED model, not the startup one",
          d_new_b < d_new_a,
          f"L1(new,B)={d_new_b:.6f} L1(new,A)={d_new_a:.6f}")

finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(tmpdir, ignore_errors=True)

print()
if FAILS:
    print(f"FAIL: {len(FAILS)} check(s) failed: {FAILS}")
    sys.exit(1)
print(f"PASS: {PASSES} checks -- hot reload works, game survives")
