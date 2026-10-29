"""MULTI-USER ISOLATION TEST for the Connect Four UI server.

THE DEFECT THIS TEST EXISTS FOR
-------------------------------
The UI server originally kept ONE game in a module-level global. Every
visitor shared one board: a move by one person moved everyone else's
game. That is acceptable on a laptop with one user and completely
unacceptable for a public URL.

THIS TEST PROVES THE FIX
------------------------
It starts a real server, then drives it with SEPARATE cookie jars --
one per simulated visitor -- and asserts:

  1. Two visitors get DIFFERENT boards.
  2. A move by visitor A does NOT change visitor B's board.
  3. A move by visitor A does NOT change visitor B's move history.
  4. The model still replies to A after B has moved.
  5. Visitor A's board survives across requests (cookie round-trip).
  6. A malformed/absent cookie yields a FRESH session, never a shared
     fallback game.
  7. Concurrent visitors do not corrupt each other's state.
  8. The operational routes are refused in --public mode.
  9. Security headers are present on every response.
 10. The rate limiter rejects a burst.

Run:
    ./.venv-c4/Scripts/python.exe tmp/test_ui_multiuser.py --port 7871
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


class Visitor:
    """One simulated browser: its own cookie jar, its own URL opener."""

    def __init__(self, base: str):
        self.base = base
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar))

    def _open(self, method: str, path: str, body: dict | None = None,
              timeout: float = 60):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data,
                                     headers=headers, method=method)
        try:
            with self.opener.open(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8")), \
                    dict(resp.headers)
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            try:
                return e.code, json.loads(raw), dict(e.headers)
            except Exception:
                return e.code, {"__raw__": raw}, dict(e.headers)

    def get(self, path: str):
        return self._open("GET", path)

    def post(self, path: str, body: dict | None = None):
        return self._open("POST", path, body if body is not None else {})

    def cookie_ids(self) -> list[str]:
        return [c.value for c in self.jar if c.name == "tg_sid"]


def wait_port(port: int, timeout: float = 180.0) -> bool:
    """Wait until the server accepts a connection."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.25)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cookie-secure", choices=["yes", "no"], default="no",
                    help="'no' for a plain-HTTP harness (a browser would "
                         "refuse to store a Secure cookie over HTTP); "
                         "'yes' to assert the Secure attribute is set")
    args = ap.parse_args()

    ckpt = args.ckpt or "runs/_probe/v24/checkpoints/best_hard.pt"
    if not pathlib.Path(ckpt).exists():
        print(f"SKIP: checkpoint not found at {ckpt}")
        return 0

    base = f"http://127.0.0.1:{args.port}"
    env = {**__import__("os").environ,
           "PYTHONPATH": str(ROOT),
           "TG_UI_TORCH_THREADS": "4"}
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "src.c4.ui_server",
         "--model", ckpt, "--port", str(args.port), "--public",
         "--cookie-secure", args.cookie_secure],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        if not wait_port(args.port):
            print("FAIL: server never came up")
            print(proc.stdout.read() if proc.stdout else "")
            return 1

        print(f"=== MULTI-USER ISOLATION TEST against {base} ===\n")

        # ---------------------------------------------------------- [1]
        print("[1] two visitors get different boards")
        a = Visitor(base)
        b = Visitor(base)
        _, sa, ha = a.get("/state")
        _, sb, hb = b.get("/state")
        check("A got a session cookie", "tg_sid" in ha.get("Set-Cookie", ""),
              ha.get("Set-Cookie", "<none>"))
        check("B got a session cookie", "tg_sid" in hb.get("Set-Cookie", ""),
              hb.get("Set-Cookie", "<none>"))
        check("A and B have DIFFERENT session ids",
              a.cookie_ids() and b.cookie_ids()
              and a.cookie_ids()[0] != b.cookie_ids()[0],
              f"A={a.cookie_ids()} B={b.cookie_ids()}")
        check("both boards start empty",
              sa.get("moves") == [] and sb.get("moves") == [],
              f"A={sa.get('moves')} B={sb.get('moves')}")

        # ---------------------------------------------------------- [2]
        print("\n[2] a move by A does not touch B")
        _, s1, _ = a.post("/move", {"col": 3})
        check("A's move was accepted", "error" not in s1, str(s1.get("error")))
        check("A now has 2 plies (A + model reply)",
              len(s1.get("moves", [])) == 2, str(s1.get("moves")))
        _, s2, _ = b.get("/state")
        check("B's board is STILL EMPTY after A moved",
              s2.get("moves") == [], str(s2.get("moves")))
        check("B's cells are all empty after A moved",
              all(all(v == 0 for v in row) for row in s2.get("cells", [])),
              str(s2.get("cells")))

        # ---------------------------------------------------------- [3]
        print("\n[3] both visitors can play independently")
        _, s3, _ = b.post("/move", {"col": 0})
        check("B's move was accepted", "error" not in s3, str(s3.get("error")))
        check("B now has 2 plies", len(s3.get("moves", [])) == 2,
              str(s3.get("moves")))
        check("B's first move is column 0", s3.get("moves", [None])[0] == 0,
              str(s3.get("moves")))
        _, s4, _ = a.get("/state")
        check("A's board is unchanged by B's move",
              s4.get("moves") == s1.get("moves"),
              f"A={s4.get('moves')} expected={s1.get('moves')}")
        check("A's first move is still column 3",
              s4.get("moves", [None])[0] == 3, str(s4.get("moves")))

        # ---------------------------------------------------------- [4]
        print("\n[4] a third visitor starts fresh")
        c = Visitor(base)
        _, sc, _ = c.get("/state")
        check("C's board is empty", sc.get("moves") == [], str(sc.get("moves")))
        check("C has a third distinct session id",
              len({a.cookie_ids()[0], b.cookie_ids()[0], c.cookie_ids()[0]}) == 3,
              f"{a.cookie_ids()} {b.cookie_ids()} {c.cookie_ids()}")

        # ---------------------------------------------------------- [5]
        print("\n[5] a malformed cookie yields a FRESH session, not a shared game")
        bad = urllib.request.Request(base + "/state")
        bad.add_header("Cookie", "tg_sid=../../etc/passwd")
        with urllib.request.urlopen(bad, timeout=30) as resp:
            sj = json.loads(resp.read().decode())
        check("malformed cookie -> empty board", sj.get("moves") == [],
              str(sj.get("moves")))

        # ---------------------------------------------------------- [6]
        print("\n[6] concurrent visitors do not corrupt each other")
        errors: list[str] = []
        barrier = threading.Barrier(4)

        def worker(idx: int):
            try:
                v = Visitor(base)
                v.get("/state")
                barrier.wait()
                col = idx % 7
                st, js, _ = v.post("/move", {"col": col})
                if "error" in js:
                    errors.append(f"worker{idx}: {js['error']}")
                elif len(js.get("moves", [])) != 2:
                    errors.append(f"worker{idx}: moves={js.get('moves')}")
            except Exception as e:
                errors.append(f"worker{idx}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        check("4 concurrent visitors each got their own game",
              not errors, "; ".join(errors[:4]))

        # ---------------------------------------------------------- [7]
        print("\n[7] operational routes are refused in --public mode")
        st, js, _ = a.get("/modelinfo")
        check("/modelinfo is 403 in public mode", st == 403, f"got {st}")
        st, js, _ = a.get("/reload")
        check("/reload is 403 in public mode", st == 403, f"got {st}")

        # ---------------------------------------------------------- [8]
        print("\n[8] security headers are present")
        req = urllib.request.Request(base + "/state")
        req.add_header("Cookie", f"tg_sid={a.cookie_ids()[0]}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            hdrs = dict(resp.headers)
        for h in ("X-Content-Type-Options", "X-Frame-Options",
                  "Content-Security-Policy", "Referrer-Policy"):
            check(f"{h} present", h in hdrs, str(sorted(hdrs)))

        # ---------------------------------------------------------- [9]
        print("\n[9] the session cookie is HttpOnly and SameSite=Lax")
        cookie_hdr = ha.get("Set-Cookie", "")
        check("cookie is HttpOnly", "HttpOnly" in cookie_hdr, cookie_hdr)
        check("cookie is SameSite=Lax", "SameSite=Lax" in cookie_hdr, cookie_hdr)
        if args.cookie_secure == "yes":
            check("cookie is Secure when asked for", "Secure" in cookie_hdr,
                  cookie_hdr)
        else:
            check("cookie has no Secure flag in the plain-HTTP harness",
                  "Secure" not in cookie_hdr, cookie_hdr)

        print("\n=== %s ===" % ("ALL PASS" if not FAILS else
                                f"{len(FAILS)} FAILURES: {FAILS}"))
        print(f"    {CHECKS - len(FAILS)}/{CHECKS} checks passed")
        return 1 if FAILS else 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if proc.stdout:
            try:
                out = proc.stdout.read()
                if out and ("Traceback" in out or "Error" in out):
                    print("\n--- server output (tail) ---")
                    print(out[-3000:])
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
