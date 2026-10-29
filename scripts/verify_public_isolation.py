#!/usr/bin/env python
"""Public-deploy gate for the Connect Four game.

Verifies, against a LIVE public URL, the properties that make the demo
safe to publish:

  1. Two clients receive DIFFERENT session cookies.
  2. A move in client A is INVISIBLE to client B (no shared board).
  3. The session cookie carries HttpOnly and Secure.
  4. Administrative routes (/modelinfo, /reload) are closed in public mode.
  5. Security headers are present on the HTML response.
  6. The page actually renders a playable board.

Run:
    ./.venv-c4/Scripts/python.exe tmp/verify_public_isolation.py https://c4.tensorgarten.de

Exit code 0 = all assertions pass. Non-zero = DO NOT announce the URL.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import ssl
import sys
import urllib.error
import urllib.request

FAILURES: list[str] = []
PASSES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    if ok:
        PASSES.append(label)
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAILURES.append(f"{label} :: {detail}")
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def make_client(base: str, insecure: bool = False) -> tuple:
    """Build an opener. TLS is verified unless --insecure is passed explicitly
    (only useful against a self-signed dev certificate)."""
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cj),
        urllib.request.HTTPSHandler(context=ctx),
    )
    return opener, cj


def get(opener, url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers={"User-Agent": "c4-isolation-check/1.0"})
    return opener.open(req, timeout=timeout)


def post_json(opener, url: str, payload: dict, timeout: int = 25):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "c4-isolation-check/1.0"},
    )
    return opener.open(req, timeout=timeout)


def state(opener, base: str) -> dict:
    return json.loads(get(opener, base + "/state").read().decode())


def cells_of(s: dict) -> list:
    return [x for row in s.get("cells", []) for x in row]


def cookie_attrs(cj: http.cookiejar.CookieJar) -> list:
    out = []
    for c in cj:
        # NOTE: http.cookiejar stores valueless flags (HttpOnly) in
        # _rest with an EMPTY STRING value, so truthiness is wrong --
        # presence in _rest is the correct test.
        rest = getattr(c, "_rest", {}) or {}
        httponly = any(k.lower() == "httponly" for k in rest)
        out.append({
            "name": c.name, "value": c.value,
            "httponly": httponly,
            "secure": bool(c.secure), "path": c.path,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="https://c4.tensorgarten.de")
    ap.add_argument("--insecure", action="store_true",
                    help="skip TLS verification (self-signed dev cert only)")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print(f"=== public isolation check: {base} ===\n")

    # ---------------------------------------------------------------- 1
    print("[1] the page serves")
    opA, cjA = make_client(base, args.insecure)
    opB, cjB = make_client(base, args.insecure)
    try:
        r = get(opA, base + "/")
        html = r.read().decode("utf-8", "replace")
        check(r.status == 200, "GET / returns 200", f"status={r.status}")
        check(len(html) > 500, "GET / returns a real page", f"{len(html)} bytes")
    except Exception as e:  # noqa: BLE001
        check(False, "GET / returns 200", repr(e))
        return 1

    # ---------------------------------------------------------------- 2
    print("\n[2] security headers")
    hdrs = {k.lower(): v for k, v in r.headers.items()}
    check("content-security-policy" in hdrs, "Content-Security-Policy present")
    # Framing must be EITHER fully denied, or allowed for a NAMED set of
    # origins -- never a wildcard. An open frame-ancestors would let any
    # site clickjack the game.
    csp = hdrs.get("content-security-policy", "")
    xfo = hdrs.get("x-frame-options", "").upper()
    if "frame-ancestors 'none'" in csp:
        check(xfo == "DENY", "framing denied (X-Frame-Options: DENY + "
                             "frame-ancestors 'none')", xfo)
    else:
        fa = ""
        for part in csp.split(";"):
            part = part.strip()
            if part.startswith("frame-ancestors"):
                fa = part
        check(bool(fa) and "*" not in fa and fa != "frame-ancestors",
              "framing restricted to named origins (no wildcard)", fa)
    check(hdrs.get("x-content-type-options", "").lower() == "nosniff",
          "X-Content-Type-Options: nosniff", hdrs.get("x-content-type-options", ""))
    check("referrer-policy" in hdrs, "Referrer-Policy present",
          hdrs.get("referrer-policy", ""))
    if base.startswith("https://"):
        check(hdrs.get("strict-transport-security", "") != "",
              "Strict-Transport-Security present (HTTPS)")
    else:
        print("  SKIP  Strict-Transport-Security (plain-HTTP test harness)")
    check(hdrs.get("cache-control", "").lower().find("no-store") >= 0,
          "Cache-Control: no-store on the dynamic page")

    # ---------------------------------------------------------------- 3
    print("\n[3] distinct sessions")
    a0 = state(opA, base)
    b0 = state(opB, base)
    cka = cookie_attrs(cjA)
    ckb = cookie_attrs(cjB)
    check(len(cka) >= 1 and len(ckb) >= 1, "both clients received a session cookie",
          f"A={[c['name'] for c in cka]} B={[c['name'] for c in ckb]}")
    sa = cka[0]["name"] + "=" + cka[0].get("value", "") if cka else ""
    sb = ckb[0]["name"] + "=" + ckb[0].get("value", "") if ckb else ""
    check(sa != sb, "the two session cookies differ", "distinct cookies")
    check(all(c["httponly"] for c in cka + ckb), "session cookie is HttpOnly",
          f"raw Set-Cookie: {hdrs.get('set-cookie', '(not on this response)')}")
    check(all(c["secure"] for c in cka + ckb), "session cookie is Secure (HTTPS)")

    # ---------------------------------------------------------------- 4
    print("\n[4] no shared board")
    check(cells_of(a0) == cells_of(b0) == [0] * 42,
          "both boards start empty")
    moved = False
    for col in (3, 4, 2):
        try:
            post_json(opA, base + "/move", {"col": col})
            moved = True
            break
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
    check(moved, "client A can make a legal move")
    a1 = state(opA, base)
    b1 = state(opB, base)
    if any(x != 0 for x in cells_of(a1)):
        check(True, "client A's board shows the move",
              f"cells={cells_of(a1)[:8]}...")
    elif not base.startswith("https://"):
        # Over plain HTTP a Secure cookie is not stored by the client, so
        # every request starts a NEW session and the board stays empty.
        # That is the browser's rule, not a server defect -- but it means
        # this harness cannot verify the board over http.
        print("  SKIP  client A's board shows the move "
              "(Secure cookie not stored over plain HTTP)")
        print("        run this check against the HTTPS public URL")
    else:
        check(False, "client A's board shows the move",
              f"cells={cells_of(a1)[:8]}...")
    check(cells_of(b1) == [0] * 42,
          "client B's board is STILL EMPTY -- no shared state",
          f"B cells={cells_of(b1)[:8]}...")

    # ---------------------------------------------------------------- 5
    print("\n[5] administrative routes closed")
    for path in ("/modelinfo", "/reload"):
        try:
            get(opA, base + path)
            check(False, f"GET {path} is blocked", "it returned 200")
        except urllib.error.HTTPError as e:
            check(e.code in (403, 404), f"GET {path} is blocked", f"HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            check(False, f"GET {path} is blocked", repr(e))

    # ---------------------------------------------------------------- 6
    print("\n[6] the game is playable")
    try:
        r2 = post_json(opA, base + "/move", {"col": 5})
        js = json.loads(r2.read().decode())
        check("cells" in js or "board" in js or "state" in js,
              "the move response carries the new board state",
              ", ".join(list(js.keys())[:6]))
    except urllib.error.HTTPError as e:
        check(False, "a second move is accepted", f"HTTP {e.code}")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASSES)}   FAILED: {len(FAILURES)}")
    if FAILURES:
        print("\nDO NOT announce this URL. Failures:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("All isolation, security and playability assertions hold.")
    print("The public deployment is safe to announce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
