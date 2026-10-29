"""Tests for the framed-embedding session cookie policy.

Covers the three properties that the Hugging Face Space needs:

  1. With no embedding allowlist the cookie stays `SameSite=Lax`
     (the CSRF protection of the standalone deployment).
  2. With an embedding allowlist AND a Secure cookie it becomes
     `SameSite=None; Secure` -- the only value a browser accepts in a
     cross-site iframe.
  3. `SameSite=None` without `Secure` is refused by the builder.

Plus the explicit Origin check that replaces the protection the Lax
cookie used to provide:

  4. A POST from an allowlisted embedding origin is accepted.
  5. A POST from a hostile origin is refused with 403.
  6. A POST without an Origin header is accepted (non-browser client).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.sessions import clear_session_cookie, session_cookie  # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(label)
        print(f"  FAIL  {label}   {detail}")


def test_lax_default() -> None:
    print("\n[1] the default cookie is SameSite=Lax")
    ck = session_cookie("A" * 43, secure=True)
    check("SameSite=Lax", "SameSite=Lax" in ck, ck)
    check("Secure present", "Secure" in ck, ck)
    check("HttpOnly present", "HttpOnly" in ck, ck)
    check("no SameSite=None", "SameSite=None" not in ck, ck)


def test_none_requires_secure() -> None:
    print("\n[2] SameSite=None requires Secure")
    try:
        session_cookie("A" * 43, secure=False, samesite="None")
        check("refused without Secure", False, "no exception raised")
    except ValueError:
        check("refused without Secure", True)

    ck = session_cookie("A" * 43, secure=True, samesite="None")
    check("accepted with Secure", "SameSite=None" in ck and "Secure" in ck, ck)


def test_invalid_samesite() -> None:
    print("\n[3] an unsupported SameSite value is refused")
    for bad in ("none", "LAX", "bogus", ""):
        try:
            session_cookie("A" * 43, secure=True, samesite=bad)
            check(f"refused {bad!r}", False, "no exception")
        except ValueError:
            check(f"refused {bad!r}", True)


def test_clear_matches() -> None:
    print("\n[4] the clearing cookie uses the same policy")
    ck = clear_session_cookie(samesite="None")
    check("clear uses SameSite=None", "SameSite=None" in ck, ck)
    check("clear expires it", "Max-Age=0" in ck, ck)
    ck2 = clear_session_cookie()
    check("default clear is Lax", "SameSite=Lax" in ck2, ck2)


def test_origin_gate() -> None:
    print("\n[5] the explicit Origin check accepts only trusted origins")
    # Import the module and exercise the gate through a minimal stub.
    from src.c4 import ui_server

    class _Hdr:
        def __init__(self, origin: str | None) -> None:
            self._origin = origin

        def get(self, key: str, default=None):
            if key.lower() == "origin":
                return self._origin
            return default

    class _Stub:
        def __init__(self, origin: str | None) -> None:
            self.headers = _Hdr(origin)

        def _origin_allowed(self) -> bool:
            # Deliberately calls the unbound method with a stub `self`:
            # the gate only reads `self.headers`.
            return ui_server.Handler._origin_allowed(self)  # type: ignore[arg-type]

    saved = set(ui_server._ALLOWED_ORIGINS)
    ui_server._ALLOWED_ORIGINS.clear()
    ui_server._ALLOWED_ORIGINS.add("https://c4.tensorgarten.de")
    ui_server._ALLOWED_ORIGINS.add("https://chris4k-connect-four-gf256.static.hf.space")
    try:
        check("own origin accepted",
              _Stub("https://c4.tensorgarten.de")._origin_allowed())
        check("embedding origin accepted",
              _Stub("https://chris4k-connect-four-gf256.static.hf.space")._origin_allowed())
        check("hostile origin refused",
              not _Stub("https://evil.example")._origin_allowed())
        check("no Origin header accepted",
              _Stub(None)._origin_allowed())
        check("null origin refused",
              not _Stub("null")._origin_allowed())
    finally:
        ui_server._ALLOWED_ORIGINS.clear()
        ui_server._ALLOWED_ORIGINS.update(saved)


def main() -> int:
    test_lax_default()
    test_none_requires_secure()
    test_invalid_samesite()
    test_clear_matches()
    test_origin_gate()
    print()
    if FAILURES:
        print(f"=== {CHECKS - len(FAILURES)} PASS, {len(FAILURES)} FAIL ===")
        for f in FAILURES:
            print(f"  failed: {f}")
        return 1
    print(f"=== ALL PASS ({CHECKS} checks) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
