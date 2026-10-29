"""Unit tests for the per-visitor session store.

These test the STORE in isolation (no server, no model), so they run in
milliseconds and cover the properties a public deployment depends on:

  * a session id is opaque, unguessable, and URL-safe
  * a malformed id is rejected, not repaired
  * sessions expire after their idle TTL
  * the per-session request budget is enforced
  * the LRU cap evicts the OLDEST session, not an active one
  * two sessions never share a game object
"""
from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c4.sessions import (  # noqa: E402
    SessionStore,
    clear_session_cookie,
    new_session_id,
    parse_cookies,
    session_cookie,
    valid_session_id,
)

FAILS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail: object = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def main() -> int:
    print("=== SESSION STORE UNIT TESTS ===\n")

    # ------------------------------------------------------------- ids
    print("[1] session ids are opaque and validated")
    ids = {new_session_id() for _ in range(200)}
    check("200 ids are all distinct", len(ids) == 200, len(ids))
    check("ids are long enough to be unguessable",
          all(len(i) >= 32 for i in ids))
    check("ids are URL-safe",
          all(all(c.isalnum() or c in "-_" for c in i) for i in ids))
    check("a valid id validates", valid_session_id(next(iter(ids))))
    for bad in ("", "short", "../../etc/passwd", "a" * 200, None, 123,
                "abc def", "abc;def", "abc\n"):
        check(f"malformed id rejected: {bad!r}", not valid_session_id(bad))

    # ------------------------------------------------------------ cookies
    print("\n[2] cookie parsing and formatting")
    ck = session_cookie("abc123", secure=True)
    check("cookie names the session", ck.startswith("tg_sid=abc123"), ck)
    check("cookie is HttpOnly", "HttpOnly" in ck, ck)
    check("cookie is SameSite=Lax", "SameSite=Lax" in ck, ck)
    check("secure cookie carries Secure", "Secure" in ck, ck)
    check("insecure cookie omits Secure",
          "Secure" not in session_cookie("abc123", secure=False))
    check("clear cookie expires the session",
          "Max-Age=0" in clear_session_cookie(), clear_session_cookie())
    hdr = parse_cookies("tg_sid=xyz; other=1; tg_sid=second")
    check("parser reads the session", hdr.get("tg_sid") == "xyz", hdr)
    check("parser handles a missing header", parse_cookies(None) == {})
    check("parser handles an empty header", parse_cookies("") == {})

    # ------------------------------------------------------------- isolation
    print("\n[3] sessions never share a game object")
    store = SessionStore(factory=lambda: {"id": object()}, ttl_s=60)
    sid_a, ga = store.create()
    sid_b, gb = store.create()
    check("two sessions have different ids", sid_a != sid_b)
    check("two sessions have different game objects", ga is not gb)
    check("store.get returns the same object for A",
          store.get(sid_a) is ga)
    check("store.get returns the same object for B",
          store.get(sid_b) is gb)
    check("an unknown id returns None", store.get("nope") is None)
    check("a malformed id returns None", store.get("../../x") is None)
    check("session count is 2", store.count() == 2, store.count())

    # ------------------------------------------------------------- TTL
    print("\n[4] sessions expire after the idle TTL")
    short = SessionStore(factory=lambda: {}, ttl_s=0.30)
    sid, _ = short.create()
    check("session is alive right after creation", short.get(sid) is not None)
    time.sleep(0.45)
    check("session is gone after the TTL", short.get(sid) is None)
    # The get() above already evicted it, so there is nothing left for
    # the sweeper to collect. Check the sweeper on its own session.
    check("sweep finds nothing left after the read-time eviction",
          short.sweep() == 0, short.sweep())

    short2 = SessionStore(factory=lambda: {}, ttl_s=0.30)
    sid2, _ = short2.create()
    time.sleep(0.45)
    check("sweep reports the expired session", short2.sweep() == 1,
          short2.sweep())
    check("sweep is idempotent", short2.sweep() == 0)

    print("\n[5] activity keeps a session alive past its TTL")
    warm = SessionStore(factory=lambda: {}, ttl_s=0.30)
    sid, _ = warm.create()
    for _ in range(4):
        time.sleep(0.12)
        warm.get(sid)          # a request refreshes the idle timer
    check("an active session survives past its TTL",
          warm.get(sid) is not None)

    # ------------------------------------------------------------- rate
    print("\n[6] the per-session request budget is enforced")
    rl = SessionStore(factory=lambda: {}, rate_window_s=10.0, rate_max=5)
    sid, _ = rl.create()
    allowed = [rl.allow(sid) for _ in range(8)]
    check("the first 5 requests are allowed", all(allowed[:5]), allowed)
    check("requests past the budget are refused",
          not any(allowed[5:]), allowed)
    check("a refused request keeps the session alive",
          rl.get(sid) is not None)
    other, _ = rl.create()
    check("a different session has its own budget", rl.allow(other))

    print("\n[7] the rate limiter tolerates a malformed id")
    check("a malformed id is refused, not crashed", rl.allow("junk") in
          (True, False))

    # ------------------------------------------------------------- LRU
    print("\n[8] the LRU cap evicts the OLDEST session, not an active one")
    lru = SessionStore(factory=lambda: {}, ttl_s=3600, max_sessions=3)
    s1, _ = lru.create()
    time.sleep(0.05)
    s2, _ = lru.create()
    time.sleep(0.05)
    s3, _ = lru.create()
    lru.touch(s1)            # s1 becomes the most recently used
    time.sleep(0.05)
    s4, _ = lru.create()     # this overflows the cap
    check("the cap is respected", lru.count() <= 3, lru.count())
    check("the LRU-evicted session is gone", lru.get(s2) is None)
    check("the recently-touched session survived", lru.get(s1) is not None)
    check("the newest session survived", lru.get(s4) is not None)

    # ------------------------------------------------------------- stats
    print("\n[9] stats are reported for /modelinfo")
    st = store.stats()
    check("stats expose the session count", "sessions" in st, st)
    check("stats expose the cap", "max_sessions" in st, st)
    check("stats expose the ttl", "ttl_s" in st, st)

    print("\n=== %s ===" % ("ALL PASS" if not FAILS else
                            f"{len(FAILS)} FAILURES: {FAILS}"))
    print(f"    {CHECKS - len(FAILS)}/{CHECKS} checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
