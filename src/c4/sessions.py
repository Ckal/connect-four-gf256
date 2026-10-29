"""Per-visitor game state for the Connect Four UI server.

WHY THIS MODULE EXISTS
----------------------
The UI server originally kept ONE game in a module-level global. That is
fine for a single user on a laptop, and fatal for a public deployment:
every visitor shared one board, so a move by one person moved everyone
else's game, and the game state was a cross-site information leak.

Public hosting needs the opposite property: two browser tabs are two
independent games, and a visitor's board is nobody else's business.

This module provides the isolation boundary:

  * ``SessionStore`` maps an opaque, unguessable session id to one
    ``Game`` instance. The id travels in a cookie.
  * Games expire after an idle TTL so an unattended public server does
    not accumulate boards forever.
  * Each session gets a small request budget, so one client cannot
    monopolise the (single) CPU-bound inference path.
  * The MODEL is deliberately shared: it is read-only during inference
    (eval mode, no_grad, immutable integer tables), so one copy serves
    every session. Only the mutable game state is per-session.

SECURITY NOTES
--------------
  * Session ids are 256 bits of ``secrets`` entropy, so they cannot be
    guessed or enumerated.
  * The cookie is ``HttpOnly`` (script cannot read it) and
    ``SameSite=Lax`` (a cross-site POST cannot ride the session). When the
    game is served into a third-party iframe it is ``SameSite=None;
    Secure`` instead, because a ``Lax`` cookie is not sent on a
    cross-site subresource request.
  * A session id supplied by a client is validated against a strict
    character set before it is used as a dict key, so a hostile id can
    never reach the filesystem or a log line unfiltered.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from typing import Any, Callable, Iterable

# A session id is 43 base64url characters (32 random bytes). Only the
# URL-safe alphabet is allowed, so an id can never contain a path
# separator, a quote, or a control character.
_SID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")

DEFAULT_TTL_S = 2 * 3600.0          # idle games are dropped after 2 h
DEFAULT_MAX_SESSIONS = 512
DEFAULT_RATE_WINDOW_S = 10.0        # sliding window for the rate limit
DEFAULT_RATE_MAX = 120              # requests per session per window


def new_session_id() -> str:
    """A fresh, unguessable session id (256 bits of entropy)."""
    return secrets.token_urlsafe(32)


def valid_session_id(sid: Any) -> bool:
    """True if `sid` is a well-formed session id.

    Applied to EVERY client-supplied id before it is used. A malformed
    id is refused rather than sanitised: there is no legitimate reason
    for a client to send one, and silently repairing it would let two
    different malformed ids collapse onto one session.
    """
    return isinstance(sid, str) and bool(_SID_RE.match(sid))


class SessionStore:
    """Thread-safe map of session id -> per-visitor game state.

    Args:
        factory: called with no arguments to build a new game object.
            The server passes a closure that binds the current agent and
            model name, so a game created after a hot reload uses the
            new model.
        ttl_s: idle lifetime of a session.
        max_sessions: hard cap on concurrent sessions. When the cap is
            reached the LEAST RECENTLY USED session is evicted, so a
            burst of new visitors cannot evict the active ones.
        rate_window_s / rate_max: per-session request budget.
    """

    def __init__(self,
                 factory: Callable[[], Any],
                 ttl_s: float = DEFAULT_TTL_S,
                 max_sessions: int = DEFAULT_MAX_SESSIONS,
                 rate_window_s: float = DEFAULT_RATE_WINDOW_S,
                 rate_max: int = DEFAULT_RATE_MAX) -> None:
        self._factory = factory
        self._ttl_s = float(ttl_s)
        self._max_sessions = int(max_sessions)
        self._rate_window_s = float(rate_window_s)
        self._rate_max = int(rate_max)
        self._lock = threading.Lock()
        # sid -> game
        self._games: dict[str, Any] = {}
        # sid -> monotonic timestamp of last use
        self._seen: dict[str, float] = {}
        # sid -> deque of request timestamps (rate limiting)
        self._hits: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ ids

    def create(self, sid: str | None = None) -> tuple[str, Any]:
        """Create (or reuse) a session.

        Returns ``(sid, game)``. If `sid` is supplied and well-formed it
        is reused when it already exists, so a client that presents a
        cookie keeps its board across a server restart of the *request*
        (not of the process).
        """
        if sid is None or not valid_session_id(sid):
            sid = new_session_id()
        now = time.monotonic()
        with self._lock:
            game = self._games.get(sid)
            if game is None:
                game = self._factory()
                self._games[sid] = game
                self._hits.setdefault(sid, [])
                if len(self._games) > self._max_sessions:
                    self._evict_lru_locked()
            self._seen[sid] = now
            return sid, game

    def get(self, sid: str | None) -> Any | None:
        """Return the game for `sid`, or None if there is no such session.

        A session that has been idle for longer than the TTL is treated
        as GONE, not resurrected. Without this check the idle timer
        would only be enforced by the background sweeper, so a visitor
        who kept sending requests would keep a board alive forever --
        and a session could be revived in the window between the TTL
        expiring and the sweeper running.
        """
        if not valid_session_id(sid):
            return None
        key = str(sid)
        now = time.monotonic()
        with self._lock:
            game = self._games.get(key)
            if game is None:
                return None
            if self._expired_locked(key, now):
                self._drop_locked(key)
                return None
            self._seen[key] = now
            return game

    def _expired_locked(self, sid: str, now: float) -> bool:
        """True when `sid` has been idle longer than the TTL.

        Caller holds the lock.
        """
        seen = self._seen.get(sid)
        return seen is None or (now - seen) > self._ttl_s

    def _drop_locked(self, sid: str) -> None:
        """Remove a session and its bookkeeping. Caller holds the lock."""
        self._games.pop(sid, None)
        self._seen.pop(sid, None)
        self._hits.pop(sid, None)

    def touch(self, sid: str) -> None:
        """Mark a session as recently used (keeps it out of the LRU)."""
        if not valid_session_id(sid):
            return
        with self._lock:
            if sid in self._games:
                self._seen[sid] = time.monotonic()

    def drop(self, sid: str) -> bool:
        """Forget a session entirely. Returns True if it existed."""
        if not valid_session_id(sid):
            return False
        with self._lock:
            existed = sid in self._games
            self._drop_locked(sid)
            return existed

    # ------------------------------------------------------------ rate limit

    def allow(self, sid: str | None) -> bool:
        """Consume one request from the session's budget.

        Returns False when the session is over budget. A rejected
        request still refreshes the session's idle timer, so a client
        hammering the server does not lose its game.
        """
        now = time.monotonic()
        with self._lock:
            key = sid if valid_session_id(sid) else None
            hits = self._hits.setdefault(key, []) if key else []
            cutoff = now - self._rate_window_s
            while hits and hits[0] < cutoff:
                hits.pop(0)
            if len(hits) >= self._rate_max:
                if sid:
                    self._seen[sid] = now
                return False
            hits.append(now)
            if sid:
                self._seen[sid] = now
            return True

    # ---------------------------------------------------------------- sweep

    def sweep(self) -> int:
        """Drop sessions idle for longer than the TTL. Returns the count."""
        now = time.monotonic()
        with self._lock:
            dead = [sid for sid in self._seen
                    if self._expired_locked(sid, now)]
            for sid in dead:
                self._drop_locked(sid)
            return len(dead)

    def _evict_lru_locked(self) -> None:
        """Evict the least recently used session. Caller holds the lock."""
        if not self._seen:
            return
        oldest = min(self._seen, key=lambda s: self._seen[s])
        self._games.pop(oldest, None)
        self._seen.pop(oldest, None)
        self._hits.pop(oldest, None)

    # --------------------------------------------------------------- access

    def all_games(self) -> Iterable[Any]:
        """Snapshot of every live game (used by the hot-reload swap)."""
        with self._lock:
            return list(self._games.values())

    def count(self) -> int:
        with self._lock:
            return len(self._games)

    def stats(self) -> dict[str, Any]:
        """Small, safe summary for /modelinfo-style diagnostics."""
        now = time.monotonic()
        with self._lock:
            ages = [now - s for s in self._seen.values()]
            return {
                "sessions": len(self._games),
                "max_sessions": self._max_sessions,
                "ttl_s": self._ttl_s,
                "rate_window_s": self._rate_window_s,
                "rate_max": self._rate_max,
                "oldest_idle_s": round(max(ages), 1) if ages else 0.0,
            }


# --------------------------------------------------------------- cookies

def parse_cookies(header: str | None) -> dict[str, str]:
    """Parse a Cookie header into a dict. Never raises.

    When the same cookie name appears more than once the FIRST value
    wins. RFC 6265 says a user agent sends cookies in order of
    decreasing path specificity, so the first occurrence is the most
    specific one -- which is the one the server intended. Taking the
    last value instead would let a broader-scoped cookie override the
    session cookie.
    """
    out: dict[str, str] = {}
    if not header:
        return out
    for part in header.split(";"):
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip().strip('"')
        if name and name not in out:
            out[name] = value
    return out


def session_cookie(sid: str, *, secure: bool = False,
                   max_age: int | None = None,
                   samesite: str = "Lax") -> str:
    """Build a Set-Cookie header value for the session cookie.

    ``HttpOnly`` keeps the id out of reach of page scripts, and ``Path=/``
    scopes it to the whole app. ``Secure`` is added when the site is
    served over HTTPS (i.e. when TLS is terminated by the proxy).

    ``samesite`` defaults to ``Lax``, which stops a cross-site form post
    from riding the session. It must be ``None`` when the game is embedded
    in a frame on ANOTHER site: a ``Lax`` cookie is not sent on a
    cross-site subresource request, so in an iframe the browser would
    drop it and every request would start a fresh, empty game. ``None``
    requires ``Secure`` (browsers reject ``SameSite=None`` without it),
    which is why ``secure=True`` is enforced alongside it.
    """
    if samesite not in ("Lax", "Strict", "None"):
        raise ValueError(f"unsupported SameSite value: {samesite!r}")
    if samesite == "None" and not secure:
        raise ValueError("SameSite=None requires Secure=True")
    parts = [f"tg_sid={sid}", "Path=/", "HttpOnly", f"SameSite={samesite}"]
    if secure:
        parts.append("Secure")
    if max_age is not None:
        parts.append(f"Max-Age={int(max_age)}")
    return "; ".join(parts)


def clear_session_cookie(*, samesite: str = "Lax") -> str:
    """Build a Set-Cookie header value that expires the session cookie."""
    if samesite not in ("Lax", "Strict", "None"):
        raise ValueError(f"unsupported SameSite value: {samesite!r}")
    return f"tg_sid=; Path=/; HttpOnly; SameSite={samesite}; Max-Age=0"
