#!/usr/bin/env python
"""Training log viewer backend.

Serves a live view of an in-progress training run: the current step, the
metric history, and the ASCII boards that ``train.py`` appends to
``snapshots.txt`` -- i.e. the games the model is actually playing while it
trains.

Run it alongside training:

    python -m src.c4.trainlog --run-dir runs/c4/v7mixed_v9 --port 7861
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_HEADER = re.compile(r"^=== step (\d+)\s+(.*?)\s*===\s*$")
_PLY = re.compile(r"^\s*ply\s+(\d+):\s*(.*)$")
_ROW = re.compile(r"^[.XOxo](?: [.XOxo]){6}\s*$")
_COORD = re.compile(r"^0 1 2 3 4 5 6\s*$")


def parse_snapshots(path: Path, max_games: int = 40) -> list[dict[str, Any]]:
    """Parse snapshots.txt into games of typed plies.

    The format written by ``train.py::_play_one_snapshot`` is:

        === step 4000  model-vs-heuristic  agent=P1 ===
          ply  1: model -> col 0
        <6 board rows + coordinate row>
          ply  2: heuristic -> col 3
        ...

    Only the final ply of each game is returned in full; earlier plies are
    kept as a light list so the viewer can scrub through a game without
    shipping every board to the browser.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    except OSError:
        return []

    games: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    ply: dict[str, Any] | None = None
    rows: list[str] = []

    def flush_ply() -> None:
        nonlocal ply, rows
        if ply is not None:
            if rows:
                ply["board"] = rows[:]
            if cur is not None:
                cur["plies"].append(ply)
        ply = None
        rows = []

    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            flush_ply()
            if cur is not None:
                cur["outcome"] = _outcome_of(cur)
                games.append(cur)
            cur = {"step": int(m.group(1)), "label": m.group(2), "plies": []}
            continue
        if cur is None:
            continue

        if line.startswith(">>>"):
            if ply is not None:
                ply["result"] = line.strip().lstrip("> ").strip()
            continue
        if line.strip().startswith("corpus:") or line.strip().startswith("elapsed:"):
            continue

        m = _PLY.match(line)
        if m:
            flush_ply()
            ply = {"ply": int(m.group(1)), "who": _who_of(m.group(2)), "col": _col_of(m.group(2))}
            continue

        if ply is not None:
            if _ROW.match(line.rstrip()) or _COORD.match(line.rstrip()):
                rows.append(line.rstrip())
            else:
                flush_ply()

    flush_ply()
    if cur is not None:
        cur["outcome"] = _outcome_of(cur)
        games.append(cur)

    return games[-max_games:]


def _who_of(desc: str) -> str:
    if "-> col" not in desc:
        return ""
    return desc.split("-> col")[0].strip()


def _col_of(desc: str) -> int | None:
    m = re.search(r"->\s*col\s+(\d+)", desc)
    return int(m.group(1)) if m else None


def _outcome_of(game: dict[str, Any]) -> str:
    for p in reversed(game["plies"]):
        if p.get("result"):
            return p["result"]
    last = game["plies"][-1] if game["plies"] else None
    return f"in progress (ply {last['ply']})" if last else "empty"


def read_metrics(path: Path, limit: int = 400) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # half-written trailing line while training appends
    except OSError:
        return []
    return out[-limit:]


def run_status(run_dir: Path) -> dict[str, Any]:
    metrics = read_metrics(run_dir / "metrics.jsonl")
    last = metrics[-1] if metrics else {}

    snap_path = run_dir / "snapshots.txt"
    games = parse_snapshots(snap_path)
    latest = games[-1] if games else None

    return {
        "run_dir": str(run_dir),
        "step": last.get("step"),
        "total_steps": None,
        "metrics": metrics,
        "games": games,
        "latest_game": latest,
        "snapshot_count": len(games),
        "corpus_size": last.get("corpus_size"),
    }


class Handler(BaseHTTPRequestHandler):
    run_dir: Path = Path(".")
    page_path: Path = Path("trainlog.html")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        pass  # keep the training console clean

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/state"):
            self._json(200, run_status(self.run_dir))
            return
        if self.path.startswith("/api/games"):
            self._json(200, {"games": parse_snapshots(self.run_dir / "snapshots.txt")})
            return
        if self.path == "/" or self.path.startswith("/index"):
            if self.page_path.exists():
                body = self.page_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404, "trainlog.html not found")
            return
        self.send_error(404, "unknown GET")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"trainlog: no such run dir: {run_dir}", file=sys.stderr)
        return 2

    here = Path(__file__).resolve().parent
    Handler.run_dir = run_dir
    Handler.page_path = here / "trainlog.html"

    print(f"trainlog: watching {run_dir} -> http://{args.host}:{args.port}/", flush=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())