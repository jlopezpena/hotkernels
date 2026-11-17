"""Serves the built site on this machine, rebuilding it when a source changes.

For writing and reviewing pages. The site itself needs no server beyond one
that hands out files.
"""
from __future__ import annotations

import logging
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import data as D
from .build import SITE, build

log = logging.getLogger(__name__)

# Address the server listens on: this machine only.
HOST = "127.0.0.1"
# File suffixes inside the package that are bytecode caches, not sources.
_IGNORED_SUFFIXES = {".pyc"}


def _sources() -> list[Path]:
    """List every file a build reads: the site's own, and the repository data it quotes."""
    own = [p for p in SITE.rglob("*") if p.is_file() and p.suffix not in _IGNORED_SUFFIXES]
    return own + [D.README, D.FINDINGS, D.PYTHON_TIMINGS, *D.KERNELS.glob("*/attempts.md")]


def _newest_change() -> float:
    """Return the latest modification time among the build's sources, in seconds since the epoch."""
    return max(p.stat().st_mtime for p in _sources())


class _Server(ThreadingHTTPServer):
    """An HTTP server that keeps the site it serves up to date with its sources."""

    def __init__(self, address: tuple[str, int], out: Path) -> None:
        """Build the site once and start listening.

        Args:
            address: Host and port to listen on.
            out: Directory the site is built into and served from.
        """
        self.out = out
        # Modification time of the newest source at the last build.
        self.built_at = 0.0
        # Requests arrive on several threads; only one may rebuild at a time.
        self.build_lock = threading.Lock()
        self.rebuild_if_stale()
        super().__init__(address, partial(_Handler, directory=str(out)))

    def rebuild_if_stale(self) -> None:
        """Rebuild the site if any source changed since the last build.

        A build that fails is logged and the previous output stays in place,
        so a half-edited page does not take the server down.
        """
        with self.build_lock:
            newest = _newest_change()
            if newest <= self.built_at:
                return
            self.built_at = newest
            try:
                build(self.out)
            except Exception:  # noqa: BLE001 - any build error is reported, not fatal
                log.exception("Build failed; serving the previous output")


class _Handler(SimpleHTTPRequestHandler):
    """Serves files from the output directory, checking for changes on each page request."""

    def do_GET(self) -> None:
        """Answer a GET, rebuilding first when the request is for a page.

        A request for any other file waits for a build that is under way, so
        it is never answered from a half-written output directory.
        """
        if self.path.split("?")[0].endswith("/"):
            self.server.rebuild_if_stale()
        else:
            with self.server.build_lock:
                pass
        super().do_GET()

    def end_headers(self) -> None:
        """Tell the browser not to cache, so an edit shows on reload."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        """Send the request log through `logging` at debug level instead of to stderr."""
        log.debug(format, *args)


def serve(out: Path, port: int) -> None:
    """Serve the site until interrupted.

    Args:
        out: Directory the site is built into and served from.
        port: TCP port to listen on.
    """
    server = _Server((HOST, port), out)
    log.info("Serving %s at http://%s:%d/ (Ctrl-C to stop)", out, HOST, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped")
    finally:
        server.server_close()
