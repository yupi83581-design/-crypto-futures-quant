"""Minimal read-only HTTP transport for the Quant Command Center snapshot."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from src.production.command_center_contract import QuantCommandCenterSnapshot


SnapshotProvider = Callable[[], QuantCommandCenterSnapshot]


def create_snapshot_server(
    host: str,
    port: int,
    snapshot_provider: SnapshotProvider,
) -> ThreadingHTTPServer:
    """Create a GET-only HTTP server exposing /snapshot."""
    if not callable(snapshot_provider):
        raise TypeError("snapshot_provider must be callable")

    class SnapshotHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self) -> None:
            # /snapshot is a public, read-only monitoring endpoint. Allow
            # browser-based presentation layers such as the AI Studio UI to
            # read the live snapshot cross-origin.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Accept, Content-Type",
            )

        def _send_json(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/snapshot":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                snapshot = snapshot_provider()
                if not isinstance(snapshot, QuantCommandCenterSnapshot):
                    raise TypeError("snapshot_provider returned invalid snapshot")
                self._send_json(HTTPStatus.OK, snapshot.to_dict())
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "snapshot unavailable", "detail": str(exc)},
                )

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self._send_cors_headers()
            self.end_headers()

        def _method_not_allowed(self) -> None:
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "GET, OPTIONS")
            self.send_header("Content-Length", "0")
            self._send_cors_headers()
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PUT(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PATCH(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_DELETE(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), SnapshotHandler)
