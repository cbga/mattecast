"""Web control panel: live preview, background library with upload, settings.

Plain http.server so there is nothing extra to install. Uploads are sent as the
raw request body (no multipart), with the file name in the query string.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from mattecast.pipeline import INTERNAL_SIZES, PREVIEW_VIEWS, Engine, PreviewHub

log = logging.getLogger(__name__)

BOUNDARY = "mattecastframe"


def _page(name: str) -> bytes:
    return resources.files("mattecast").joinpath(name).read_bytes()


_VIRTUAL_IFACES = ("lo", "docker", "br-", "virbr", "veth", "vmnet", "vboxnet", "lxc", "cni", "flannel")


def _local_addresses() -> list:
    """IPv4 addresses another device could use to reach this computer.

    Looked up at runtime (nothing is stored), home network first, Tailscale last.
    """
    found = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True,
                             text=True, timeout=2).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "inet" and not parts[1].startswith(_VIRTUAL_IFACES):
                found.append(parts[3].split("/")[0])
    except (OSError, subprocess.SubprocessError):
        pass
    if not found:
        try:
            found = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout.split()
        except (OSError, subprocess.SubprocessError):
            found = []
    tailnet = ipaddress.ip_network("100.64.0.0/10")
    addrs = []
    for a in found:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if ip.version == 4 and not ip.is_loopback and not ip.is_link_local and a not in addrs:
            addrs.append(a)
    return sorted(addrs, key=lambda a: ipaddress.ip_address(a) in tailnet)


class ControlServer:
    def __init__(self, engine: Engine, preview: PreviewHub, host: str, port: int,
                 audio=None, avsync=None, calibrator=None) -> None:
        self.engine = engine
        self.bg = engine.backgrounds
        self.preview = preview
        self.audio = audio
        self.avsync = avsync
        self.calibrator = calibrator
        self._view_before_sync: Optional[str] = None
        handler = self._make_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="control", daemon=True)

    def start(self) -> None:
        self.thread.start()
        host, port = self.httpd.server_address[:2]
        shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        log.info("control panel: http://%s:%d/%s", shown, port, "  (listening on all interfaces)" if host == "0.0.0.0" else "")

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ----- A/V sync test --------------------------------------------------------

    def sync_prepare(self) -> dict:
        """Show the raw camera in the preview so the phone can be aimed.

        The cut-out usually removes the phone from the output, so the normal preview
        would not show whether the camera sees the flashes. The measurement itself
        always uses the raw camera frame.
        """
        if self._view_before_sync is None and self.preview.view != "camera":
            self._view_before_sync = self.preview.view
        self.preview.view = "camera"
        host, port = self.httpd.server_address[:2]
        loopback = host.startswith("127.") or host in ("localhost", "::1")
        if loopback:
            urls = []
        elif host in ("", "0.0.0.0", "::"):
            urls = [f"http://{a}:{port}/sync" for a in _local_addresses()]
        else:
            urls = [f"http://{host}:{port}/sync"]
        return {"ok": True, "urls": urls, "loopback_only": loopback}

    def sync_restore(self) -> None:
        if self._view_before_sync is not None:
            if self.preview.view == "camera":
                self.preview.view = self._view_before_sync
            self._view_before_sync = None

    # ----- API ---------------------------------------------------------------

    def state(self) -> dict:
        eng = self.engine
        return {
            "stats": eng.stats(),
            "background": self.bg.spec,
            "library": self.bg.list(),
            "preview_view": self.preview.view,
            "internal_sizes": list(INTERNAL_SIZES),
            "events": [{"t": t, "msg": m} for t, m in list(eng.matter.events)[-8:]],
            "audio": None if self.audio is None else {**self.audio.stats(), "sync": self.avsync.state()},
            "sync_test": None if self.calibrator is None else self.calibrator.result(),
        }

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet
                log.debug("http %s", fmt % args)

            # -- helpers
            def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: Optional[dict] = None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj).encode())

            def _error(self, msg: str, code: int = 400):
                self._json({"error": msg}, code)

            def _body_json(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            # -- routes
            def do_GET(self):  # noqa: N802
                url = urlparse(self.path)
                path = url.path
                try:
                    if path == "/":
                        return self._send(200, _page("panel.html"), "text/html; charset=utf-8")
                    if path == "/sync":
                        return self._send(200, _page("sync.html"), "text/html; charset=utf-8")
                    if path == "/api/state":
                        return self._json(server.state())
                    if path == "/api/level":
                        a = server.audio
                        return self._json({"level_db": -120.0 if a is None else float(a.level_db),
                                           "muted": bool(a and a.muted)})
                    if path == "/preview.mjpg":
                        return self._mjpeg()
                    if path.startswith("/api/thumb/"):
                        name = unquote(path[len("/api/thumb/"):])
                        return self._send(200, server.bg.thumbnail(name), "image/jpeg")
                    self._error("not found", 404)
                except (FileNotFoundError, ValueError) as exc:
                    self._error(str(exc), 404)

            def do_POST(self):  # noqa: N802
                url = urlparse(self.path)
                path = url.path
                try:
                    if path == "/api/upload":
                        name = parse_qs(url.query).get("name", [""])[0]
                        length = int(self.headers.get("Content-Length") or 0)
                        saved = server.bg.save_upload(name, self.rfile, length)
                        select = parse_qs(url.query).get("select", ["1"])[0] == "1"
                        if select:
                            server.bg.select({"type": "library", "name": saved})
                        return self._json({"name": saved})
                    if path == "/api/background":
                        spec = self._body_json()
                        if spec.get("type") not in ("none", "blur", "color", "library"):
                            return self._error("type must be none, blur, color or library")
                        server.bg.select(spec)
                        return self._json({"ok": True})
                    if path == "/api/delete":
                        server.bg.delete(str(self._body_json().get("name", "")))
                        return self._json({"ok": True})
                    if path == "/api/reseed":
                        server.engine.matter.request_reseed("panel")
                        return self._json({"ok": True})
                    if path == "/api/settings":
                        body = self._body_json()
                        view = body.pop("preview_view", None)
                        if view is not None:
                            if view not in PREVIEW_VIEWS:
                                return self._error("bad preview view")
                            server.preview.view = view
                            server._view_before_sync = None  # the user chose a view, keep it
                        if "internal_size" in body and not 144 <= int(body["internal_size"]) <= 1080:
                            return self._error("internal_size must be between 144 and 1080")
                        if "refine" in body and body["refine"] not in ("guided", "bilinear"):
                            return self._error("refine must be guided or bilinear")
                        allowed = {"internal_size", "refine", "defringe", "mirror"}
                        bad = set(body) - allowed
                        if bad:
                            return self._error(f"cannot change {sorted(bad)} at runtime")
                        if body:
                            server.engine.update_settings(**body)
                        return self._json({"ok": True})
                    if path == "/api/audio":
                        if server.audio is None:
                            return self._error("audio forwarding is off (start with --audio-in/--audio-out)")
                        body = self._body_json()
                        if "muted" in body:
                            server.audio.muted = bool(body.pop("muted"))
                        if body:
                            server.avsync.update(**body)
                        return self._json({"ok": True})
                    if path == "/api/sync/prepare":
                        if server.audio is None:
                            return self._error("the sync test needs audio forwarding (--audio-in/--audio-out)")
                        return self._json(server.sync_prepare())
                    if path == "/api/sync/start":
                        if server.audio is None:
                            return self._error("the sync test needs audio forwarding (--audio-in/--audio-out)")
                        server.sync_prepare()
                        server.calibrator.start()
                        return self._json({"ok": True})
                    if path == "/api/sync/stop":
                        server.calibrator.clear()
                        server.sync_restore()
                        return self._json({"ok": True})
                    if path == "/api/sync/apply":
                        res = server.calibrator.result()
                        if res["offset_ms"] is None:
                            return self._error("no measurement yet")
                        server.avsync.update(cam_offset_ms=res["offset_ms"], mode="auto")
                        server.calibrator.clear()
                        server.sync_restore()
                        return self._json({"ok": True, "cam_offset_ms": res["offset_ms"]})
                    self._error("not found", 404)
                except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                    self._error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    log.exception("request failed")
                    self._error(f"{type(exc).__name__}: {exc}", 500)

            def _mjpeg(self):
                hub = server.preview
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                hub.attach()
                seen = 0
                try:
                    while not server.engine.stop_event.is_set():
                        seq, rgb = hub.slot.get(seen, timeout=1.0)
                        if seq <= seen or rgb is None:
                            continue
                        seen = seq
                        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if not ok:
                            continue
                        data = jpg.tobytes()
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode()
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    hub.detach()
                    self.close_connection = True

        return Handler
