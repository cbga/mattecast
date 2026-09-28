"""Small threading and statistics helpers shared across the pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """A one-item mailbox. Writers overwrite; readers wait for something newer.

    Every stage in the pipeline only ever wants the freshest item, so there is no
    queue to build up latency when a consumer falls behind.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: Optional[T] = None
        self._seq = 0

    def put(self, item: T) -> int:
        with self._cond:
            self._item = item
            self._seq += 1
            self._cond.notify_all()
            return self._seq

    def get(self, after: int = -1, timeout: Optional[float] = None) -> Tuple[int, Optional[T]]:
        """Return (seq, item). If nothing newer than `after` arrives in time, seq <= after."""
        with self._cond:
            if self._seq <= after:
                self._cond.wait_for(lambda: self._seq > after, timeout)
            return self._seq, self._item

    def peek(self) -> Tuple[int, Optional[T]]:
        with self._cond:
            return self._seq, self._item


class Rolling:
    """Rolling window of samples with percentile readout."""

    def __init__(self, size: int = 240) -> None:
        self._v: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()

    def add(self, x: float) -> None:
        with self._lock:
            self._v.append(x)

    def pct(self, p: float) -> Optional[float]:
        with self._lock:
            if not self._v:
                return None
            s = sorted(self._v)
        k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
        return s[k]


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window: float = 2.0) -> None:
        self._t: deque[float] = deque()
        self._window = window
        self._lock = threading.Lock()

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._t.append(now)
            while self._t and now - self._t[0] > self._window:
                self._t.popleft()

    def rate(self) -> float:
        now = time.monotonic()
        with self._lock:
            while self._t and now - self._t[0] > self._window:
                self._t.popleft()
            if len(self._t) < 2:
                return 0.0
            span = self._t[-1] - self._t[0]
            return (len(self._t) - 1) / span if span > 0 else 0.0


def _app_dir(env: str, xdg: str, fallback: str) -> Path:
    """~/.cache/mattecast or ~/.local/share/mattecast. Moves the folder over from the
    project's old name (remote-cam) the first time, so downloads and uploads survive."""
    root = Path(os.environ.get(xdg) or os.path.expanduser(fallback))
    override = os.environ.get(env)
    p = Path(override) if override else root / "mattecast"
    old = root / "remote-cam"
    if not override and not p.exists() and old.is_dir():
        try:
            old.rename(p)
        except OSError:
            pass
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    return _app_dir("MATTECAST_CACHE", "XDG_CACHE_HOME", "~/.cache")


def data_dir() -> Path:
    return _app_dir("MATTECAST_DATA", "XDG_DATA_HOME", "~/.local/share")


def parse_size(text: str) -> Tuple[int, int]:
    """'1920x1080' -> (1920, 1080)."""
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:  # pragma: no cover - argparse reports it
        raise ValueError(f"expected WIDTHxHEIGHT, got {text!r}") from exc


class StateStore:
    """Tiny JSON file for settings that should survive restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self._data = json.loads(path.read_text())
            if not isinstance(self._data, dict):
                self._data = {}
        except (OSError, ValueError):
            self._data = {}

    def get(self, key: str, default=None):
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            try:
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(self._data, indent=1))
                tmp.replace(self.path)
            except OSError:
                pass
