"""Audio/video sync: how long to hold the microphone so it meets the picture.

For something that happens at time T (a clap, a word):

    picture reaches HDMI at  T + L_cam + L_video
    sound reaches HDMI at    T + L_mic + delay + L_sink

    L_cam    camera exposure, encode, USB, decode, until OpenCV hands us the frame
    L_video  our pipeline plus waiting for the display flip (measured every frame)
    L_mic    microphone and capture buffer, until we read the block
    L_sink   playback buffer to the HDMI port (reported by the sound server)

so delay = L_video + (L_cam - L_mic) - L_sink + out_skew.

`L_cam - L_mic` ("camera offset") is the only unknown. It defaults to a
typical webcam value and can be measured with the flash-and-beep test: a phone
shows /sync, which flashes white and beeps at the same instant once a second; we
time the flash in the camera frames and the beep in the mic stream. `out_skew`
covers the capture card showing video a little later than audio (one HDMI
scanout at 60 Hz by default).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from statistics import median
from typing import Callable, Dict, List, Optional

import numpy as np

from mattecast.utils import StateStore

log = logging.getLogger(__name__)

DEFAULTS = {"mode": "auto", "manual_ms": 150.0, "cam_offset_ms": 50.0, "out_skew_ms": 16.0}


class AVSync:
    def __init__(self, store: StateStore, video_latency_ms: Callable[[], Optional[float]]) -> None:
        self.store = store
        self.video_latency_ms = video_latency_ms
        self.sink_latency_ms: Callable[[], float] = lambda: 40.0
        saved = store.get("avsync") or {}
        self.cfg = {k: saved.get(k, v) for k, v in DEFAULTS.items()}
        self._last = self.cfg["manual_ms"]

    def update(self, **kw) -> None:
        for k, v in kw.items():
            if k not in DEFAULTS:
                raise ValueError(f"unknown sync setting {k}")
            if k == "mode":
                if v not in ("auto", "manual"):
                    raise ValueError("mode must be auto or manual")
            else:
                v = float(v)
                if not -500 <= v <= 1500:
                    raise ValueError(f"{k} out of range")
            self.cfg[k] = v
        self.store.set("avsync", self.cfg)

    def breakdown(self) -> Dict:
        video = self.video_latency_ms()
        sink = self.sink_latency_ms()
        auto = None
        if video is not None:
            auto = video + self.cfg["cam_offset_ms"] + self.cfg["out_skew_ms"] - sink
        return {"video_ms": video, "sink_ms": sink, "auto_ms": auto}

    def target_ms(self) -> float:
        if self.cfg["mode"] == "manual":
            return float(self.cfg["manual_ms"])
        auto = self.breakdown()["auto_ms"]
        if auto is None:  # no video measured yet: hold the last value
            return self._last
        # Ignore jitter of a few ms; the bridge converges smoothly within that band anyway.
        if abs(auto - self._last) > 3:
            self._last = max(0.0, auto)
        return self._last

    def state(self) -> Dict:
        b = self.breakdown()
        return {
            **self.cfg,
            "target_ms": round(self.target_ms(), 1),
            "video_ms": None if b["video_ms"] is None else round(b["video_ms"], 1),
            "sink_ms": round(b["sink_ms"], 1),
        }


class SyncCalibrator:
    """Pairs flashes seen by the camera with beeps heard by the mic."""

    TONE_HZ = 1000.0
    RATE = 48000

    def __init__(self, target_pairs: int = 10, timeout: float = 45.0) -> None:
        self.target_pairs = target_pairs
        self.timeout = timeout
        self.active = False
        self._lock = threading.RLock()  # _pair() reports via result() while holding it
        self._reset()

    def _reset(self) -> None:
        self.flashes: deque = deque(maxlen=64)
        self.beeps: deque = deque(maxlen=64)
        self.offsets: List[float] = []
        self._baseline: Optional[np.ndarray] = None
        self._last_flash = -1.0
        self._last_beep = -1.0
        self._floor = None
        self._phase = 0
        self._t0 = time.monotonic()

    def start(self) -> None:
        with self._lock:
            self._reset()
            self.active = True
        log.info("A/V sync test started")

    def stop(self) -> None:
        self.active = False

    def clear(self) -> None:
        with self._lock:
            self.active = False
            self._reset()

    # ---- video: called by the engine with an 8x8 grid of mean luma in [0,1]

    def video_frame(self, t: float, grid: np.ndarray) -> None:
        if not self.active:
            return
        if time.monotonic() - self._t0 > self.timeout:
            self.active = False
            return
        with self._lock:
            if self._baseline is None:
                self._baseline = grid.copy()
                return
            rise = float((grid - self._baseline).max())
            if rise > 0.15 and t - self._last_flash > 0.5:
                self._last_flash = t
                self.flashes.append(t)
                self._pair()
            self._baseline = 0.7 * self._baseline + 0.3 * grid

    # ---- audio: called by the audio bridge for every 10 ms block

    def audio_chunk(self, t_read: float, chunk: np.ndarray, source_latency: float) -> None:
        if not self.active:
            return
        x = chunk.astype(np.float32).mean(axis=1) / 32768.0
        n = len(x)
        k = np.arange(self._phase, self._phase + n)
        self._phase += n
        base = x * np.exp(-2j * np.pi * self.TONE_HZ * k / self.RATE)
        w = 48  # 1 ms smoothing
        env = np.abs(np.convolve(base, np.ones(w) / w, mode="same")) * 2
        with self._lock:
            level = float(np.median(env))
            if self._floor is None:
                self._floor = level
            thr = max(self._floor * 10, 0.01)
            hits = np.nonzero(env > thr)[0]
            if len(hits):
                t_onset = t_read - (n - hits[0]) / self.RATE
                if t_onset - self._last_beep > 0.5:
                    self._last_beep = t_onset
                    self.beeps.append(t_onset)
                    self._pair()
            else:
                self._floor = 0.95 * self._floor + 0.05 * level

    def _pair(self) -> None:
        # Match every flash with the nearest unmatched beep within 350 ms.
        while self.flashes and self.beeps:
            f, b = self.flashes[0], self.beeps[0]
            if abs(f - b) <= 0.35:
                self.offsets.append((f - b) * 1000)
                self.flashes.popleft()
                self.beeps.popleft()
            elif f < b:
                self.flashes.popleft()
            else:
                self.beeps.popleft()
        if len(self.offsets) >= self.target_pairs:
            self.active = False
            log.info("A/V sync test done: %s", self.result())

    def result(self) -> Dict:
        with self._lock:
            offs = list(self.offsets)
        if not offs:
            return {"active": self.active, "pairs": 0, "offset_ms": None, "spread_ms": None}
        med = median(offs)
        spread = median(abs(o - med) for o in offs)
        return {
            "active": self.active,
            "pairs": len(offs),
            "target": self.target_pairs,
            "offset_ms": round(float(med), 1),
            "spread_ms": round(float(spread), 1),
        }
