"""Forward a microphone to the HDMI output, delayed so it lines up with the video.

The capture card turns HDMI audio into a USB microphone on the call machine, so
sending the webcam's mic down the same cable keeps sound and picture on one
device. The video spends tens of milliseconds in the camera and the matting
pipeline, so the audio has to wait the same amount (see avsync.py).

Audio goes through libpulse's simple API, which PipeWire serves too, so any
source or sink listed by `pactl` works. Microphone and HDMI run on separate
clocks; a small controller keeps the buffered delay on target by occasionally
stretching or squeezing a 10 ms block by one sample (0.21 %, about 2 ms of
correction per second), which is inaudible and far more than any realistic
clock drift. Bigger changes, like a new delay setting, jump directly.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

RATE = 48000
CHANNELS = 2
CHUNK = 480  # 10 ms
_U32 = 0xFFFFFFFF


# ----- libpulse-simple via ctypes ------------------------------------------------


class _SampleSpec(ctypes.Structure):
    _fields_ = [("format", ctypes.c_int), ("rate", ctypes.c_uint32), ("channels", ctypes.c_uint8)]


class _BufferAttr(ctypes.Structure):
    _fields_ = [
        ("maxlength", ctypes.c_uint32),
        ("tlength", ctypes.c_uint32),
        ("prebuf", ctypes.c_uint32),
        ("minreq", ctypes.c_uint32),
        ("fragsize", ctypes.c_uint32),
    ]


_lib = None


def _pulse():
    global _lib
    if _lib is None:
        name = ctypes.util.find_library("pulse-simple") or "libpulse-simple.so.0"
        lib = ctypes.CDLL(name)
        lib.pa_simple_new.restype = ctypes.c_void_p
        lib.pa_simple_new.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.POINTER(_SampleSpec), ctypes.c_void_p, ctypes.POINTER(_BufferAttr), ctypes.POINTER(ctypes.c_int),
        ]
        lib.pa_simple_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        lib.pa_simple_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
        lib.pa_simple_get_latency.restype = ctypes.c_uint64
        lib.pa_simple_get_latency.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.pa_simple_free.argtypes = [ctypes.c_void_p]
        lib.pa_strerror.restype = ctypes.c_char_p
        lib.pa_strerror.argtypes = [ctypes.c_int]
        _lib = lib
    return _lib


class PulseStream:
    """A blocking 48 kHz stereo s16 stream to one named source or sink."""

    PLAYBACK, RECORD = 1, 2

    def __init__(self, direction: int, device: str, name: str, latency_ms: float) -> None:
        lib = _pulse()
        self._lib = lib
        spec = _SampleSpec(3, RATE, CHANNELS)  # PA_SAMPLE_S16LE
        nbytes = int(RATE * latency_ms / 1000) * CHANNELS * 2
        if direction == self.PLAYBACK:
            attr = _BufferAttr(_U32, nbytes, _U32, _U32, _U32)
        else:
            attr = _BufferAttr(_U32, _U32, _U32, _U32, nbytes)
        err = ctypes.c_int(0)
        self._h = lib.pa_simple_new(
            None, b"mattecast", direction, device.encode(), name.encode(),
            ctypes.byref(spec), None, ctypes.byref(attr), ctypes.byref(err),
        )
        if not self._h:
            raise RuntimeError(f"cannot open {device}: {lib.pa_strerror(err.value).decode()}")

    def _check(self, rc: int, err: ctypes.c_int, what: str) -> None:
        if rc < 0:
            raise RuntimeError(f"{what}: {self._lib.pa_strerror(err.value).decode()}")

    def read(self, frames: int) -> np.ndarray:
        buf = np.empty((frames, CHANNELS), dtype=np.int16)
        err = ctypes.c_int(0)
        self._check(self._lib.pa_simple_read(self._h, buf.ctypes.data, buf.nbytes, ctypes.byref(err)), err, "read")
        return buf

    def write(self, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data, dtype=np.int16)
        err = ctypes.c_int(0)
        self._check(self._lib.pa_simple_write(self._h, data.ctypes.data, data.nbytes, ctypes.byref(err)), err, "write")

    def latency(self) -> float:
        """Seconds of audio between the device and this process."""
        err = ctypes.c_int(0)
        usec = self._lib.pa_simple_get_latency(self._h, ctypes.byref(err))
        return usec / 1e6

    def close(self) -> None:
        if self._h:
            self._lib.pa_simple_free(self._h)
            self._h = None


# ----- device lookup ----------------------------------------------------------------


def list_pulse(kind: str) -> List[Tuple[str, str]]:
    """[(name, state)] of `pactl list short sources|sinks`."""
    if not shutil.which("pactl"):
        raise RuntimeError("pactl not found (sudo apt install pulseaudio-utils)")
    out = subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            rows.append((parts[1], parts[-1]))
    return rows


def _activate_hdmi_profile(sink: str) -> bool:
    """GPUs expose one HDMI/DP port at a time as a card profile. If the wanted sink
    (alsa_output.<card>.<profile>) is missing, switch the card to that profile."""
    if not sink.startswith("alsa_output."):
        return False
    card_id, _, profile = sink[len("alsa_output."):].rpartition(".")
    if not card_id or not profile:
        return False
    rc = subprocess.run(
        ["pactl", "set-card-profile", f"alsa_card.{card_id}", f"output:{profile}"], capture_output=True, text=True
    )
    if rc.returncode == 0:
        log.info("switched alsa_card.%s to profile output:%s", card_id, profile)
        time.sleep(0.5)
        return True
    return False


def resolve_device(kind: str, query: str) -> str:
    """Exact name, or a unique case-insensitive substring of one."""
    names = [n for n, _ in list_pulse(kind)]
    if kind == "sinks" and query not in names and _activate_hdmi_profile(query):
        names = [n for n, _ in list_pulse(kind)]
    if query in names:
        return query
    if kind == "sources":
        names = [n for n in names if not n.endswith(".monitor")] or names
    hits = [n for n in names if query.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    listing = "\n  ".join(names) or "(none)"
    what = "no" if not hits else "several"
    raise SystemExit(f"{what} {kind} match {query!r}. Available:\n  {listing}")


# ----- keep the mic on exactly the devices we chose -----------------------------------------

STREAM_NAME = f"mattecast mic to HDMI ({os.getpid()})"
SOURCE_STREAM_NAME = f"mattecast webcam mic ({os.getpid()})"
_KINDS = {
    # stream list        header            device key   device list
    "sink-inputs": ("Sink Input #", "Sink:", "sinks"),
    "source-outputs": ("Source Output #", "Source:", "sources"),
}


def _device_index(kind: str, name: str) -> Optional[str]:
    out = subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == name:
            return parts[0]
    return None


def _our_streams(kind: str, media_name: str) -> List[Tuple[str, str]]:
    """[(stream id, device index)] for our stream called `media_name`."""
    header, key, _ = _KINDS[kind]
    out = subprocess.run(["pactl", "list", kind], capture_output=True, text=True).stdout
    found, sid, dev = [], None, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith(header):
            sid, dev = s[len(header):], None
        elif s.startswith(key) and sid is not None:
            dev = s.split(":", 1)[1].strip()
        elif s.startswith("media.name = ") and sid is not None:
            if s[len("media.name = "):].strip('"') == media_name:
                found.append((sid, dev or ""))
    return found


def _our_sink_inputs() -> List[Tuple[str, str]]:
    return _our_streams("sink-inputs", STREAM_NAME)


def _sink_index(name: str) -> Optional[str]:
    return _device_index("sinks", name)


class RouteGuard:
    """PipeWire (and PulseAudio's rescue-streams module) move a stream to the default
    device when its own device disappears, e.g. while the HDMI link retrains after a
    mode change. For a live microphone that means your own voice coming out of the
    speakers or, with Sunshine as the default output, out of Moonlight; on the input
    side it would silently forward some other microphone. The guard watches
    `pactl subscribe` and has the bridge drop a stream the moment it sits anywhere
    but the chosen device; the bridge reconnects once that device is back."""

    def __init__(self, sink: str, source: str, drop_output: Callable[[], None], drop_input: Callable[[], None]) -> None:
        self.checks = [
            ("sink-inputs", STREAM_NAME, sink, drop_output),
            ("source-outputs", SOURCE_STREAM_NAME, source, drop_input),
        ]
        self.kills = 0
        self._last_warn = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._threads = [
            threading.Thread(target=self._listen, name="audio-guard-events", daemon=True),
            threading.Thread(target=self._loop, name="audio-guard", daemon=True),
        ]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._proc is not None:
            self._proc.terminate()

    def _listen(self) -> None:
        try:
            self._proc = subprocess.Popen(["pactl", "subscribe"], stdout=subprocess.PIPE, text=True)
        except OSError as exc:
            log.warning("cannot watch audio routing (%s); checking every 2 s instead", exc)
            return
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            if any(k in line for k in ("sink", "source", "card", "server")):
                self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(2.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.check()
            except Exception as exc:  # noqa: BLE001
                log.debug("audio guard check failed: %s", exc)

    def check(self) -> None:
        for kind, media, device, drop in self.checks:
            target = _device_index(_KINDS[kind][2], device)
            for _sid, dev in _our_streams(kind, media):
                if dev and dev == target:
                    continue
                # Closing the stream from our side drops whatever it had buffered on the
                # wrong device at once, and does not touch saved per-app mute or volume.
                drop()
                self.kills += 1
                now = time.monotonic()
                if now - self._last_warn > 10:
                    self._last_warn = now
                    log.warning("audio stream was moved off %s (to #%s); disconnected it so nothing plays or "
                                "records on another device, reconnecting when it is back", device, dev or "?")
                break


# ----- the bridge -------------------------------------------------------------------


class _Fifo:
    """Frames (N, 2) int16 in arrival order, with cheap pop/drop/pad."""

    def __init__(self) -> None:
        self._q: deque = deque()
        self.count = 0

    def push(self, a: np.ndarray) -> None:
        self._q.append(a)
        self.count += len(a)

    def push_front(self, a: np.ndarray) -> None:
        self._q.appendleft(a)
        self.count += len(a)

    def pop(self, n: int) -> Tuple[np.ndarray, int]:
        """Up to n frames; returns (frames, missing)."""
        parts, need = [], n
        while need > 0 and self._q:
            head = self._q[0]
            if len(head) <= need:
                parts.append(self._q.popleft())
                need -= len(head)
            else:
                parts.append(head[:need])
                self._q[0] = head[need:]
                need = 0
        got = n - need
        self.count -= got
        out = np.concatenate(parts) if parts else np.zeros((0, CHANNELS), np.int16)
        if need:
            out = np.concatenate([out, np.zeros((need, CHANNELS), np.int16)])
        return out, need

    def drop(self, n: int) -> None:
        self.pop(min(n, self.count))


def _stretch(frames: np.ndarray, n: int) -> np.ndarray:
    """Linear resample (m, 2) -> (n, 2)."""
    m = len(frames)
    if m == n or m < 2:
        return frames[:n] if m >= n else np.concatenate([frames, np.zeros((n - m, CHANNELS), np.int16)])
    xo = np.linspace(0.0, 1.0, m)
    xn = np.linspace(0.0, 1.0, n)
    f = frames.astype(np.float32)
    out = np.stack([np.interp(xn, xo, f[:, c]) for c in range(CHANNELS)], axis=1)
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class AudioBridge:
    """mic -> delay line -> HDMI. `target_ms` is polled every block, so the delay
    follows the video latency live."""

    TOL = int(0.004 * RATE)  # inside this band, leave it alone
    JUMP = int(0.015 * RATE)  # beyond this (start-up, a new delay setting), jump straight there
    MAX_BUFFER = 2 * RATE

    def __init__(
        self,
        open_source: Callable[[], object],
        open_sink: Callable[[], object],
        target_ms: Callable[[], float],
        on_chunk: Optional[Callable[[float, np.ndarray, float], None]] = None,
        names: Tuple[str, str] = ("", ""),
    ) -> None:
        self._open_source = open_source
        self._open_sink = open_sink
        self.target_ms = target_ms
        self.on_chunk = on_chunk
        self.names = names
        self.muted = False
        self._fifo = _Fifo()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fill_ema: Optional[float] = None
        self.sink_latency = 0.04
        self.source_latency = 0.01
        self.level_db = -120.0
        self._peak = 0
        self._peak_t = time.monotonic()
        self.underruns = 0
        self.corrections = 0
        self.errors = 0
        self.running = False
        self.guard: Optional[RouteGuard] = None
        self._drop_sink = threading.Event()
        self._drop_source = threading.Event()
        self._threads = [
            threading.Thread(target=self._reader, name="audio-in", daemon=True),
            threading.Thread(target=self._writer, name="audio-out", daemon=True),
        ]

    def start(self) -> None:
        self.running = True
        for t in self._threads:
            t.start()
        if self.guard is not None:
            self.guard.start()

    def close(self) -> None:
        self._stop.set()
        if self.guard is not None:
            self.guard.close()
        for t in self._threads:
            t.join(timeout=2)
        self.running = False

    def drop_output(self) -> None:
        """Disconnect the output stream now and reconnect when possible."""
        self._drop_sink.set()

    def drop_input(self) -> None:
        self._drop_source.set()

    # ---- threads

    def _reopen(self, opener, what: str):
        tries = 0
        while not self._stop.is_set():
            try:
                stream = opener()
                if tries:
                    log.info("audio %s reconnected", what)
                return stream
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                if tries % 15 == 0:  # first failure, then every ~30 s
                    log.warning("audio %s unavailable (%s); retrying", what, exc)
                tries += 1
                if self._stop.wait(2.0):
                    return None
        return None

    def _reader(self) -> None:
        src = self._reopen(self._open_source, "input")
        n = 0
        while src is not None and not self._stop.is_set():
            if self._drop_source.is_set():
                self._drop_source.clear()
                _close(src)
                src = self._reopen(self._open_source, "input")
                continue
            try:
                chunk = src.read(CHUNK)
                t = time.monotonic()
                n += 1
                if n % 50 == 0:
                    self.source_latency = src.latency()
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                log.warning("audio input failed: %s", exc)
                _close(src)
                src = self._reopen(self._open_source, "input")
                continue
            if self._drop_source.is_set():
                continue  # read from a device we are about to leave: discard it
            self._meter(chunk, t)
            if self.on_chunk is not None:
                try:
                    self.on_chunk(t, chunk, self.source_latency)
                except Exception:  # noqa: BLE001
                    log.exception("audio analysis failed")
            with self._lock:
                self._fifo.push(chunk)
                if self._fifo.count > self.MAX_BUFFER:
                    self._fifo.drop(self._fifo.count - self.MAX_BUFFER)
        _close(src)

    def _meter(self, chunk: np.ndarray, t: float) -> None:
        self._peak = max(self._peak, int(np.abs(chunk.astype(np.int32)).max()))
        if t - self._peak_t >= 0.1:
            self.level_db = 20 * np.log10(max(self._peak, 1) / 32768.0)
            self._peak = 0
            self._peak_t = t

    def _writer(self) -> None:
        sink = self._reopen(self._open_sink, "output")
        n = 0
        while sink is not None and not self._stop.is_set():
            if self._drop_sink.is_set():
                self._drop_sink.clear()
                _close(sink)
                with self._lock:
                    self._fill_ema = None  # the new stream starts with an empty buffer
                sink = self._reopen(self._open_sink, "output")
                continue
            target = int(max(0.0, min(1500.0, self.target_ms())) * RATE / 1000)
            with self._lock:
                fill = self._fifo.count
                ema = fill if self._fill_ema is None else 0.95 * self._fill_ema + 0.05 * fill
                err = ema - target
                if abs(err) > self.JUMP:
                    if err < 0:
                        self._fifo.push_front(np.zeros((int(-err), CHANNELS), np.int16))
                    else:
                        self._fifo.drop(int(err))
                    ema = target
                    err = 0
                self._fill_ema = ema
                take = CHUNK + (1 if err > self.TOL else -1 if err < -self.TOL else 0)
                frames, missing = self._fifo.pop(take)
            if missing:
                self.underruns += 1
            if take != CHUNK:
                frames = _stretch(frames, CHUNK)
                self.corrections += 1
            if self.muted:
                frames = np.zeros_like(frames)
            try:
                sink.write(frames)
                n += 1
                if n % 50 == 0:
                    self.sink_latency = sink.latency()
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                log.warning("audio output failed: %s", exc)
                _close(sink)
                sink = self._reopen(self._open_sink, "output")
        _close(sink)

    # ---- reporting

    def buffered_ms(self) -> float:
        ema = self._fill_ema
        return 0.0 if ema is None else ema * 1000 / RATE

    def stats(self) -> dict:
        return {
            "input": self.names[0],
            "output": self.names[1],
            "muted": self.muted,
            "level_db": round(float(self.level_db), 1),
            "buffer_ms": round(self.buffered_ms(), 1),
            "sink_ms": round(self.sink_latency * 1000, 1),
            "source_ms": round(self.source_latency * 1000, 1),
            "underruns": self.underruns,
            "errors": self.errors,
            "blocked_moves": 0 if self.guard is None else self.guard.kills,
        }


def _close(stream) -> None:
    if stream is not None:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def pulse_bridge(source: str, sink: str, target_ms, on_chunk=None) -> AudioBridge:
    def open_sink() -> PulseStream:
        # Never create the stream while the capture card's output is missing: the
        # sound server would put it on the default output instead.
        names = [n for n, _ in list_pulse("sinks")]
        if sink not in names:
            _activate_hdmi_profile(sink)
            names = [n for n, _ in list_pulse("sinks")]
            if sink not in names:
                raise RuntimeError(f"{sink} is not available")
        return PulseStream(PulseStream.PLAYBACK, sink, STREAM_NAME, latency_ms=40)

    def open_source() -> PulseStream:
        if source not in [n for n, _ in list_pulse("sources")]:
            raise RuntimeError(f"{source} is not available")
        return PulseStream(PulseStream.RECORD, source, SOURCE_STREAM_NAME, latency_ms=10)

    bridge = AudioBridge(
        open_source,
        open_sink,
        target_ms,
        on_chunk,
        names=(source, sink),
    )
    bridge.guard = RouteGuard(sink, source, bridge.drop_output, bridge.drop_input)
    return bridge
