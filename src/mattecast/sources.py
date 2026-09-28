"""Frame sources: a V4L2 webcam, or a video file for testing."""

from __future__ import annotations

import glob
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from mattecast.utils import LatestSlot

log = logging.getLogger(__name__)


@dataclass
class Frame:
    seq: int
    t: float  # time.monotonic() when the frame was read
    bgr: np.ndarray  # (H, W, 3) uint8


class Source:
    eof = False
    fps: float = 30.0
    size: Tuple[int, int] = (0, 0)

    def get(self, after: int, timeout: float) -> Optional[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _ThreadedSource(Source):
    def __init__(self) -> None:
        self._slot: LatestSlot[Frame] = LatestSlot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)

    def get(self, after: int, timeout: float) -> Optional[Frame]:
        seq, fr = self._slot.get(after, timeout)
        return fr if seq > after else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class CameraSource(_ThreadedSource):
    """Continuously reads a webcam in its own thread, keeping only the newest frame."""

    def __init__(self, device: str, size: Tuple[int, int] = (1920, 1080), fps: float = 30, fourcc: str = "MJPG") -> None:
        super().__init__()
        self.device = int(device) if str(device).isdigit() else device
        self.req = (size, fps, fourcc)
        self.cap = self._open()
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        (w, h), fps, fourcc = self.req
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.device, api)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.device!r}")
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS) or fps
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        cc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) if code else "?"
        self.size, self.fps = (aw, ah), float(afps)
        log.info("camera %s: %dx%d @ %.1f fps, %s", self.device, aw, ah, afps, cc)
        if (aw, ah) != (w, h):
            log.warning("camera gave %dx%d instead of %dx%d", aw, ah, w, h)
        return cap

    def _run(self) -> None:
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails in (1, 30):
                    log.warning("camera read failed (%d)", fails)
                if fails >= 30:
                    time.sleep(1.0)
                    try:
                        self.cap.release()
                        self.cap = self._open()
                        fails = 0
                    except RuntimeError as exc:
                        log.warning("reopen failed: %s", exc)
                else:
                    time.sleep(0.01)
                continue
            fails = 0
            self._slot.put(Frame(0, time.monotonic(), frame))

    def get(self, after: int, timeout: float) -> Optional[Frame]:
        seq, fr = self._slot.get(after, timeout)
        if seq <= after or fr is None:
            return None
        fr.seq = seq
        return fr

    def close(self) -> None:
        super().close()
        self.cap.release()


class FileSource(_ThreadedSource):
    """Plays a video file. realtime=True emulates a camera (paced, frames dropped
    if processing is slow); realtime=False hands out every frame in order."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False) -> None:
        super().__init__()
        self.path = path
        self.realtime = realtime
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._seq = 0
        log.info("file %s: %dx%d @ %.2f fps (%s)", path, *self.size, self.fps, "paced" if realtime else "every frame")
        if realtime:
            self._thread.start()

    def _read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if ok:
            return frame
        if self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                return frame
        self.eof = True
        return None

    def _run(self) -> None:
        period = 1.0 / max(1.0, self.fps)
        next_t = time.monotonic()
        while not self._stop.is_set():
            frame = self._read()
            if frame is None:
                return
            now = time.monotonic()
            if next_t > now:
                time.sleep(next_t - now)
            next_t = max(next_t + period, time.monotonic() - period)
            self._slot.put(Frame(0, time.monotonic(), frame))

    def get(self, after: int, timeout: float) -> Optional[Frame]:
        if self.realtime:
            seq, fr = self._slot.get(after, timeout)
            if seq <= after or fr is None:
                return None
            fr.seq = seq
            return fr
        frame = self._read()
        if frame is None:
            return None
        self._seq += 1
        return Frame(self._seq, time.monotonic(), frame)

    def close(self) -> None:
        if self.realtime:
            super().close()
        self.cap.release()


def list_cameras() -> List[Tuple[str, str]]:
    out = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda p: int("".join(filter(str.isdigit, p)) or 0)):
        name_file = f"/sys/class/video4linux/{os.path.basename(dev)}/name"
        try:
            with open(name_file) as f:
                name = f.read().strip()
        except OSError:
            name = "?"
        out.append((dev, name))
    return out
