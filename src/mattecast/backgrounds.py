"""Backgrounds: original camera, blur, solid color, still image, looping GIF, looping video.

Every background renders a (3, H, W) float tensor on the pipeline device for a
given wall-clock time, so animated ones play at their own speed regardless of
the camera frame rate. Uploaded files live in a library folder and can be
switched at runtime from the control panel.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence

from mattecast.refine import cover_resize, gaussian_blur
from mattecast.utils import LatestSlot, StateStore

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ANIM_EXT = {".gif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
ALLOWED_EXT = IMAGE_EXT | ANIM_EXT | VIDEO_EXT
MAX_ANIM_FRAMES = 900


def _rgb_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    return t.permute(2, 0, 1).float().div_(255.0)


def _hex_to_rgb(text: str) -> Tuple[int, int, int]:
    text = text.strip()
    if text.startswith("#"):
        text = text[1:]
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    r, g, b = (int(v) for v in text.split(","))
    return r, g, b


def file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in ANIM_EXT:
        return "animation"
    if ext in {".webp", ".png"}:
        try:
            with Image.open(path) as im:
                if getattr(im, "is_animated", False) and getattr(im, "n_frames", 1) > 1:
                    return "animation"
        except Exception:  # noqa: BLE001
            pass
    return "image"


class Background:
    kind = "base"

    def __init__(self, size_hw: Tuple[int, int], device: torch.device) -> None:
        self.H, self.W = size_hw
        self.device = device

    def frame(self, now: float, src: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NoneBackground(Background):
    """Effect off: the pipeline sends the camera image untouched."""

    kind = "none"

    def frame(self, now, src, alpha):
        return src


class ColorBackground(Background):
    kind = "color"

    def __init__(self, size_hw, device, color: str = "#00b140") -> None:
        super().__init__(size_hw, device)
        r, g, b = _hex_to_rgb(color)
        self._t = torch.tensor([r, g, b], device=device, dtype=torch.float32).div(255).view(3, 1, 1).expand(3, self.H, self.W)

    def frame(self, now, src, alpha):
        return self._t


class BlurBackground(Background):
    """Blurs the real room with the subject masked out, so the person does not
    smear into the blur and leave a glow around their silhouette."""

    kind = "blur"

    def __init__(self, size_hw, device, strength: float = 12.0) -> None:
        super().__init__(size_hw, device)
        self.strength = float(strength)

    def frame(self, now, src, alpha):
        f = 4
        h, w = max(1, self.H // f), max(1, self.W // f)
        small = F.interpolate(src[None].float(), size=(h, w), mode="area")
        a = F.interpolate(alpha[None, None].float(), size=(h, w), mode="area")
        a = F.max_pool2d(a, 5, 1, 2)  # grow the subject a little to swallow edge pixels
        wgt = 1.0 - a
        num = gaussian_blur(small * wgt, self.strength)
        den = gaussian_blur(wgt, self.strength)
        plain = gaussian_blur(small, self.strength)
        mix = (den / 0.25).clamp(0, 1)
        out = mix * (num / den.clamp_min(1e-4)) + (1 - mix) * plain
        return F.interpolate(out, size=(self.H, self.W), mode="bilinear", align_corners=False)[0]


class ImageBackground(Background):
    kind = "image"

    def __init__(self, size_hw, device, path: Path) -> None:
        super().__init__(size_hw, device)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((4096, 4096))
            rgb = np.asarray(im)
        self._t = cover_resize(_rgb_to_tensor(rgb, device), (self.H, self.W)).contiguous()

    def frame(self, now, src, alpha):
        return self._t


class AnimatedBackground(Background):
    """GIF / animated WebP / APNG, looped with the file's own frame timing."""

    kind = "animation"

    def __init__(self, size_hw, device, path: Path) -> None:
        super().__init__(size_hw, device)
        frames: List[np.ndarray] = []
        durations: List[float] = []
        with Image.open(path) as im:
            for i, fr in enumerate(ImageSequence.Iterator(im)):
                if i >= MAX_ANIM_FRAMES:
                    log.warning("%s: keeping the first %d frames", path.name, MAX_ANIM_FRAMES)
                    break
                d = fr.info.get("duration", 100) or 100
                if d < 20:  # browsers treat tiny delays as 100 ms; so do we
                    d = 100
                rgb = fr.convert("RGB")
                if rgb.width > 2 * self.W or rgb.height > 2 * self.H:
                    rgb.thumbnail((2 * self.W, 2 * self.H))
                frames.append(np.asarray(rgb).copy())
                durations.append(d / 1000.0)
        if not frames:
            raise ValueError("no frames in animation")
        self._frames = frames
        self._ends = np.cumsum(durations)
        self._total = float(self._ends[-1])
        self._t0 = time.monotonic()
        self._idx = -1
        self._cur: Optional[torch.Tensor] = None

    def frame(self, now, src, alpha):
        t = (now - self._t0) % self._total
        idx = int(np.searchsorted(self._ends, t, side="right"))
        idx = min(idx, len(self._frames) - 1)
        if idx != self._idx or self._cur is None:
            self._cur = cover_resize(_rgb_to_tensor(self._frames[idx], self.device), (self.H, self.W)).contiguous()
            self._idx = idx
        return self._cur


class VideoBackground(Background):
    """Loops a video file. Decoding and resizing run in a thread at the video's
    own frame rate; the pipeline just grabs whatever frame is current."""

    kind = "video"

    def __init__(self, size_hw, device, path: Path) -> None:
        super().__init__(size_hw, device)
        self.path = path
        cap = cv2.VideoCapture(str(path))
        ok, first = cap.read() if cap.isOpened() else (False, None)
        if not ok:
            cap.release()
            raise ValueError(f"cannot decode video {path.name}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.fps = float(min(max(fps, 1.0), 120.0))
        self._cap = cap
        self._slot: LatestSlot[np.ndarray] = LatestSlot()
        self._slot.put(self._fit(first))
        self._seen = -1
        self._cur: Optional[torch.Tensor] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="video-bg", daemon=True)
        self._thread.start()

    def _fit(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = max(self.H / h, self.W / w)
        nh, nw = max(self.H, round(h * s)), max(self.W, round(w * s))
        if (nh, nw) != (h, w):
            bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        y0, x0 = (nh - self.H) // 2, (nw - self.W) // 2
        return np.ascontiguousarray(bgr[y0 : y0 + self.H, x0 : x0 + self.W, ::-1])  # to RGB

    def _run(self) -> None:
        period = 1.0 / self.fps
        next_t = time.monotonic() + period
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(str(self.path))
                    ok, frame = self._cap.read()
                    if not ok:
                        log.error("video background %s stopped decoding", self.path.name)
                        return
            rgb = self._fit(frame)
            now = time.monotonic()
            if next_t > now:
                if self._stop.wait(next_t - now):
                    return
            elif now - next_t > 0.5:  # fell far behind (e.g. system hiccup): resync
                next_t = now
            next_t += period
            self._slot.put(rgb)

    def frame(self, now, src, alpha):
        seq, rgb = self._slot.peek()
        if seq != self._seen or self._cur is None:
            self._cur = _rgb_to_tensor(rgb, self.device)
            self._seen = seq
        return self._cur

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class BackgroundManager:
    """Owns the active background and the uploaded-file library."""

    def __init__(self, library: Path, size_hw: Tuple[int, int], device: torch.device, store: StateStore) -> None:
        self.library = library
        self.library.mkdir(parents=True, exist_ok=True)
        self.size_hw = size_hw
        self.device = device
        self.store = store
        self._lock = threading.Lock()
        self._current: Background = BlurBackground(size_hw, device)
        self.spec: Dict = {"type": "blur", "strength": 12}
        self._thumbs: Dict[Tuple[str, float], bytes] = {}

    # ----- selection --------------------------------------------------------

    def parse_cli(self, text: str) -> Dict:
        low = text.strip().lower()
        if low in ("none", "original", "off"):
            return {"type": "none"}
        if low == "blur" or low.startswith("blur:"):
            strength = float(low.split(":", 1)[1]) if ":" in low else 12.0
            return {"type": "blur", "strength": strength}
        if low.startswith("color:"):
            return {"type": "color", "color": text.split(":", 1)[1]}
        p = Path(text).expanduser()
        if p.is_file():
            return {"type": "path", "path": str(p.resolve())}
        if (self.library / text).is_file():
            return {"type": "library", "name": text}
        raise ValueError(f"background {text!r} is not blur/color:/none, a file, or a library item")

    def build(self, spec: Dict) -> Background:
        kind = spec.get("type")
        if kind == "none":
            return NoneBackground(self.size_hw, self.device)
        if kind == "blur":
            return BlurBackground(self.size_hw, self.device, float(spec.get("strength", 12)))
        if kind == "color":
            return ColorBackground(self.size_hw, self.device, str(spec.get("color", "#00b140")))
        if kind == "library":
            path = self._library_path(str(spec.get("name", "")))
        elif kind == "path":
            path = Path(str(spec["path"]))
        else:
            raise ValueError(f"unknown background type {kind!r}")
        if not path.is_file():
            raise FileNotFoundError(path.name)
        k = file_kind(path)
        if k == "video":
            return VideoBackground(self.size_hw, self.device, path)
        if k == "animation":
            return AnimatedBackground(self.size_hw, self.device, path)
        return ImageBackground(self.size_hw, self.device, path)

    def select(self, spec: Dict, persist: bool = True) -> None:
        bg = self.build(spec)  # may take a while (decoding a GIF); do it outside the lock
        with self._lock:
            old, self._current, self.spec = self._current, bg, dict(spec)
        old.close()
        log.info("background -> %s", self.describe())
        if persist and spec.get("type") != "path":
            self.store.set("background", spec)

    def restore(self) -> Optional[Dict]:
        return self.store.get("background")

    def frame(self, now: float, src: torch.Tensor, alpha: torch.Tensor) -> Tuple[str, torch.Tensor]:
        with self._lock:
            bg = self._current
        return bg.kind, bg.frame(now, src, alpha)

    def fallback(self) -> None:
        """Called when the active background throws: drop to blur so the call goes on."""
        with self._lock:
            old, self._current, self.spec = self._current, BlurBackground(self.size_hw, self.device), {"type": "blur", "strength": 12}
        old.close()

    def describe(self) -> str:
        s = self.spec
        if s.get("type") in ("library",):
            return s.get("name", "?")
        if s.get("type") == "path":
            return Path(s["path"]).name
        return s.get("type", "?")

    def close(self) -> None:
        with self._lock:
            self._current.close()

    # ----- library ----------------------------------------------------------

    def _library_path(self, name: str) -> Path:
        p = (self.library / name).resolve()
        if p.parent != self.library.resolve():
            raise ValueError("bad name")
        return p

    def list(self) -> List[Dict]:
        items = []
        for p in sorted(self.library.iterdir(), key=lambda q: q.stat().st_mtime, reverse=True):
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in ALLOWED_EXT:
                items.append({"name": p.name, "kind": file_kind(p), "bytes": p.stat().st_size})
        return items

    def save_upload(self, filename: str, stream, length: int, max_bytes: int = 1 << 30) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise ValueError(f"unsupported file type {ext or '(none)'}")
        if length <= 0 or length > max_bytes:
            raise ValueError("file is empty or too large")
        stem = _NAME_RE.sub("_", Path(filename).stem).strip("._")[:60] or "background"
        name = f"{stem}{ext}"
        n = 1
        while (self.library / name).exists():
            name = f"{stem}-{n}{ext}"
            n += 1
        tmp = self.library / f".upload-{time.time_ns()}{ext}"
        try:
            remaining = length
            with open(tmp, "wb") as f:
                while remaining > 0:
                    chunk = stream.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ValueError("upload was cut off")
                    f.write(chunk)
                    remaining -= len(chunk)
            self._validate(tmp)
            tmp.rename(self.library / name)
        finally:
            tmp.unlink(missing_ok=True)
        return name

    @staticmethod
    def _validate(path: Path) -> None:
        if path.suffix.lower() in VIDEO_EXT:
            cap = cv2.VideoCapture(str(path))
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if not ok:
                raise ValueError("could not decode that video")
            return
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:  # noqa: BLE001
            raise ValueError("not a readable image file") from exc

    def delete(self, name: str) -> None:
        p = self._library_path(name)
        if self.spec.get("type") == "library" and self.spec.get("name") == name:
            raise ValueError("that background is in use; switch to another one first")
        p.unlink()

    def thumbnail(self, name: str, width: int = 320) -> bytes:
        p = self._library_path(name)
        key = (name, p.stat().st_mtime)
        if key in self._thumbs:
            return self._thumbs[key]
        if p.suffix.lower() in VIDEO_EXT:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n // 10)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError("cannot read video")
            im = Image.fromarray(frame[:, :, ::-1])
        else:
            with Image.open(p) as src:
                im = ImageOps.exif_transpose(src).convert("RGB")
        im.thumbnail((width, width))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        data = buf.getvalue()
        self._thumbs[key] = data
        return data
