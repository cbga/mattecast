"""The processing thread: camera frame in, composited frame out.

    camera BGR (H,W) -> GPU -> crop/resize to output -> downscale to the model's
    internal size -> MatAnyone 2 alpha -> guided upsample to output size ->
    foreground color estimation -> composite over background -> uint8 -> sinks
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from mattecast.backgrounds import BackgroundManager
from mattecast.matter import StreamingMatter
from mattecast.refine import (
    bilinear_upsample,
    cover_resize,
    estimate_foreground,
    guided_upsample,
    resize_short_side,
)
from mattecast.sinks import FrameOut, Sink
from mattecast.sources import Frame, Source
from mattecast.utils import LatestSlot, RateMeter, Rolling

log = logging.getLogger(__name__)

INTERNAL_SIZES = (288, 360, 432, 540, 720)
PREVIEW_VIEWS = ("composite", "alpha", "camera")


@dataclass
class Settings:
    internal_size: int = 540
    refine: str = "guided"  # guided | bilinear
    gf_radius: int = 2
    gf_eps: float = 1e-3
    defringe: bool = True
    alpha_lo: float = 0.02
    alpha_hi: float = 0.98
    mirror: bool = False


class PreviewHub:
    """Low-res preview for the control panel. Only rendered while someone watches."""

    def __init__(self, width: int = 640, max_fps: float = 15.0) -> None:
        self.width = width
        self.min_dt = 1.0 / max_fps
        self.slot: LatestSlot = LatestSlot()
        self.view = "composite"
        self._clients = 0
        self._lock = threading.Lock()
        self._last = 0.0

    def attach(self) -> None:
        with self._lock:
            self._clients += 1

    def detach(self) -> None:
        with self._lock:
            self._clients -= 1

    def wanted(self, now: float) -> Optional[str]:
        if self._clients <= 0 or now - self._last < self.min_dt:
            return None
        self._last = now
        return self.view


class Engine(threading.Thread):
    def __init__(
        self,
        source: Source,
        matter: StreamingMatter,
        backgrounds: BackgroundManager,
        sinks: List[Sink],
        preview: Optional[PreviewHub],
        settings: Settings,
        out_size: Tuple[int, int],
        device: torch.device,
        precision: str = "fp16",
        max_frames: int = 0,
        profile: bool = False,
        stop: Optional[threading.Event] = None,
        calibrator=None,
    ) -> None:
        super().__init__(name="engine", daemon=True)
        self.source = source
        self.matter = matter
        self.backgrounds = backgrounds
        self.sinks = sinks
        self.preview = preview
        self.settings = settings
        self.W, self.H = out_size
        self.device = device
        self.max_frames = max_frames
        self.profile = profile
        self.stop_event = stop or threading.Event()
        self.calibrator = calibrator
        self.done = threading.Event()
        self._tasks: "queue.SimpleQueue[Callable[[Engine], None]]" = queue.SimpleQueue()
        self._last_good: Optional[FrameOut] = None
        self._errors = 0

        if device.type == "cuda" and precision in ("fp16", "bf16"):
            dtype = torch.float16 if precision == "fp16" else torch.bfloat16
            self._amp = lambda: torch.autocast("cuda", dtype=dtype)
        else:
            self._amp = contextlib.nullcontext

        self.frames = 0
        self.fps = RateMeter()
        self.t_proc = Rolling()
        self.t_latency = Rolling()
        self.stage: Dict[str, Rolling] = {k: Rolling() for k in ("prep", "matte", "refine", "compose", "download")}
        self.internal_hw: Tuple[int, int] = (0, 0)

    # ----- control ----------------------------------------------------------

    def submit(self, fn: Callable[["Engine"], None]) -> None:
        self._tasks.put(fn)

    def update_settings(self, **changes) -> None:
        def apply(eng: "Engine") -> None:
            s = eng.settings
            for k, v in changes.items():
                if not hasattr(s, k):
                    raise ValueError(f"unknown setting {k}")
                if k == "internal_size":
                    v = int(v)
                    if v != s.internal_size:
                        eng.matter.request_reseed(f"internal size {v}")
                setattr(s, k, type(getattr(s, k))(v))

        self.submit(apply)

    def stats(self) -> Dict:
        def r(x):
            return None if x is None else round(x, 1)

        d = {
            "frames": self.frames,
            "fps": round(self.fps.rate(), 1),
            "proc_ms_p50": r(self.t_proc.pct(50)),
            "proc_ms_p95": r(self.t_proc.pct(95)),
            "latency_ms_p50": r(self.t_latency.pct(50)),
            "internal": f"{self.internal_hw[1]}x{self.internal_hw[0]}",
            "output": f"{self.W}x{self.H}",
            "seeded": self.matter.seeded,
            "iou": None if self.matter.last_iou is None else round(self.matter.last_iou, 2),
            "reseeds": self.matter.reseeds,
            "errors": self._errors,
            "settings": asdict(self.settings),
        }
        if self.profile:
            d["stages_ms_p50"] = {k: r(v.pct(50)) for k, v in self.stage.items()}
        return d

    # ----- thread body ------------------------------------------------------

    def run(self) -> None:
        last_seq = -1
        last_log = time.monotonic()
        try:
            while not self.stop_event.is_set():
                frame = self.source.get(last_seq, timeout=0.5)
                if frame is None:
                    if self.source.eof:
                        log.info("input finished")
                        break
                    continue
                last_seq = frame.seq
                self._drain_tasks()
                out = self._process_safely(frame)
                if out is not None:
                    for sink in self.sinks:
                        sink.put(out)
                self.frames += 1
                if self.max_frames and self.frames >= self.max_frames:
                    break
                now = time.monotonic()
                if now - last_log > 5:
                    last_log = now
                    s = self.stats()
                    log.info(
                        "%.1f fps | proc p50 %s ms p95 %s ms | latency p50 %s ms | model %s | tracking %s iou %s | bg %s",
                        s["fps"], s["proc_ms_p50"], s["proc_ms_p95"], s["latency_ms_p50"], s["internal"],
                        "yes" if s["seeded"] else "no", s["iou"], self.backgrounds.describe(),
                    )
                    if self.profile:
                        log.info("stages p50 ms: %s", s["stages_ms_p50"])
        except Exception:  # noqa: BLE001
            log.error("engine crashed:\n%s", traceback.format_exc())
        finally:
            self.done.set()
            self.stop_event.set()

    def _drain_tasks(self) -> None:
        while True:
            try:
                fn = self._tasks.get_nowait()
            except queue.Empty:
                return
            try:
                fn(self)
            except Exception as exc:  # noqa: BLE001
                log.warning("setting change failed: %s", exc)

    def _process_safely(self, frame: Frame) -> Optional[FrameOut]:
        try:
            with torch.inference_mode():
                out = self._process(frame)
            self._last_good = out
            return out
        except Exception:  # noqa: BLE001
            self._errors += 1
            log.error("frame failed:\n%s", traceback.format_exc())
            # Fail closed: never fall back to the raw camera, it would show the real room.
            self.matter.request_reseed("error")
            if self.backgrounds.spec.get("type") not in ("blur", "none", "color"):
                self.backgrounds.fallback()
            return self._last_good

    def _mark(self, name: str, t0: float) -> float:
        if not self.profile:
            return t0
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.stage[name].add((t1 - t0) * 1000)
        return t1

    def _process(self, frame: Frame) -> FrameOut:
        s = self.settings
        t_start = time.perf_counter()
        t = t_start

        x = torch.from_numpy(frame.bgr).to(self.device, non_blocking=True)
        src = x.permute(2, 0, 1).flip(0).float().div_(255.0)  # (3,H,W) RGB
        src = cover_resize(src, (self.H, self.W))
        if s.mirror:
            src = src.flip(-1)
        if self.calibrator is not None and self.calibrator.active:
            # A/V sync test: coarse brightness grid to spot the phone's flash.
            luma = (src[0] * 0.299 + src[1] * 0.587 + src[2] * 0.114)[None, None]
            grid = F.adaptive_avg_pool2d(luma, (8, 8))[0, 0].float().cpu().numpy()
            self.calibrator.video_frame(frame.t, grid)
        lr = resize_short_side(src, s.internal_size).contiguous()
        self.internal_hw = tuple(lr.shape[-2:])
        t = self._mark("prep", t)

        with self._amp():
            alpha_lr = self.matter.process(lr)
        alpha_lr = alpha_lr.float()
        t = self._mark("matte", t)

        if alpha_lr.shape[-2:] == src.shape[-2:]:
            alpha = alpha_lr
        elif s.refine == "guided":
            alpha = guided_upsample(alpha_lr, lr.float(), src, s.gf_radius, s.gf_eps)
        else:
            alpha = bilinear_upsample(alpha_lr, (self.H, self.W))
        if s.alpha_hi > s.alpha_lo:
            alpha = ((alpha - s.alpha_lo) / (s.alpha_hi - s.alpha_lo)).clamp_(0, 1)
        t = self._mark("refine", t)

        now = time.monotonic()
        kind, bg = self.backgrounds.frame(now, src, alpha)
        if kind == "none":
            comp = src
        else:
            fg = estimate_foreground(src, alpha) if s.defringe else src
            a = alpha[None]
            comp = fg * a + bg * (1 - a)
        t = self._mark("compose", t)

        rgb = (comp.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        view = self.preview.wanted(now) if self.preview is not None else None
        if view is not None:
            self._publish_preview(view, comp, src, alpha)
        self._mark("download", t)

        t_end = time.perf_counter()
        self.t_proc.add((t_end - t_start) * 1000)
        self.t_latency.add((time.monotonic() - frame.t) * 1000)
        self.fps.tick()
        return FrameOut(frame.seq, frame.t, rgb)

    def _publish_preview(self, view: str, comp, src, alpha) -> None:
        img = {"alpha": alpha[None].expand(3, -1, -1), "camera": src}.get(view, comp)
        pw = self.preview.width
        ph = max(1, round(pw * self.H / self.W))
        small = F.interpolate(img[None].float(), size=(ph, pw), mode="bilinear", align_corners=False, antialias=True)[0]
        rgb = (small.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        self.preview.slot.put(rgb)
