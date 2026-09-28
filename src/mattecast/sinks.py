"""Outputs: a fullscreen window on the capture-card monitor, a v4l2loopback
virtual camera, or a video file."""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import cv2
import numpy as np

from mattecast.detect import attach_desktop_session
from mattecast.utils import LatestSlot, Rolling

log = logging.getLogger(__name__)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def _pygame():
    """Import pygame without its pkg_resources deprecation noise."""
    attach_desktop_session()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pkg_resources.*")
        import pygame
    return pygame


def _init_display(pygame) -> None:
    pygame.display.init()
    driver = pygame.display.get_driver()
    if driver in ("offscreen", "dummy") and os.environ.get("SDL_VIDEODRIVER") not in ("offscreen", "dummy"):
        pygame.display.quit()
        raise SystemExit(
            f"SDL picked the invisible '{driver}' video driver: no desktop session was found.\n"
            "Log in to the desktop on this machine (Moonlight is fine), then run again, or point at it by hand:\n"
            "  export DISPLAY=:0 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority"
        )


@dataclass
class FrameOut:
    seq: int
    t_capture: float
    rgb: np.ndarray  # (H, W, 3) uint8, C-contiguous


class Sink:
    name = "sink"

    def put(self, out: FrameOut) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class V4L2Sink(Sink):
    name = "v4l2"

    def __init__(self, device: str, size: Tuple[int, int], fps: float) -> None:
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise SystemExit("v4l2 output needs pyvirtualcam: uv sync --extra v4l2") from exc
        w, h = size
        self.cam = pyvirtualcam.Camera(
            width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB, device=device, backend="v4l2loopback"
        )
        log.info("virtual camera %s (%dx%d)", self.cam.device, w, h)

    def put(self, out: FrameOut) -> None:
        self.cam.send(out.rgb)

    def close(self) -> None:
        self.cam.close()


class FileSink(Sink):
    name = "file"

    def __init__(self, path: str, size: Tuple[int, int], fps: float) -> None:
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot write {path}")
        self.path = path

    def put(self, out: FrameOut) -> None:
        self.writer.write(cv2.cvtColor(out.rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()
        log.info("wrote %s", self.path)


class DisplaySink(Sink):
    """Hands frames to the display loop, which must run on the main thread."""

    name = "display"

    def __init__(self) -> None:
        self.slot: LatestSlot[FrameOut] = LatestSlot()
        self.flip_latency = Rolling(120)  # ms from camera read to the frame being shown

    def put(self, out: FrameOut) -> None:
        self.slot.put(out)


def run_display(
    sink: DisplaySink,
    stop: threading.Event,
    size: Tuple[int, int],
    fullscreen: bool,
    display_index: int,
    on_key: Callable[[str], None],
) -> None:
    """Blocking display loop. In fullscreen mode it opens a borderless window
    covering the chosen monitor (the one wired to the HDMI capture card)."""
    pygame = _pygame()

    _init_display(pygame)
    n = pygame.display.get_num_displays()
    log.info("SDL video driver %s, displays %s", pygame.display.get_driver(), pygame.display.get_desktop_sizes())
    if display_index >= n:
        raise SystemExit(f"display {display_index} does not exist ({n} found); try `mattecast displays`")
    w, h = size
    if fullscreen:
        flags = pygame.FULLSCREEN | pygame.SCALED
        win = (w, h)
    else:
        flags = pygame.RESIZABLE
        win = (w // 2, h // 2)
    try:
        screen = pygame.display.set_mode(win, flags, display=display_index, vsync=1)
    except pygame.error:
        screen = pygame.display.set_mode(win, flags, display=display_index)
    pygame.display.set_caption("mattecast output")
    pygame.mouse.set_visible(not fullscreen)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    log.info("display %d: %s %dx%d", display_index, "fullscreen" if fullscreen else "window", *screen.get_size())

    seen = 0
    try:
        while not stop.is_set():
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    stop.set()
                elif ev.type == pygame.KEYDOWN:
                    key = pygame.key.name(ev.key)
                    if key == "q" or (key == "escape" and not fullscreen):
                        stop.set()
                    else:
                        on_key(key)
            seq, out = sink.slot.get(seen, timeout=0.02)
            if seq <= seen or out is None:
                continue
            seen = seq
            surf = pygame.image.frombuffer(out.rgb.data, (out.rgb.shape[1], out.rgb.shape[0]), "RGB")
            target = screen.get_size()
            if surf.get_size() != target:
                surf = pygame.transform.smoothscale(surf, target)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            sink.flip_latency.add((time.monotonic() - out.t_capture) * 1000)
    finally:
        pygame.display.quit()


def list_displays() -> tuple:
    pygame = _pygame()
    _init_display(pygame)
    try:
        return pygame.display.get_driver(), list(pygame.display.get_desktop_sizes())
    finally:
        pygame.display.quit()


def identify_displays(seconds: float = 4.0) -> None:
    """Show a big index number on each monitor in turn, as a desktop-fullscreen
    window (no video mode switch, which some compositors refuse)."""
    pygame = _pygame()

    _init_display(pygame)
    pygame.font.init()
    sizes = pygame.display.get_desktop_sizes()
    print(f"SDL video driver: {pygame.display.get_driver()}, {len(sizes)} display(s): {sizes}", flush=True)
    colors = [(255, 140, 0), (0, 170, 255), (60, 200, 90), (230, 60, 160)]
    try:
        for i, (w, h) in enumerate(sizes):
            print(f"  showing {i} on display {i} ({w}x{h}) for {seconds:.0f}s", flush=True)
            screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN | pygame.SCALED, display=i)
            pygame.display.set_caption(f"mattecast display {i}")
            big = pygame.font.Font(None, h * 2 // 3).render(str(i), True, (0, 0, 0))
            small = pygame.font.Font(None, h // 12).render(f"display {i}   {w}x{h}", True, (0, 0, 0))
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                pygame.event.pump()
                screen.fill(colors[i % len(colors)])
                screen.blit(big, big.get_rect(center=(w // 2, h // 2 - h // 14)))
                screen.blit(small, small.get_rect(center=(w // 2, h - h // 8)))
                pygame.display.flip()
                time.sleep(0.05)
    finally:
        pygame.display.quit()
