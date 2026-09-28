"""Streaming wrapper around MatAnyone 2's stateful `InferenceCore.step()`.

Offline, MatAnyone 2 takes a video plus a hand-made first-frame mask. Live, we
have to supply that mask ourselves, notice when tracking has gone wrong, and
start over. This module owns that state machine:

    unseeded --(person found by segmenter)--> seed + warmup --> tracking
    tracking --(IoU with segmenter stays low)--> reseed (or unseeded if nobody is there)
    any      --(manual request / internal size change)--> reseed
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from mattecast.models import inference_core_class

log = logging.getLogger(__name__)


def _closing(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Morphological closing of a (h,w) {0,1} mask; fills small holes and gaps."""
    x = mask[None, None].float()
    pad = k // 2
    x = F.max_pool2d(x, k, 1, pad)
    x = -F.max_pool2d(-x, k, 1, pad)
    return x[0, 0]


class StreamingMatter:
    def __init__(
        self,
        network: torch.nn.Module,
        cfg,
        device: torch.device,
        segmenter=None,
        *,
        warmup: int = 10,
        check_every: int = 15,
        drift_iou: float = 0.5,
        drift_patience: int = 3,
        min_area: float = 0.01,
        check_size: int = 320,
        seed_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.device = device
        self.segmenter = segmenter
        self.warmup = warmup
        self.check_every = check_every
        self.drift_iou = drift_iou
        self.drift_patience = drift_patience
        self.min_area = min_area
        self.check_size = check_size
        self._static_mask = seed_mask
        self._core_cls = inference_core_class()

        self.proc = None
        self._reseed = threading.Event()
        self._frames = 0
        self._bad = 0
        self.last_iou: Optional[float] = None
        self.reseeds = 0
        self.events: deque = deque(maxlen=30)

    # ----- public ---------------------------------------------------------

    @property
    def seeded(self) -> bool:
        return self.proc is not None

    def request_reseed(self, reason: str = "manual") -> None:
        self._reseed_reason = reason
        self._reseed.set()

    def process(self, img: torch.Tensor) -> torch.Tensor:
        """img (3,h,w) RGB float in [0,1] on device -> alpha (h,w) float32."""
        h, w = img.shape[-2:]
        if self._reseed.is_set():
            self._reseed.clear()
            self._event(f"reseed requested ({getattr(self, '_reseed_reason', 'manual')})")
            self.proc = None

        if self.proc is None:
            mask = self._initial_mask(img)
            if mask is None:
                return torch.zeros((h, w), device=img.device)
            return self._seed(img, mask)

        prob = self.proc.step(img)
        alpha = prob[1].float()
        self._frames += 1
        if self.segmenter is not None and self.check_every > 0 and self._frames % self.check_every == 0:
            alpha = self._check_drift(img, alpha)
        return alpha

    # ----- internals ------------------------------------------------------

    def _event(self, msg: str) -> None:
        log.info(msg)
        self.events.append((time.time(), msg))

    def _initial_mask(self, img: torch.Tensor) -> Optional[torch.Tensor]:
        h, w = img.shape[-2:]
        if self._static_mask is not None:
            m = torch.from_numpy(self._static_mask).to(img.device).float()[None, None]
            m = F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 127
            if self.segmenter is not None:
                self._static_mask = None  # later reseeds come from the detector
            return m
        if self.segmenter is None:
            return None
        person = self.segmenter(img) > 0.5
        if person.float().mean().item() < self.min_area:
            return None
        return person

    def _seed(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        k = max(3, round(min(h, w) / 100)) | 1
        m255 = _closing(mask, k) * 255.0
        proc = self._core_cls(self.network, cfg=self.cfg, device=self.device)
        proc.step(img, m255, objects=[1])
        prob = None
        for _ in range(max(1, self.warmup)):
            prob = proc.step(img, first_frame_pred=True)
        self.proc = proc
        self.last_iou = None
        self._frames = 0
        self._bad = 0
        self.reseeds += 1
        self._event(f"seeded at {w}x{h} (#{self.reseeds})")
        return prob[1].float()

    def _check_drift(self, img: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        seg = self.segmenter(img, self.check_size) > 0.5
        mat = alpha > 0.5
        seg_area = seg.float().mean().item()
        mat_area = mat.float().mean().item()
        if seg_area < self.min_area and mat_area < self.min_area:
            iou = 1.0
        else:
            inter = (seg & mat).sum().item()
            union = (seg | mat).sum().item()
            iou = inter / max(union, 1)
        self.last_iou = iou
        self._bad = self._bad + 1 if iou < self.drift_iou else 0
        if self._bad < self.drift_patience:
            return alpha
        if seg_area >= self.min_area:
            self._event(f"tracking drifted (IoU {iou:.2f}), reseeding")
            return self._seed(img, seg)
        self._event("subject left the frame, waiting")
        self.proc = None
        self.last_iou = None
        return torch.zeros_like(alpha)
