"""Person segmentation used to start (and restart) MatAnyone 2 tracking.

MatAnyone 2 propagates a matte from memory, so it needs someone to tell it who
the subject is on the first frame. A coarse semantic "person" mask is enough:
the model's warmup refines edges on its own. The same segmenter is reused at a
low rate as a drift detector.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ARCHS = ("deeplabv3_resnet50", "deeplabv3_mobilenet", "none")

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_VOC_PERSON = 15


class PersonSegmenter:
    def __init__(self, arch: str, device: torch.device, size: int = 520, pretrained: bool = True) -> None:
        from torchvision.models import segmentation as seg

        if arch == "deeplabv3_resnet50":
            weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            model = seg.deeplabv3_resnet50(weights=weights, weights_backbone=None)
        elif arch == "deeplabv3_mobilenet":
            weights = seg.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = seg.deeplabv3_mobilenet_v3_large(weights=weights, weights_backbone=None)
        else:
            raise ValueError(f"unknown segmenter {arch!r}; choose from {ARCHS}")
        self.person = weights.meta["categories"].index("person") if weights is not None else _VOC_PERSON
        self.model = model.eval().to(device)
        self.size = size
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, img: torch.Tensor, size: int = 0) -> torch.Tensor:
        """img (3,h,w) RGB in [0,1] -> person probability (h,w) in [0,1].

        `size` overrides the short side the detector runs at (drift checks use a
        smaller one: a coarse mask is plenty for an overlap test)."""
        h, w = img.shape[-2:]
        s = (size or self.size) / min(h, w)
        x = img[None]
        if abs(s - 1) > 1e-3:
            x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=s < 1)
        x = (x - self.mean) / self.std
        logits = self.model(x)["out"].float()
        prob = logits.softmax(1)[:, self.person : self.person + 1]
        return F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
