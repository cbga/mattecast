"""GPU image operators: box filters, fast guided upsampling, foreground estimation.

Everything here runs in float32. Callers must keep these ops outside autocast:
the integral-image box filter overflows in float16.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_count_cache: Dict[Tuple, torch.Tensor] = {}


def _diff(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    # x is a cumulative sum along `dim`; return windowed sums of width 2r+1.
    n = x.shape[dim]
    left = x.narrow(dim, r, r + 1)
    middle = x.narrow(dim, 2 * r + 1, n - 2 * r - 1) - x.narrow(dim, 0, n - 2 * r - 1)
    right = x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2 * r - 1, r)
    return torch.cat([left, middle, right], dim)


def box_sum(x: torch.Tensor, r: int) -> torch.Tensor:
    """Sum over a (2r+1)^2 window, O(1) per pixel. x is (B, C, H, W)."""
    return _diff(_diff(x.cumsum(2), r, 2).cumsum(3), r, 3)


def _clamp_radius(h: int, w: int, r: int) -> int:
    return max(1, min(r, (min(h, w) - 2) // 2))


def box_mean(x: torch.Tensor, r: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    r = _clamp_radius(h, w, r)
    key = (h, w, r, x.device, x.dtype)
    n = _count_cache.get(key)
    if n is None:
        n = box_sum(torch.ones((1, 1, h, w), device=x.device, dtype=x.dtype), r)
        if len(_count_cache) > 32:
            _count_cache.clear()
        _count_cache[key] = n
    return box_sum(x, r) / n


def _guided_coeffs_gray(i: torch.Tensor, p: torch.Tensor, r: int, eps: float):
    mean_i = box_mean(i, r)
    mean_p = box_mean(p, r)
    cov_ip = box_mean(i * p, r) - mean_i * mean_p
    var_i = box_mean(i * i, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box_mean(a, r), box_mean(b, r)


def _guided_coeffs_color(img: torch.Tensor, p: torch.Tensor, r: int, eps: float):
    """He et al. color guided filter. img (1,3,h,w), p (1,1,h,w)."""
    m = lambda t: box_mean(t, r)  # noqa: E731
    ir, ig, ib = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    mr, mg, mb = m(ir), m(ig), m(ib)
    mp = m(p)
    cr = m(ir * p) - mr * mp
    cg = m(ig * p) - mg * mp
    cb = m(ib * p) - mb * mp
    vrr = m(ir * ir) - mr * mr + eps
    vrg = m(ir * ig) - mr * mg
    vrb = m(ir * ib) - mr * mb
    vgg = m(ig * ig) - mg * mg + eps
    vgb = m(ig * ib) - mg * mb
    vbb = m(ib * ib) - mb * mb + eps
    # Adjugate of the symmetric 3x3 covariance, per pixel.
    irr = vgg * vbb - vgb * vgb
    irg = vgb * vrb - vrg * vbb
    irb = vrg * vgb - vgg * vrb
    igg = vrr * vbb - vrb * vrb
    igb = vrb * vrg - vrr * vgb
    ibb = vrr * vgg - vrg * vrg
    det = vrr * irr + vrg * irg + vrb * irb
    ar = (irr * cr + irg * cg + irb * cb) / det
    ag = (irg * cr + igg * cg + igb * cb) / det
    ab = (irb * cr + igb * cg + ibb * cb) / det
    b = mp - ar * mr - ag * mg - ab * mb
    return m(torch.cat([ar, ag, ab], 1)), m(b)


def guided_upsample(
    alpha_lr: torch.Tensor,
    guide_lr: torch.Tensor,
    guide_hr: torch.Tensor,
    radius: int = 2,
    eps: float = 1e-3,
    color: bool = True,
) -> torch.Tensor:
    """Fast guided filter (Wu et al. 2018): fit a local linear model between the
    low-res guide and alpha, then apply it to the full-res guide so alpha edges
    snap to real image edges instead of being bilinearly smeared.

    alpha_lr (h,w), guide_lr (3,h,w), guide_hr (3,H,W) -> alpha (H,W)
    """
    p = alpha_lr[None, None].float()
    i_lr = guide_lr[None].float()
    i_hr = guide_hr[None].float()
    if not color:
        wts = torch.tensor([0.299, 0.587, 0.114], device=p.device).view(1, 3, 1, 1)
        i_lr = (i_lr * wts).sum(1, keepdim=True)
        i_hr = (i_hr * wts).sum(1, keepdim=True)
        a, b = _guided_coeffs_gray(i_lr, p, radius, eps)
    else:
        a, b = _guided_coeffs_color(i_lr, p, radius, eps)
    size = i_hr.shape[-2:]
    a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
    return ((a * i_hr).sum(1, keepdim=True) + b).clamp_(0, 1)[0, 0]


def bilinear_upsample(alpha_lr: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(alpha_lr[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]


def _blur_fusion(img, fg, bg, a, r):
    ba = box_mean(a, r)
    bf = box_mean(fg * a, r) / (ba + 1e-5)
    bb = box_mean(bg * (1 - a), r) / ((1 - ba) + 1e-5)
    fg = bf + a * (img - a * bf - (1 - a) * bb)
    return fg.clamp_(0, 1), bb


def estimate_foreground(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Blur-fusion foreground estimation (Forte & Pitie, ICIP 2021).

    Removes the old background's color from semi-transparent edge pixels (hair,
    motion blur) so they do not glow when composited over a new background.
    image (3,H,W), alpha (H,W) -> foreground (3,H,W)
    """
    h = image.shape[-2]
    img = image[None].float()
    a = alpha[None, None].float()
    r1 = max(8, round(90 * h / 1080))
    r2 = max(2, round(6 * h / 1080))
    fg, bg = _blur_fusion(img, img, img, a, r1)
    fg, _ = _blur_fusion(img, fg, bg, a, r2)
    return fg[0]


def gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable gaussian blur with reflect padding. x (B,C,H,W)."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device)
    c = x.shape[1]
    r = k.numel() // 2
    h, w = x.shape[-2:]
    # Reflect padding needs pad < dim; fall back to replicate for tiny inputs.
    mode = "reflect" if r < min(h, w) else "replicate"
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), ky, groups=c)
    return x


def resize_short_side(img: torch.Tensor, short: int) -> torch.Tensor:
    """img (3,H,W) -> (3,h,w) with min(h,w) == short (never upsamples)."""
    h, w = img.shape[-2:]
    s = short / min(h, w)
    if s >= 1:
        return img
    size = (max(1, round(h * s)), max(1, round(w * s)))
    return F.interpolate(img[None], size=size, mode="area")[0]


def cover_resize(img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Scale to cover (H,W) keeping aspect ratio, then center crop. img (3,h,w) float."""
    H, W = size_hw
    h, w = img.shape[-2:]
    if (h, w) == (H, W):
        return img
    s = max(H / h, W / w)
    nh, nw = max(H, round(h * s)), max(W, round(w * s))
    if (nh, nw) != (h, w):
        img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False, antialias=s < 1)[0]
    y0, x0 = (nh - H) // 2, (nw - W) // 2
    return img[:, y0 : y0 + H, x0 : x0 + W]
