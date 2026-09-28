"""Fetch and load MatAnyone 2.

The upstream package cannot currently be pip-installed as a dependency (its
wheel build fails on recent hatchling) and its code and weights carry a
non-commercial license, so mattecast does not vendor it. Instead, like
torch.hub, it downloads a pinned source snapshot and the official checkpoint
into ~/.cache/mattecast on first run and imports from there.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch

from mattecast.utils import cache_dir

log = logging.getLogger(__name__)

MATANYONE_COMMIT = "0079197acd6d16a741f71558809c06c586c579e0"
MATANYONE_ARCHIVE = f"https://github.com/pq-yang/MatAnyone2/archive/{MATANYONE_COMMIT}.tar.gz"
WEIGHTS_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
WEIGHTS_SHA256 = "5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914"


def _download(url: str, dest: Path, sha256: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "mattecast"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct // 10 != shown:
                    shown = pct // 10
                    print(f"  {dest.name}: {pct}% of {total / 1e6:.0f} MB", file=sys.stderr, flush=True)
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: got {digest.hexdigest()}")
    tmp.replace(dest)


def _extract_package(tarball: Path, dest: Path) -> None:
    """Extract only matanyone2/ plus license files, refusing unsafe paths."""
    tmp = Path(tempfile.mkdtemp(prefix="matanyone2-", dir=dest.parent))
    try:
        with tarfile.open(tarball, "r:gz") as tar:
            for m in tar.getmembers():
                parts = m.name.split("/", 1)
                if len(parts) < 2:
                    continue
                rel = parts[1]
                keep = rel.startswith("matanyone2/") or rel in ("LICENSE.txt", "LICENSE", "README.md")
                if not keep or not (m.isfile() or m.isdir()):
                    continue
                target = (tmp / rel).resolve()
                if not str(target).startswith(str(tmp.resolve()) + os.sep):
                    raise RuntimeError(f"unsafe path in archive: {m.name}")
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                with src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        if not (tmp / "matanyone2" / "__init__.py").exists():
            raise RuntimeError("archive did not contain the matanyone2 package")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def ensure_matanyone_source(override: Optional[str] = None) -> Path:
    """Return a directory containing the `matanyone2` package and put it on sys.path."""
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "matanyone2" / "__init__.py").exists():
            raise FileNotFoundError(f"{root} does not contain matanyone2/__init__.py")
    else:
        root = cache_dir() / f"MatAnyone2-{MATANYONE_COMMIT[:12]}"
        if not (root / "matanyone2" / "__init__.py").exists():
            tarball = cache_dir() / f"MatAnyone2-{MATANYONE_COMMIT}.tar.gz"
            _download(MATANYONE_ARCHIVE, tarball)
            _extract_package(tarball, root)
            tarball.unlink(missing_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ensure_weights(override: Optional[str] = None) -> Path:
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    p = cache_dir() / "matanyone2.pth"
    marker = p.with_name(p.name + ".sha256")
    if p.is_file() and marker.is_file() and marker.read_text().strip() == WEIGHTS_SHA256:
        return p
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == WEIGHTS_SHA256:
            marker.write_text(WEIGHTS_SHA256)
            return p
    _download(WEIGHTS_URL, p, WEIGHTS_SHA256)
    marker.write_text(WEIGHTS_SHA256)
    return p


def load_matanyone(
    device: torch.device,
    source: Optional[str] = None,
    weights: Optional[str] = None,
) -> Tuple[torch.nn.Module, object]:
    """Build the MatAnyone 2 network and its inference config without hydra."""
    root = ensure_matanyone_source(source)
    ckpt = ensure_weights(weights)

    from matanyone2.model.matanyone2 import MatAnyone2  # type: ignore[import-not-found]
    from omegaconf import OmegaConf

    logging.getLogger("matanyone2").setLevel(logging.ERROR)
    cfg_dir = root / "matanyone2" / "config"
    cfg = OmegaConf.load(cfg_dir / "eval_matanyone_config.yaml")
    for key in ("defaults", "hydra"):
        cfg.pop(key, None)
    cfg.model = OmegaConf.load(cfg_dir / "model" / "base.yaml")
    # The checkpoint overwrites every backbone weight, so skip the ImageNet download.
    cfg.model.pretrained_resnet = False
    # We resize frames ourselves and keep them on the GPU.
    cfg.max_internal_size = -1
    cfg.weights = str(ckpt)

    net = MatAnyone2(cfg, single_object=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    net.load_weights(state)
    net.to(device).eval()
    return net, cfg


def inference_core_class():
    from matanyone2.inference.inference_core import (
        InferenceCore,  # type: ignore[import-not-found]
    )

    return InferenceCore
