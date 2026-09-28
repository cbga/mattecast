"""Plug-and-play discovery of the capture card and the webcam's microphone.

An HDMI capture card announces itself to the GPU like a monitor would: its EDID
carries a product name ("Cam Link 4K", "USB Capture HDMI", ...). That name shows
up in two places we can read without special permissions:

  * video: `xrandr --verbose` prints each output's EDID, so we know which monitor
    is the card, and from the X output order which SDL display index it gets;
  * audio: the GPU's HDMI audio function copies it into the ELD, which PipeWire
    and PulseAudio attach to the matching HDMI port in `pactl list cards`.

The webcam's microphone is found by USB identity: the V4L2 node and the ALSA
source of one physical device share vendor id and serial number.

Everything here is best effort and returns None when unsure; explicit command
line options always win.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Lower-case fragments of EDID / ELD names used by common HDMI-to-USB capture devices.
CAPTURE_HINTS = (
    "cam link", "camlink", "elgato", "game capture", "hd60", "magewell", "usb capture",
    "avermedia", "live gamer", "live streamer", "macrosilicon", "ms2109", "ms2130", "hdmi to usb",
    "capture", "atem mini", "blackmagic", "ripsaw", "nzxt signal",
)


def hint_list(extra: Optional[str]) -> tuple:
    return ((extra.lower(),) if extra else ()) + CAPTURE_HINTS


def _matches(text: str, hints: tuple) -> bool:
    t = text.lower()
    return any(h in t for h in hints)


# ----- desktop session ---------------------------------------------------------------


def attach_desktop_session() -> None:
    """When started from SSH or a TTY there is no DISPLAY / WAYLAND_DISPLAY, and SDL
    silently falls back to an invisible offscreen driver. Attach to the logged-in
    desktop of the same user instead: its X server (Xorg or GNOME's Xwayland) if one
    is running, otherwise its Wayland socket."""
    if os.environ.get("SDL_VIDEODRIVER") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    uid = os.getuid()
    run = Path(f"/run/user/{uid}")
    x_sockets = []
    for sock in Path("/tmp/.X11-unix").glob("X*"):
        try:
            if sock.stat().st_uid == uid and sock.name[1:].isdigit():
                x_sockets.append(int(sock.name[1:]))
        except OSError:
            pass
    if x_sockets:
        os.environ["DISPLAY"] = f":{min(x_sockets)}"
        if not os.environ.get("XAUTHORITY"):
            candidates = [run / "gdm" / "Xauthority", *sorted(run.glob(".mutter-Xwaylandauth.*")), Path.home() / ".Xauthority"]
            for c in candidates:
                if c.is_file():
                    os.environ["XAUTHORITY"] = str(c)
                    break
        log.info("no DISPLAY in this shell; using the desktop session at DISPLAY=%s (XAUTHORITY=%s)",
                 os.environ["DISPLAY"], os.environ.get("XAUTHORITY", "unset"))
        return
    wl = sorted(run.glob("wayland-[0-9]"))
    if wl:
        os.environ.setdefault("XDG_RUNTIME_DIR", str(run))
        os.environ["WAYLAND_DISPLAY"] = wl[0].name
        os.environ["SDL_VIDEODRIVER"] = "wayland"
        log.info("no DISPLAY in this shell; using the Wayland session %s", wl[0].name)


# ----- video: which monitor is the capture card ---------------------------------------


@dataclass
class Monitor:
    output: str
    connected: bool
    primary: bool
    active: bool
    width: int
    height: int
    x: int
    y: int
    refresh: Optional[float]
    edid_name: str
    edid_vendor: str
    sdl_index: Optional[int] = None


def _edid_info(hexstr: str):
    try:
        edid = bytes.fromhex(hexstr)
    except ValueError:
        return "", ""
    if len(edid) < 128:
        return "", ""
    raw = (edid[8] << 8) | edid[9]
    vendor = "".join(chr(((raw >> s) & 0x1F) + 64) for s in (10, 5, 0))
    name = ""
    for off in (54, 72, 90, 108):
        d = edid[off : off + 18]
        if d[0:3] == b"\x00\x00\x00" and d[3] == 0xFC:
            name = d[5:18].split(b"\n")[0].decode("ascii", "replace").strip()
    return name, vendor


_OUT_RE = re.compile(r"^(\S+) (connected|disconnected)( primary)?(?: (\d+)x(\d+)\+(\d+)\+(\d+))?")


def list_monitors() -> List[Monitor]:
    """Outputs from `xrandr --verbose` in X order, with SDL's display index filled in
    (SDL numbers the primary output 0, then the other active outputs in X order)."""
    attach_desktop_session()
    if not shutil.which("xrandr") or not os.environ.get("DISPLAY"):
        return []
    try:
        out = subprocess.run(["xrandr", "--verbose"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    mons: List[Monitor] = []
    cur: Optional[dict] = None
    edid_lines: Optional[list] = None
    in_current_mode = False

    def flush():
        if cur is not None:
            name, vendor = _edid_info("".join(cur.pop("edid", [])))
            mons.append(Monitor(edid_name=name, edid_vendor=vendor, **cur))

    for line in out.splitlines():
        m = _OUT_RE.match(line)
        if m:
            flush()
            edid_lines = None
            in_current_mode = False
            cur = {
                "output": m.group(1), "connected": m.group(2) == "connected", "primary": bool(m.group(3)),
                "active": m.group(4) is not None,
                "width": int(m.group(4) or 0), "height": int(m.group(5) or 0),
                "x": int(m.group(6) or 0), "y": int(m.group(7) or 0), "refresh": None, "edid": [],
            }
            continue
        if cur is None:
            continue
        s = line.strip()
        if s == "EDID:":
            edid_lines = cur["edid"]
            continue
        if edid_lines is not None:
            if re.fullmatch(r"[0-9a-fA-F]+", s):
                edid_lines.append(s)
                continue
            edid_lines = None
        # Mode blocks: "1920x1080 (0x1c9) 148.500MHz ... *current", then "h: ...", then "v: ... clock 60.00Hz".
        if "*current" in s:
            in_current_mode = True
        elif in_current_mode and s.startswith("v:"):
            m2 = re.search(r"clock\s+([\d.]+)Hz", s)
            if m2 and cur["refresh"] is None:
                cur["refresh"] = float(m2.group(1))
            in_current_mode = False
    flush()
    active = [m for m in mons if m.active]
    order = [m for m in active if m.primary] + [m for m in active if not m.primary]
    for i, m in enumerate(order):
        m.sdl_index = i
    return mons


def find_capture_monitor(extra_hint: Optional[str] = None) -> Optional[Monitor]:
    hints = hint_list(extra_hint)
    hits = [m for m in list_monitors() if m.connected and _matches(f"{m.edid_name} {m.output}", hints)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning("several capture-like monitors: %s; pass --display-index", [m.edid_name for m in hits])
    return None


# ----- audio: which HDMI port is the capture card ------------------------------------------


@dataclass
class HdmiPort:
    card: str  # alsa_card.pci-...
    port: str
    description: str
    product: str
    available: bool
    profile: str  # output:hdmi-stereo-extraN

    @property
    def sink(self) -> str:
        return f"alsa_output.{self.card[len('alsa_card.'):]}.{self.profile[len('output:'):]}"


def list_hdmi_ports() -> List[HdmiPort]:
    if not shutil.which("pactl"):
        return []
    try:
        out = subprocess.run(["pactl", "list", "cards"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    ports: List[HdmiPort] = []
    card = None
    in_ports = False
    cur: Optional[dict] = None

    def flush():
        if cur and cur.get("profile"):
            ports.append(HdmiPort(**cur))

    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Name: alsa_card."):
            flush()
            cur, card, in_ports = None, s[len("Name: "):], False
            continue
        if s == "Ports:":
            in_ports = True
            continue
        if not in_ports or card is None:
            continue
        m = re.match(r"^((?:hdmi|dp|iec958)[\w-]*output[\w-]*): (.*)$", s)
        if m and not s.startswith("Part of"):
            flush()
            desc = m.group(2)
            cur = {
                "card": card, "port": m.group(1), "description": desc, "product": "",
                "available": not re.search(r"not available|unavailable", desc), "profile": "",
            }
            continue
        if cur is None:
            continue
        m = re.match(r'^device\.product\.name = "(.*)"$', s)
        if m:
            cur["product"] = m.group(1)
            continue
        if s.startswith("Part of profile(s):"):
            profs = [p.strip() for p in s.split(":", 1)[1].split(",")]
            stereo = [p for p in profs if p.startswith("output:hdmi-stereo")] or [p for p in profs if p.startswith("output:")]
            cur["profile"] = stereo[0] if stereo else ""
            flush()
            cur = None
    flush()
    return ports


def find_capture_audio(extra_hint: Optional[str] = None) -> Optional[HdmiPort]:
    hints = hint_list(extra_hint)
    hits = [p for p in list_hdmi_ports() if p.available and _matches(f"{p.product} {p.description}", hints)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning("several capture-like HDMI audio ports: %s; pass --audio-out", [p.sink for p in hits])
    return None


# ----- the webcam's own microphone -----------------------------------------------------


def _usb_identity(video_dev: str) -> Optional[tuple]:
    """(vendor_id, serial, product) of the USB device behind /dev/videoN."""
    node = Path(video_dev).name
    if node.isdigit():
        node = f"video{node}"
    dev = Path(f"/sys/class/video4linux/{node}/device")
    try:
        p = dev.resolve()
    except OSError:
        return None
    def read(d: Path, f: str) -> str:
        return (d / f).read_text().strip() if (d / f).is_file() else ""

    for d in [p, *p.parents]:
        if (d / "idVendor").is_file():
            return read(d, "idVendor"), read(d, "serial"), read(d, "product")
    return None


def find_camera_mic(video_dev: str) -> Optional[str]:
    ident = _usb_identity(video_dev)
    if not ident or not shutil.which("pactl"):
        return None
    vendor, serial, product = ident
    try:
        out = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    names = [ln.split("\t")[1] for ln in out.splitlines() if "\t" in ln]
    names = [n for n in names if n.startswith("alsa_input.usb-") and not n.endswith(".monitor")]
    tag = f"usb-{vendor}_".lower()
    hits = [n for n in names if tag in n.lower() and (not serial or serial.lower() in n.lower())]
    if not hits and product:
        key = re.sub(r"[^a-z0-9]", "", product.lower())
        hits = [n for n in names if tag in n.lower() and key and key in re.sub(r"[^a-z0-9]", "", n.lower())]
    return hits[0] if len(hits) == 1 else None
