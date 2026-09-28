"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time

from mattecast import __version__
from mattecast.utils import StateStore, cache_dir, data_dir, parse_size

log = logging.getLogger("mattecast")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("input")
    src = g.add_mutually_exclusive_group()
    src.add_argument("--camera", default="/dev/video0", help="V4L2 device or index (default /dev/video0)")
    src.add_argument("--input", metavar="VIDEO", help="use a video file instead of a camera (for testing)")
    g.add_argument("--camera-size", default="1920x1080", type=parse_size, help="requested camera resolution")
    g.add_argument("--camera-fps", default=30.0, type=float)
    g.add_argument("--camera-format", default="MJPG", help="FOURCC to request, e.g. MJPG or YUYV")
    g.add_argument("--loop", action="store_true", help="loop the --input video")
    g.add_argument("--no-pace", action="store_true", help="process every --input frame as fast as possible")

    g = p.add_argument_group("model")
    g.add_argument("--device", default="auto", help="cuda, cuda:1, cpu or auto")
    g.add_argument("--precision", default="fp16", choices=("fp16", "bf16", "fp32"))
    g.add_argument("--internal-size", type=int, default=540, help="model resolution, short side (default 540)")
    g.add_argument("--warmup", type=int, default=10, help="refinement steps on the first frame after seeding")
    g.add_argument("--seeder", default="deeplabv3_resnet50", choices=("deeplabv3_resnet50", "deeplabv3_mobilenet", "none"))
    g.add_argument("--seed-mask", metavar="PNG", help="first-frame mask (white = person) instead of auto detection")
    g.add_argument("--check-every", type=int, default=15, help="frames between drift checks (0 disables)")
    g.add_argument("--drift-iou", type=float, default=0.5, help="reseed when IoU with the detector stays below this")
    g.add_argument("--drift-patience", type=int, default=3, help="consecutive bad checks before reseeding")
    g.add_argument("--matanyone-src", metavar="DIR", help="use a local MatAnyone2 checkout instead of the pinned download")
    g.add_argument("--weights", metavar="PTH", help="use a local matanyone2.pth")

    g = p.add_argument_group("compositing")
    g.add_argument("--size", default="1920x1080", type=parse_size, help="output resolution (default 1920x1080)")
    g.add_argument("--background", help="blur, blur:20, color:#00b140, none, an image/GIF/video path, or a library name")
    g.add_argument("--refine", default="guided", choices=("guided", "bilinear"))
    g.add_argument("--gf-radius", type=int, default=2)
    g.add_argument("--gf-eps", type=float, default=1e-3)
    g.add_argument("--no-defringe", action="store_true", help="skip foreground color estimation")
    g.add_argument("--mirror", action="store_true")

    g = p.add_argument_group("output")
    g.add_argument("--display", default="auto", choices=("auto", "fullscreen", "window", "none"),
                   help="auto: fullscreen on the capture card if one is found, else a preview window")
    g.add_argument("--display-index", type=int, default=None, help="monitor index (default: the detected capture card)")
    g.add_argument("--capture-name", metavar="TEXT", help="part of the capture card's name, if auto detection misses it")
    g.add_argument("--set-mode", action="store_true", help="switch the capture card output to the output size at 60 Hz (xrandr)")
    g.add_argument("--v4l2", metavar="DEV", help="also write to a v4l2loopback device, e.g. /dev/video10")
    g.add_argument("--record", metavar="MP4", help="also record the output to a file")
    g.add_argument("--output-fps", type=float, default=30.0, help="frame rate declared to v4l2/record")
    g.add_argument("--control-host", default="127.0.0.1", help="use 0.0.0.0 to open the panel from other machines")
    g.add_argument("--control-port", type=int, default=8765)
    g.add_argument("--no-control", action="store_true")
    g.add_argument("--max-frames", type=int, default=0, help="stop after N frames (testing)")
    g.add_argument("--profile", action="store_true", help="time each stage (adds GPU syncs)")

    g = p.add_argument_group("audio (forward a mic down the same HDMI cable)")
    g.add_argument("--audio-in", metavar="SOURCE", default="auto",
                   help="auto (the webcam's own mic), none, or a pactl source name / unique part of it")
    g.add_argument("--audio-out", metavar="SINK", default="auto",
                   help="auto (the capture card's HDMI port), none, or a pactl sink name")
    g.add_argument("--audio-delay", metavar="auto|MS", help="auto (follow the video latency) or a fixed delay in ms")
    g.add_argument("--mute", action="store_true", help="start with the mic muted")


def _choose_display(args, out_size):
    """-> (mode, index). With --display auto, go fullscreen on the capture card if found."""
    from mattecast.detect import find_capture_monitor

    mode, index = args.display, args.display_index
    if mode == "none":
        return mode, 0
    mon = None
    if index is None and mode in ("auto", "fullscreen"):
        mon = find_capture_monitor(args.capture_name)
    if mon is not None:
        index = mon.sdl_index
        rate = f"@{mon.refresh:.0f}Hz" if mon.refresh else ""
        log.info("capture card %r on %s (%dx%d%s) -> display %s",
                 mon.edid_name or mon.edid_vendor, mon.output, mon.width, mon.height, rate, index)
        want_w, want_h = out_size
        off = (mon.width, mon.height) != (want_w, want_h) or (mon.refresh is not None and mon.refresh < 50)
        cmd = ["xrandr", "--output", mon.output, "--mode", f"{want_w}x{want_h}", "--rate", "60"]
        if off and args.set_mode:
            rc = subprocess.run(cmd, capture_output=True, text=True)
            log.info("set %s to %dx%d@60: %s", mon.output, want_w, want_h, "ok" if rc.returncode == 0 else rc.stderr.strip())
            time.sleep(1.0)
        elif off:
            log.warning("the capture card runs at %dx%d%s; %dx%d at 60 Hz gives the lowest latency: %s (or --set-mode)",
                        mon.width, mon.height, rate, want_w, want_h, " ".join(cmd))
        if mode == "auto":
            mode = "fullscreen"
    elif mode == "auto":
        if index is not None:
            mode = "fullscreen"
        else:
            log.warning("no capture card monitor found (X11 EDID); showing a preview window. "
                        "Use --display fullscreen --display-index N, see `mattecast detect`")
            mode = "window"
    return mode, index or 0


def _choose_audio(args):
    """-> (source, sink) names, or (None, None) when forwarding stays off."""
    from mattecast.audio import resolve_device
    from mattecast.detect import find_camera_mic, find_capture_audio

    if args.audio_in == "none" or args.audio_out == "none":
        return None, None
    if not shutil.which("pactl"):
        log.info("audio forwarding off: pactl not found")
        return None, None
    if args.audio_in == "auto":
        src = None if args.input else find_camera_mic(args.camera)
        if src is None:
            log.info("audio forwarding off: the camera has no microphone we could match (use --audio-in)")
            return None, None
    else:
        src = resolve_device("sources", args.audio_in)
    if args.audio_out == "auto":
        port = find_capture_audio(args.capture_name)
        if port is None:
            log.info("audio forwarding off: no HDMI audio port reports a capture card (use --audio-out)")
            return None, None
        log.info("capture card audio: %s on %s", port.product or port.description, port.sink)
        sink = resolve_device("sinks", port.sink)
    else:
        sink = resolve_device("sinks", args.audio_out)
    return src, sink


def _resolve_device(name: str):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log.info("device %s: %s", dev, torch.cuda.get_device_name(dev))
    else:
        log.warning("running on CPU: expect well under 1 fps; use --internal-size 288 for testing")
    return dev


def cmd_run(args) -> int:
    import cv2
    import torch

    from mattecast.audio import pulse_bridge
    from mattecast.avsync import AVSync, SyncCalibrator
    from mattecast.backgrounds import BackgroundManager
    from mattecast.control import ControlServer
    from mattecast.matter import StreamingMatter
    from mattecast.models import load_matanyone
    from mattecast.pipeline import Engine, PreviewHub, Settings
    from mattecast.seeder import PersonSegmenter
    from mattecast.sinks import DisplaySink, FileSink, V4L2Sink, run_display
    from mattecast.sources import CameraSource, FileSource

    device = _resolve_device(args.device)
    out_w, out_h = args.size

    log.info("loading MatAnyone 2")
    net, cfg = load_matanyone(device, args.matanyone_src, args.weights)
    segmenter = None
    if args.seeder != "none":
        log.info("loading person detector %s", args.seeder)
        segmenter = PersonSegmenter(args.seeder, device)
    seed_mask = None
    if args.seed_mask:
        seed_mask = cv2.imread(args.seed_mask, cv2.IMREAD_GRAYSCALE)
        if seed_mask is None:
            raise SystemExit(f"cannot read {args.seed_mask}")
    if segmenter is None and seed_mask is None:
        raise SystemExit("--seeder none needs --seed-mask")

    matter = StreamingMatter(
        net, cfg, device, segmenter,
        warmup=args.warmup, check_every=args.check_every,
        drift_iou=args.drift_iou, drift_patience=args.drift_patience, seed_mask=seed_mask,
    )

    ddir = data_dir()
    store = StateStore(ddir / "state.json")
    bgman = BackgroundManager(ddir / "backgrounds", (out_h, out_w), device, store)
    spec = None
    if args.background:
        spec = bgman.parse_cli(args.background)
    else:
        spec = bgman.restore()
    try:
        bgman.select(spec or {"type": "blur", "strength": 12})
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load background %s (%s); using blur", spec, exc)
        bgman.select({"type": "blur", "strength": 12})

    if args.input:
        source = FileSource(args.input, realtime=not args.no_pace, loop=args.loop)
    else:
        source = CameraSource(args.camera, args.camera_size, args.camera_fps, args.camera_format)

    display_mode, display_index = _choose_display(args, (out_w, out_h))
    sinks = []
    display = None
    if display_mode != "none":
        display = DisplaySink()
        sinks.append(display)
    if args.v4l2:
        sinks.append(V4L2Sink(args.v4l2, (out_w, out_h), args.output_fps))
    if args.record:
        fps = source.fps if args.input else args.output_fps
        sinks.append(FileSink(args.record, (out_w, out_h), fps))

    settings = Settings(
        internal_size=args.internal_size, refine=args.refine, gf_radius=args.gf_radius, gf_eps=args.gf_eps,
        defringe=not args.no_defringe, mirror=args.mirror,
    )
    preview = PreviewHub()
    stop = threading.Event()
    calibrator = SyncCalibrator()
    engine = Engine(
        source, matter, bgman, sinks, preview, settings, (out_w, out_h), device,
        precision=args.precision, max_frames=args.max_frames, profile=args.profile, stop=stop,
        calibrator=calibrator,
    )

    def video_latency_ms():
        shown = display.flip_latency.pct(50) if display is not None else None
        return shown if shown is not None else engine.t_latency.pct(50)

    avsync = AVSync(store, video_latency_ms)
    if args.audio_delay:
        if args.audio_delay == "auto":
            avsync.update(mode="auto")
        else:
            avsync.update(mode="manual", manual_ms=float(args.audio_delay))
    bridge = None
    src_name, sink_name = _choose_audio(args)
    if src_name and sink_name:
        bridge = pulse_bridge(src_name, sink_name, avsync.target_ms, calibrator.audio_chunk)
        bridge.muted = args.mute
        avsync.sink_latency_ms = lambda: bridge.sink_latency * 1000
        log.info("audio: %s -> %s (delay %s)", src_name, sink_name,
                 "auto" if avsync.cfg["mode"] == "auto" else f"{avsync.cfg['manual_ms']:.0f} ms")

    server = None
    if not args.no_control:
        server = ControlServer(engine, preview, args.control_host, args.control_port,
                               audio=bridge, avsync=avsync, calibrator=calibrator)
        server.start()

    def on_key(key: str) -> None:
        if key == "r":
            matter.request_reseed("key")
        elif key == "m" and bridge is not None:
            bridge.muted = not bridge.muted
            log.info("mic %s", "muted" if bridge.muted else "live")

    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    engine.start()
    if bridge is not None:
        bridge.start()
    try:
        if display is not None:
            run_display(display, stop, (out_w, out_h), display_mode == "fullscreen", display_index, on_key)
        else:
            while not stop.is_set():
                stop.wait(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if bridge is not None:
            bridge.close()
        engine.join(timeout=5)
        source.close()
        for s in sinks:
            try:
                s.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("closing %s: %s", s.name, exc)
        bgman.close()
        if server is not None:
            server.close()
    st = engine.stats()
    log.info("done: %d frames, proc p50 %s ms, p95 %s ms", st["frames"], st["proc_ms_p50"], st["proc_ms_p95"])
    if torch.cuda.is_available() and device.type == "cuda":
        log.info("peak GPU memory %.2f GB", torch.cuda.max_memory_allocated(device) / 1e9)
    return 0


def cmd_detect(args) -> int:
    from mattecast.detect import (
        find_camera_mic,
        find_capture_audio,
        find_capture_monitor,
        list_hdmi_ports,
        list_monitors,
    )
    from mattecast.sources import list_cameras

    print("Monitors (xrandr EDID):")
    mons = list_monitors()
    if not mons:
        print("  none found (no X session, or Wayland without Xwayland EDID)")
    for m in mons:
        if not m.connected:
            continue
        geo = f"{m.width}x{m.height}+{m.x}+{m.y}" if m.active else "off"
        rate = f" @{m.refresh:.0f}Hz" if m.refresh else ""
        print(f"  display {m.sdl_index if m.sdl_index is not None else '-'}: {m.output:10s} {geo}{rate}"
              f"  {'primary ' if m.primary else ''}EDID {m.edid_vendor} {m.edid_name!r}")
    mon = find_capture_monitor(args.capture_name)
    print("  -> capture card:", f"display {mon.sdl_index} ({mon.output})" if mon else "not found")

    print("\nHDMI/DP audio ports:")
    for p in list_hdmi_ports():
        print(f"  {p.sink:60s} {'available' if p.available else 'unplugged':10s} {p.product or '-'}  [{p.description}]")
    port = find_capture_audio(args.capture_name)
    print("  -> capture card audio:", port.sink if port else "not found")

    print("\nCameras and their microphones:")
    for dev, name in list_cameras():
        mic = find_camera_mic(dev)
        print(f"  {dev}: {name}  mic: {mic or '-'}")
    return 0


def cmd_audio(args) -> int:
    from mattecast.audio import list_pulse

    print("Microphones (use with --audio-in, any unique part of the name works):")
    for name, state in list_pulse("sources"):
        if not name.endswith(".monitor"):
            print(f"  {name}  [{state}]")
    print("\nOutputs (use with --audio-out; the GPU's HDMI/DP ports show up as alsa_output.pci-...hdmi-stereo*):")
    for name, state in list_pulse("sinks"):
        print(f"  {name}  [{state}]")
    print("\nA GPU exposes one HDMI/DP port at a time. List the others with")
    print("  pactl list cards | grep -E 'Name: alsa_card|output:hdmi'")
    print("and pass the full sink name of the port you want; mattecast switches the card to it.")
    return 0


def cmd_displays(args) -> int:
    from mattecast.sinks import list_displays

    driver, sizes = list_displays()
    print(f"SDL video driver: {driver}")
    for i, (w, h) in enumerate(sizes):
        print(f"  {i}: {w}x{h}")
    if shutil.which("xrandr"):
        print("\nxrandr --listmonitors:")
        subprocess.run(["xrandr", "--listmonitors"], check=False)
    print("\nRun `mattecast identify` to flash each index on its monitor, or `mattecast detect`.")
    return 0


def cmd_identify(args) -> int:
    from mattecast.sinks import identify_displays

    identify_displays(args.seconds)
    return 0


def cmd_cameras(args) -> int:
    from mattecast.sources import list_cameras

    cams = list_cameras()
    if not cams:
        print("no /dev/video* devices")
    for dev, name in cams:
        print(f"  {dev}: {name}")
    print("\nFormats for one device: v4l2-ctl -d /dev/video0 --list-formats-ext")
    return 0


def cmd_fetch(args) -> int:
    import torch

    from mattecast.models import ensure_matanyone_source, ensure_weights
    from mattecast.seeder import PersonSegmenter

    print("MatAnyone 2 source:", ensure_matanyone_source())
    print("MatAnyone 2 weights:", ensure_weights())
    for arch in ("deeplabv3_resnet50",) if not args.all else ("deeplabv3_resnet50", "deeplabv3_mobilenet"):
        PersonSegmenter(arch, torch.device("cpu"))
        print("person detector ready:", arch)
    print("cache:", cache_dir())
    return 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mattecast", description="MatAnyone 2 virtual background for video calls")
    ap.add_argument("--version", action="version", version=f"mattecast {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="start the pipeline")
    _add_run_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("displays", help="list monitors and their indices")
    p.set_defaults(func=cmd_displays)
    p = sub.add_parser("identify", help="show each monitor's index on that monitor")
    p.add_argument("--seconds", type=float, default=4.0)
    p.set_defaults(func=cmd_identify)
    p = sub.add_parser("detect", help="show which monitor, HDMI audio port and mic would be used")
    p.add_argument("--capture-name", metavar="TEXT")
    p.set_defaults(func=cmd_detect)
    p = sub.add_parser("audio-devices", help="list microphones and outputs for --audio-in/--audio-out")
    p.set_defaults(func=cmd_audio)
    p = sub.add_parser("cameras", help="list V4L2 cameras")
    p.set_defaults(func=cmd_cameras)
    p = sub.add_parser("fetch-models", help="download MatAnyone 2 and the person detector ahead of time")
    p.add_argument("--all", action="store_true", help="also fetch the mobilenet detector")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("PIL", "matanyone2", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    t0 = time.monotonic()
    code = args.func(args)
    log.debug("exit after %.1fs", time.monotonic() - t0)
    sys.exit(code)
