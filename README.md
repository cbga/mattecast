# mattecast

**Studio-quality virtual backgrounds for any video call, computed on a spare NVIDIA GPU
and delivered to your meeting computer as a plain webcam and microphone.**

Zoom, Teams, Meet and FaceTime blur or replace your background with small models that
have to run on a laptop. mattecast runs a research-grade video matting model
([MatAnyone 2](https://github.com/pq-yang/MatAnyone2), CVPR 2026) on a desktop GPU
instead, so hair, fingers and anything that moves keep clean edges. The finished
picture leaves the GPU computer as an ordinary HDMI signal, and a USB capture card
turns it into a normal webcam on the computer you actually take the call on. Nothing
needs to be installed on that computer, and it spends no power on the effect.

```
 webcam ──USB──> GPU computer (Linux) ──HDMI──> capture card ──USB──> meeting computer ──> Zoom
                 cuts you out                   (Cam Link etc.)        (Mac, Windows, Linux)
                 adds your background
                 delays the mic to match
```

- **Plug and play.** Finds the capture card, its HDMI audio and the webcam's microphone on its own.
- **No clicking to start.** Detects the person automatically and recovers if tracking goes wrong.
- **Sharp edges.** Hair-level detail at 1080p, without the colored halo of the old wall.
- **Any background.** Blur, a color, a photo, a looping GIF or a looping video, switchable live.
- **Microphone in sync.** Your mic travels down the same cable, delayed to match your lips.
- **Web control panel.** Preview, backgrounds, mute and quality settings from any browser.
- **Private.** Everything runs on your own computer. Video never leaves it.

> **Status: early.** Developed and tested on one setup: Ubuntu, an RTX 5090, an Elgato
> Cam Link 4K, a Logitech Brio webcam, and Zoom on a Mac. Other hardware should work
> but is untested. **Non-commercial use only** (see [License](#license)).

## Contents

- [How it works, in one minute](#how-it-works-in-one-minute)
- [What you need](#what-you-need)
- [Setup, step by step](#setup-step-by-step)
- [Everyday use](#everyday-use)
- [The control panel](#the-control-panel)
- [Command reference](#command-reference)
- [FAQ and troubleshooting](#faq-and-troubleshooting)
- [Technical details](#technical-details)
- [License](#license) and [Credits](#credits)

## How it works, in one minute

You use **two computers**:

- The **GPU computer** is a Linux desktop with an NVIDIA graphics card. The webcam is
  plugged into it, and mattecast runs on it.
- The **meeting computer** is where you open Zoom (or Teams, Meet, FaceTime...). It
  can be a Mac, a Windows PC or a Linux laptop. You install nothing on it.

Between them sits an **HDMI capture card**, a small USB stick with an HDMI input. To the
GPU computer it looks like a second monitor. To the meeting computer it looks like a
webcam with a microphone. mattecast draws the finished video full screen on that
"monitor" and plays the microphone through its HDMI audio, so whatever the GPU
computer shows there arrives in Zoom as your camera and mic.

## What you need

### Hardware

| Item | Details |
| --- | --- |
| GPU computer | Linux desktop with an NVIDIA RTX graphics card. Tested on an RTX 5090. Other RTX cards should work, possibly with a lower quality setting (see [FAQ](#the-video-stutters-or-proc-is-above-33-ms)). |
| Webcam | Any USB webcam, plugged into the GPU computer. If it has a microphone, mattecast uses it. |
| HDMI capture card | A "UVC" capture card that needs no driver: Elgato Cam Link 4K (tested), Magewell USB Capture HDMI, AVerMedia Live Streamer, and similar. Avoid the cheapest USB 2.0 sticks; they compress the picture and smear fine detail. |
| HDMI cable | From a free output **on the graphics card** to the capture card. If the card only has a DisplayPort free, use a DisplayPort to HDMI cable. |
| Meeting computer | Any computer with a USB 3 port (blue inside, or USB-C) for the capture card. |

### Software

- **GPU computer:** Ubuntu 22.04 or 24.04 with the desktop (other Linux distributions
  work too, but the commands below are for Ubuntu), the NVIDIA driver, and about
  **15 GB of free disk space**. An internet connection is needed during setup.
- **Meeting computer:** nothing to install.

Setup takes about 30 to 60 minutes the first time, mostly waiting for downloads.

## Setup, step by step

**How to read the commands.** Every gray box is something you type into a **Terminal**
on the GPU computer. Open one with **Ctrl + Alt + T**. Copy a whole box, paste it into
the terminal with **Ctrl + Shift + V**, and press **Enter**. Lines that start with `#`
are comments; you can paste them too, they do nothing. When a command asks for your
password, type your login password (nothing appears while you type; that is normal)
and press Enter.

After each step there is a **You should see** note. If you see something else, the
linked FAQ entry explains what to do.

### Step 1. Check the NVIDIA driver

```bash
nvidia-smi
```

**You should see** a table with your graphics card's name (for example
`NVIDIA GeForce RTX 4080`) and a line `Driver Version: 5xx.xx`. The driver version
should be **570 or newer** (required for RTX 50 series cards, recommended for all).

If the command is not found, or the version is older, install the recommended driver
and restart the computer:

```bash
sudo ubuntu-drivers install
sudo reboot
```

Still not working: see [FAQ: nvidia-smi does not work](#nvidia-smi-is-not-found-or-cannot-talk-to-the-driver).

### Step 2. Install a few helper tools

```bash
sudo apt update
sudo apt install -y git curl tmux pulseaudio-utils v4l-utils x11-xserver-utils
```

These are small standard tools: `git` downloads the code, `pulseaudio-utils` lets
mattecast route audio, `v4l-utils` talks to webcams, `x11-xserver-utils` talks to
monitors, and `tmux` keeps programs running after you close a remote session.

**You should see** the installation finish without red `E:` errors.

### Step 3. Install uv (the Python installer mattecast uses)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

**You should see** a line like `uv 0.x.y`. uv installs the right Python version and
all of mattecast's packages into its own folder, without touching the rest of your
system.

If `uv: command not found` appears, close the terminal, open a new one, and try
`uv --version` again.

### Step 4. Download mattecast

```bash
cd ~
git clone https://github.com/YOUR-GITHUB-NAME/mattecast.git
cd mattecast
```

**You should see** a new folder `mattecast` in your home folder, and the terminal
prompt now ends with `mattecast$`. All later commands must be run inside this folder.
If you open a new terminal later, go back into it first with `cd ~/mattecast`.

### Step 5. Install mattecast's packages

```bash
uv sync
```

This downloads several gigabytes (PyTorch and NVIDIA's libraries), so it can take 5 to
20 minutes. Then check that the graphics card is usable:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**You should see** `True` followed by your graphics card's name.
Seeing `False` or an error: [FAQ: PyTorch cannot use the GPU](#pytorch-says-false-or-no-kernel-image-is-available).

### Step 6. Download the AI models

```bash
uv run mattecast fetch-models
```

This downloads about 300 MB once (the MatAnyone 2 matting model and a person
detector). **You should see** it finish with `person detector ready`.

### Step 7. Connect the cables

1. Plug the **webcam** into a USB port of the GPU computer.
2. Connect an HDMI cable from a free output **on the graphics card** (the ports on the
   card itself, usually low on the back of the computer, not the ones higher up on
   the motherboard) to the **capture card**.
3. Plug the **capture card** into a USB 3 port of the **meeting computer**.

### Step 8. Turn the capture card into a second screen

On the GPU computer, open **Settings → Displays**. A new display has appeared (often
named after the capture card).

1. At the top, choose **Join Displays** (not Mirror).
2. Click the new display and set **Resolution** to **1920 × 1080** and **Refresh Rate**
   to **60 Hz**. Capture cards often start at 3840 × 2160 at 30 Hz; that works, but
   1080p at 60 Hz is faster and sharper.
3. Click **Apply**, then **Keep Changes**.

**You should see** your desktop wallpaper stretch onto the new screen.

**Recommended:** on the Ubuntu login screen, click your name, then the gear icon in the
bottom right corner, and pick **Ubuntu on Xorg** before entering your password.
mattecast finds the capture card automatically only in an Xorg session. On the default
Wayland session it still works, but you have to tell it which screen to use
([FAQ](#detect-says-no-x-session-or-wayland)). To check which one you are in, run
`echo $XDG_SESSION_TYPE` in a terminal on the desktop: it prints `x11` or `wayland`.

### Step 9. Check the capture card on the meeting computer

- **Mac:** open **QuickTime Player**, choose **File → New Movie Recording**, click the
  small arrow next to the red record button, and pick the capture card (for example
  **Cam Link 4K**) under Camera.
- **Windows:** open the **Camera** app and use the switch-camera button until you see
  the capture card.

**You should see** the GPU computer's second screen (its wallpaper). Close QuickTime or
the Camera app afterwards: an open preview can keep the camera busy, and QuickTime can
play the capture card's sound back to you as an echo.
Black picture or no signal: [FAQ](#the-meeting-computer-shows-black-or-no-signal).

### Step 10. Let mattecast check everything

```bash
uv run mattecast detect
```

**You should see** three findings, similar to this (names differ per computer):

```
Monitors (xrandr EDID):
  display 0: HDMI-0     1920x1080+0+0 @60Hz  primary EDID ... 'DELL U2720Q'
  display 1: DP-1       1920x1080+1920+0 @60Hz  EDID EGT 'Cam Link 4K'
  -> capture card: display 1 (DP-1)

HDMI/DP audio ports:
  alsa_output.pci-0000_01_00.1.hdmi-stereo-extra1   available  Cam Link 4K
  -> capture card audio: alsa_output.pci-0000_01_00.1.hdmi-stereo-extra1

Cameras and their microphones:
  /dev/video0: Logitech BRIO  mic: alsa_input.usb-046d_Logitech_BRIO_...
```

- The **capture card** line tells mattecast which screen to draw on.
- The **capture card audio** line tells it where to send the microphone.
- The **camera** line shows which webcam and microphone belong together.

If any of these says `not found`, the matching FAQ entry explains why:
[screen](#mattecast-does-not-find-the-capture-card-screen),
[audio](#the-log-says-audio-forwarding-off),
[camera](#cannot-open-camera-devvideo0).

### Step 11. Start mattecast

```bash
uv run mattecast run
```

The first start takes about 20 seconds while the models load and the graphics card
tunes itself. Then the capture card screen turns into a full screen video of you with
a blurred background. The terminal prints a status line every 5 seconds:

```
I mattecast: capture card 'Cam Link 4K' on DP-1 (1920x1080@60Hz) -> display 1
I mattecast: audio: alsa_input.usb-046d_Logitech_BRIO_... -> alsa_output.pci-...hdmi-stereo-extra1 (delay auto)
I mattecast.control: control panel: http://localhost:8765/
I mattecast.matter: seeded at 960x540 (#1)
I mattecast.pipeline: 30.0 fps | proc p50 18.1 ms p95 20.6 ms | latency p50 25.0 ms | model 960x540 | tracking yes iou 0.93 | bg blur
```

- `30.0 fps` is how many frames per second go out. 30 is the goal.
- `proc` is how long each frame takes to process. It must stay **below 33 ms** for 30 fps.
- `tracking yes` means it has found you. `searching` means nobody is in view yet.

To stop mattecast, click the terminal and press **Ctrl + C**.

### Step 12. Pick the camera and microphone in Zoom

On the meeting computer, in Zoom go to **Settings → Video** and choose the capture card
(for example **Cam Link 4K**) as the camera. Then in **Settings → Audio**, choose the
same capture card as the **microphone**. Other apps have the same two settings.

On a Mac, the first time an app uses the capture card, macOS asks for permission to use
the camera and the microphone. Click **Allow**. If you clicked Don't Allow earlier,
turn it on in **System Settings → Privacy & Security → Camera** and **→ Microphone**.

Do not turn on Zoom's own virtual background or blur; mattecast already did the work.

### Step 13. Open the control panel

On the GPU computer, open **http://localhost:8765** in a web browser. You see a live
preview and all the settings ([tour](#the-control-panel)).

To use the panel from the meeting computer or your phone instead, start mattecast with
an extra option that says which network address to listen on:

```bash
# Only people on your own home network can open it:
uv run mattecast run --control-host 0.0.0.0

# If you use Tailscale, the panel is reachable only through your Tailscale account:
uv run mattecast run --control-host $(tailscale ip -4)
```

Then browse to `http://<the GPU computer's address>:8765`. Find the address with
`hostname -I` (first number) or, for Tailscale, `tailscale ip -4`.
The panel has no password, so never do this on a public network such as a café or
hotel Wi-Fi.

### Step 14. Choose a background

In the panel, under **Background**:

- Click **Original**, **Blur** or **Color** for the built-in ones.
- Click **+ Upload** (or drag a file onto the page) to add your own photo, GIF or short
  video. It is used immediately and stays in the list for next time.

Or choose one when starting: `uv run mattecast run --background ~/Pictures/office.jpg`.

### Step 15. Fix lip sync (optional, two minutes)

mattecast delays your microphone so it matches your lips, and it already guesses the
right amount. To measure it exactly, your phone flashes and beeps in front of the
webcam, and mattecast times when the camera sees each flash and when the microphone
hears each beep.

> [!IMPORTANT]
> **Do this test while the preview shows Camera, not Output.** Camera is the plain
> webcam picture with nothing cut out. In the normal Output view the cut-out usually
> removes the phone and paints your background over it, so you cannot tell whether the
> webcam actually sees the flashes. The panel switches to Camera by itself when you
> click **Measure with your phone**. If the preview still shows your background, click
> **Camera** above the preview yourself.

Before you start, make sure your phone can open the panel. The test page lives on the
GPU computer, so mattecast must have been started with `--control-host 0.0.0.0` (phone
on the same Wi-Fi) or `--control-host $(tailscale ip -4)` (Tailscale on the phone), as
in Step 13. If you started it without that option, stop it with `Ctrl+C` and start it
again with the option.

1. In the panel, click **Measure with your phone**.
   - The preview switches to **Camera** (the button above the preview lights up).
   - A box shows the address to open on your phone, ending in `/sync`. If it says your
     phone cannot reach mattecast yet, see the paragraph above.
2. Get the phone ready:
   - Turn the **volume up** and **silent mode off**.
   - Turn the **screen brightness up**.
   - **Disconnect Bluetooth headphones or speakers**, otherwise the beep plays there
     instead of in the room.
3. Open the `/sync` address on the phone and press **Start**.
4. Right away, **turn the phone around so its screen faces the webcam lens**, 20 to 40
   cm in front of it, with the phone's speaker pointing toward the webcam. The screen
   now flashes white and beeps once a second.
5. Look at the **Camera** preview in the panel: the whole phone screen should be in the
   picture and you should see it flash. Move the phone until it is.
6. **Hold still and keep the room quiet for about 15 seconds.** The panel counts the
   flashes (for example `6/10 flashes: camera 48 ms behind mic`).
7. When it says **done**, press **Apply** in the panel (or on the phone). The preview
   goes back to the view you had before.

If nothing is detected, see
[The phone sync test finds nothing](#the-phone-sync-test-finds-nothing-or-the-spread-is-large).

That is all. mattecast is set up.

## Everyday use

**Start:** open a terminal and run

```bash
cd ~/mattecast
uv run mattecast run
```

Add any options you like, for example `--control-host 0.0.0.0`. Your last background
and audio settings are remembered.

**Stop:** press **Ctrl + C** in that terminal.

**Keys** (click the mattecast window first): `r` detects the person again, `m` mutes
or unmutes the microphone, `q` quits.

**Starting it from another computer over SSH** works; mattecast attaches to the
desktop that is logged in on the GPU computer. Run it inside `tmux` so it keeps
running when you disconnect:

```bash
tmux new -s cam                    # opens a session named "cam"
cd ~/mattecast && uv run mattecast run
# press Ctrl + B, then D, to leave it running in the background
tmux attach -t cam                 # later: come back to it
```

**Update to a newer version:**

```bash
cd ~/mattecast
git pull
uv sync
```

**Start automatically when you log in (optional).** Create a small service file:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/mattecast.service <<'EOF'
[Unit]
Description=mattecast virtual camera
After=graphical-session.target
PartOf=graphical-session.target

[Service]
WorkingDirectory=%h/mattecast
ExecStart=%h/.local/bin/uv run mattecast run
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical-session.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now mattecast
```

Watch its messages with `journalctl --user -u mattecast -f`, stop it with
`systemctl --user stop mattecast`, and turn autostart off with
`systemctl --user disable mattecast`. Put extra options at the end of the
`ExecStart=` line.

## The control panel

| Area | What it does |
| --- | --- |
| Top bar | Frames per second, processing time, total delay, model resolution, and whether you are being tracked. |
| Preview | **Output** is what Zoom receives. **Matte** shows the cut-out mask (white is you). **Camera** shows the plain webcam picture with nothing cut out; use it to aim the phone during the sync test. The switch only affects the preview, never what Zoom gets. |
| Re-detect person | Forgets the current cut-out and finds the person again. Use it if part of you went missing. |
| Background | Original, Blur (with strength slider), Color, and your uploaded images, GIFs and videos. Hover a tile and click × to delete it. |
| Microphone | Level meter, Mute, the audio delay (Auto or a fixed value), the camera offset, and **Measure with your phone** (the lip-sync test, which switches the preview to Camera while it runs). |
| Quality | Model resolution (higher is sharper and slower), edge upsampling, fringe removal on hair, and mirroring. |
| Events | When the person was detected, lost or re-detected. |

## Command reference

| Command | What it does |
| --- | --- |
| `mattecast run` | Start. All options are below. |
| `mattecast detect` | Show which screen, HDMI audio port and microphone would be used. |
| `mattecast cameras` | List webcams. |
| `mattecast displays` | List screens and their numbers. |
| `mattecast identify` | Show each screen's number in big digits on that screen. |
| `mattecast audio-devices` | List microphones and audio outputs. |
| `mattecast fetch-models` | Download the models ahead of time. |

Prefix every command with `uv run` inside the `mattecast` folder, for example
`uv run mattecast detect`. Add `--help` to any command to see all its options.

### Options for `mattecast run`

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | `/dev/video0` | Which webcam. |
| `--camera-size`, `--camera-fps`, `--camera-format` | `1920x1080`, `30`, `MJPG` | What to ask the webcam for. |
| `--display` | `auto` | `auto`: full screen on the capture card if found, else a small preview window. Also `fullscreen`, `window`, `none`. |
| `--display-index` | detected | Screen number to use (see `mattecast identify`). |
| `--capture-name` | | Part of your capture card's name, if it is not recognized. |
| `--set-mode` | off | Switch the capture card screen to 1920x1080 at 60 Hz on start. |
| `--background` | last used, else `blur` | `blur`, `blur:20`, `color:#00b140`, `none`, a file path, or the name of an uploaded file. |
| `--audio-in` | `auto` | Microphone: `auto` (the webcam's), `none`, or part of a name from `mattecast audio-devices`. |
| `--audio-out` | `auto` | Where the mic goes: `auto` (the capture card), `none`, or a name. |
| `--audio-delay` | last used, else `auto` | `auto` or a fixed delay in milliseconds, for example `150`. |
| `--mute` | off | Start with the microphone muted. |
| `--control-host` | `127.0.0.1` | Network address for the panel. `0.0.0.0` means every network. |
| `--control-port` | `8765` | Panel port. |
| `--internal-size` | `540` | Model resolution. `360` or `432` are faster, `720` is sharper. |
| `--size` | `1920x1080` | Output resolution. |
| `--mirror` | off | Flip left and right. |
| `--refine`, `--no-defringe` | guided, on | Edge options (also in the panel). |
| `--seeder` | `deeplabv3_resnet50` | Person detector. `deeplabv3_mobilenet` is lighter. |
| `--check-every`, `--drift-iou`, `--drift-patience` | `15`, `0.5`, `3` | How often and how strictly tracking is checked. |
| `--v4l2` | | Also create a virtual camera on this Linux computer ([FAQ](#can-i-use-it-without-a-capture-card)). |
| `--record` | | Also save the output to an MP4 file. |
| `--input` | | Use a video file instead of a webcam, for testing. Add `--loop` to repeat it. |
| `--profile` | off | Show how long each processing stage takes. |

### Where mattecast keeps its files

| Folder | Contents |
| --- | --- |
| `~/mattecast` | The program (wherever you cloned it). |
| `~/.cache/mattecast` | Downloaded models (about 300 MB). Change with the `MATTECAST_CACHE` environment variable. |
| `~/.local/share/mattecast` | Your uploaded backgrounds and saved settings. Change with `MATTECAST_DATA`. |

## FAQ and troubleshooting

### Installation

#### `nvidia-smi` is not found, or cannot talk to the driver

The NVIDIA driver is missing or not loaded. Install the recommended one and restart:

```bash
sudo ubuntu-drivers install
sudo reboot
```

If `ubuntu-drivers` is not available (older Ubuntu), use **Software & Updates →
Additional Drivers**, choose the newest "NVIDIA driver (proprietary, tested)", click
Apply Changes, and restart. After Secure Boot prompts during installation, choose
**Enroll MOK** on the blue screen at the next boot and enter the password you set.

#### `uv: command not found`

Open a new terminal (the installer only updates new ones), or run
`source $HOME/.local/bin/env`. If it is still missing, repeat Step 3.

#### `uv sync` is very slow or stops with an error

It downloads several gigabytes. Check the internet connection and free disk space
(`df -h ~` should show at least 15 GB available), then simply run `uv sync` again; it
continues where it stopped. Behind a company proxy, set `HTTPS_PROXY` first.
mattecast needs a 64-bit Intel or AMD processor (x86_64) and Linux; `uv sync` fails on
other systems.

#### PyTorch says `False`, or "no kernel image is available"

- `False`: the driver is missing or too old for this PyTorch build (CUDA 12.8). Update
  the driver to 570 or newer (Step 1), restart, and check again.
- "no kernel image is available for execution on the device": the installed PyTorch
  does not support your graphics card. Delete the environment and install again:
  `rm -rf .venv && uv sync`.
- GTX 10-series and older cards are not supported by current PyTorch builds.

#### `fetch-models` cannot download

The models come from `github.com` (MatAnyone 2) and `download.pytorch.org` (person
detector). If one of those sites is blocked on your network, download the matting
model manually from
<https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth>
on another computer, copy it over, and start with `--weights /path/to/matanyone2.pth`.
If the MatAnyone 2 source cannot be downloaded, clone
<https://github.com/pq-yang/MatAnyone2> elsewhere and pass `--matanyone-src /path/to/MatAnyone2`.

#### Can I run mattecast on Windows or macOS, or with an AMD or Intel GPU?

Not at the moment. The GPU computer must run Linux with an NVIDIA card. The meeting
computer can be anything. Without a GPU it falls back to the processor, which manages
less than one frame per second.

### Screen and capture card

#### mattecast does not find the capture card screen

1. Run `uv run mattecast detect`. Is the capture card listed under Monitors at all?
   - Not listed: the screen is off or the cable is loose. Repeat Step 7 and Step 8.
   - Listed, but not picked as the capture card: its name is unusual. Start with
     `--capture-name "part of the name"` using the name shown by `detect`.
2. You can always choose the screen yourself. Run `uv run mattecast identify`; each
   screen shows its number in big digits for a few seconds (the capture card's number
   shows up on the meeting computer, in QuickTime or the Camera app). Then start with
   `uv run mattecast run --display fullscreen --display-index 1` (your number).

#### `detect` says "no X session, or Wayland"

Automatic screen detection needs an Xorg desktop session. Either log out and pick
**Ubuntu on Xorg** on the login screen (gear icon, bottom right), or keep Wayland and
choose the screen yourself with `--display fullscreen --display-index N` as described
above. Audio and camera detection work in both.

#### "SDL picked the invisible 'offscreen' video driver"

mattecast could not find a logged-in desktop to draw on. Log in on the GPU computer
itself (a remote desktop like Moonlight or VNC counts), then start mattecast again.
It is fine to start mattecast from SSH as long as someone is logged in to the desktop.

#### The meeting computer shows black, or "no signal"

- The HDMI cable must go into the **graphics card**, not the motherboard's own port.
- In **Settings → Displays**, the capture card screen must be on and set to **Join
  Displays**, not Mirror or off.
- Plug the capture card into a **USB 3** port on the meeting computer, directly rather
  than through a hub.
- Try another output on the graphics card, or another cable.
- Make sure no other app on the meeting computer (Photo Booth, QuickTime, OBS) is using
  the capture card at the same time.

#### The meeting computer shows my desktop wallpaper instead of the video

mattecast is not running, or it opened on a different screen. Look at the terminal: the
line `display N: fullscreen` tells you which screen it chose. See
[mattecast does not find the capture card screen](#mattecast-does-not-find-the-capture-card-screen).

#### The picture looks washed out, or blacks look gray or crushed

The graphics card is sending "limited range" color to the capture card. Open
**NVIDIA X Server Settings** (`nvidia-settings`), select the capture card's display in
the left list, and set **Color Range** to **Full** on its Controls tab.

#### The capture card screen is 3840 × 2160 at 30 Hz

It works, but 1920 × 1080 at 60 Hz reacts faster. Change it in **Settings → Displays**
(Step 8), or start mattecast with `--set-mode`. If you type `xrandr` commands over SSH
and get "Can't open display", put `DISPLAY=:0` in front, for example
`DISPLAY=:0 xrandr --output DP-1 --mode 1920x1080 --rate 60` (use the output name from
`mattecast detect`).

#### The mouse pointer or the top bar shows on the capture card screen

mattecast's full screen window hides the pointer and covers the top bar. If you still
see them, mattecast is not running full screen on that screen; check the
`display N: fullscreen` line in the terminal. Keep your mouse on your main screen.

### Camera and picture

#### "cannot open camera /dev/video0"

- List the webcams with `uv run mattecast cameras`. Many webcams create several
  entries; the first one is usually the picture. Try another with `--camera /dev/video2`.
- Close any other program using the webcam (a browser tab with a video call, Cheese, OBS).
- "Permission denied": add yourself to the video group, then log out and back in:
  `sudo usermod -aG video $USER`.

#### It only runs at about 15 fps

The webcam lengthens its exposure in dim light and halves its frame rate. Add light
in front of you, or tell the webcam to keep 30 fps (the picture gets a bit darker):

```bash
v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0
# older systems call it: v4l2-ctl -d /dev/video0 -c exposure_auto_priority=0
```

This resets when the webcam is unplugged. Also make sure the webcam is in a USB 3 port.

#### It shows only the background, or the panel says "tracking searching"

mattecast has not found a person yet. Sit in view of the camera, facing it, with your
head and shoulders visible and reasonable light. It starts tracking within a second.
If nobody is in view it deliberately shows just the background.

#### Part of me disappears, or the chair or an object is included

- Click **Re-detect person** in the panel (or press `r` in the mattecast window) while
  sitting normally.
- Look at the **Matte** preview: white is what counts as you. Objects you hold are
  often kept, which is usually what you want.
- Good, even light on your face and a background that is darker or lighter than you
  make a big difference.

#### There is a halo or glow around my hair

Make sure **Remove color fringe on hair edges** is on and **Edge upsampling** is set to
**Guided filter** in the panel. A bright window directly behind you makes this harder;
more light on your face helps.

#### The video stutters, or `proc` is above 33 ms

- Lower the model resolution in the panel (**Quality → Model resolution**) to 432p or
  360p, or start with `--internal-size 432`.
- Close games or other programs using the graphics card (`nvidia-smi` lists them).
- Use `--seeder deeplabv3_mobilenet` for a lighter person detector.
- Start with `--profile` to see which stage is slow.

#### The picture freezes for a few seconds right after starting

Normal. The first frame waits while the models load and the graphics card tunes itself
(about 20 seconds). After that it runs smoothly.

#### My image is mirrored, or should be

Zoom mirrors your self view only for you; others see you the right way round. Leave
mirroring off unless you have a specific reason, then use `--mirror` or the panel.

#### It uses a lot of graphics memory

On an RTX 5090 the peak is about 14 GB, mostly used once while the graphics card picks
its fastest settings at startup. Cards with less memory choose smaller settings on
their own.

### Audio

#### Zoom does not offer the capture card as a microphone, or it stays silent

- On a Mac, allow microphone access (**System Settings → Privacy & Security →
  Microphone**) for Zoom and restart Zoom.
- Look at the mattecast terminal for a line starting with `audio:`. If it says
  `audio forwarding off`, see the next question.
- Look at the level meter in the panel's Microphone section while you speak. If it
  moves, sound is leaving the GPU computer. If it does not move, check that the panel
  does not say Muted and that the right microphone is used (`mattecast detect`).

#### The log says "audio forwarding off"

| Message ends with | Meaning and fix |
| --- | --- |
| `the camera has no microphone we could match` | The webcam has no mic, or mattecast could not tell which mic belongs to it. List mics with `uv run mattecast audio-devices` and pass one: `--audio-in BRIO` (any unique part of the name). |
| `no HDMI audio port reports a capture card` | The capture card's HDMI audio was not recognized. Find the graphics card's HDMI outputs with `uv run mattecast audio-devices` and try them one by one with `--audio-out hdmi-stereo-extra1` (and `-extra2`, and so on) until the meeting computer receives sound. |
| `pactl not found` | Install it: `sudo apt install pulseaudio-utils`. |

To turn the microphone feature off on purpose, use `--audio-in none` and pick the
meeting computer's own microphone in Zoom.

#### I hear my own voice (echo)

- On the meeting computer, close QuickTime or any other app that shows a live preview
  of the capture card: QuickTime plays the capture card's sound through your speakers.
- In Zoom, make sure the **speaker** is not set to the capture card.
- mattecast itself never plays the microphone anywhere except the capture card. If
  an audio device disappears for a moment (for example when the screen resolution
  changes), it disconnects the microphone instead of letting the system send it to
  your speakers, and reconnects when the capture card is back. The terminal then shows
  `audio stream was moved off ... disconnected it`.

#### My voice and lips are out of sync

Run the phone test in [Step 15](#step-15-fix-lip-sync-optional-two-minutes), with the
preview on **Camera**. If you prefer to set it by hand, switch the panel's delay to
**Manual** and adjust the slider while watching a local Zoom recording. Sound up to
about 45 ms early or 125 ms late is not noticeable to most people.

#### The phone sync test finds nothing, or the spread is large

Go through these in order:

1. **Switch the preview to Camera.** Click **Camera** above the preview. You must be
   able to see the phone in the preview. In the Output view the phone is usually cut
   away, which hides whether the webcam sees it.
2. **The phone's screen must face the webcam lens**, not you. It is easy to hold it the
   wrong way round after pressing Start.
3. **The whole screen must be in the picture**, 20 to 40 cm from the webcam, not hidden
   by your hand or fingers. Screen brightness up.
4. **The beep must be audible in the room:** volume up, silent mode off, no Bluetooth
   headphones or speakers connected to the phone, the phone's speaker not covered.
5. **Hold still and keep quiet** for the whole test. Talking, music or a fan close to the
   webcam can hide the beep.
6. If the panel says **spread** is above about 25 ms, the result is unreliable. The
   usual reason is a camera running at a low frame rate in a dark room (see
   [It only runs at about 15 fps](#it-only-runs-at-about-15-fps)): turn on
   more light and run the test again.

The test works the same whatever background is selected; the Camera preview is only
there so that you can aim the phone.

#### My other speakers or monitors connected to the graphics card lost their sound

A graphics card can send sound to only one of its HDMI or DisplayPort outputs at a
time, and mattecast switches it to the capture card. Speakers plugged into the
computer itself, USB headsets and Bluetooth are not affected.

#### How do I mute quickly?

Click **Mute** in the panel, press `m` in the mattecast window, or simply use Zoom's own
mute button.

### Control panel and network

#### The panel does not open from another computer or my phone

- Start mattecast with `--control-host 0.0.0.0` (home network) or
  `--control-host $(tailscale ip -4)` (Tailscale). Without this option the panel only
  opens on the GPU computer itself. The terminal line `control panel: http://...`
  shows where it is listening.
- Use the GPU computer's address, from `hostname -I` or `tailscale ip -4`, not `localhost`.
- The other device must be on the same home network, or signed in to the same Tailscale
  account.
- If a firewall is on (`sudo ufw status` says active), allow the port:
  `sudo ufw allow 8765/tcp`.

#### Is `--control-host 0.0.0.0` safe?

The panel has no password: anyone who can reach it can change your background, mute
you or upload files. That is fine on your own home network. Do not use it on public
Wi-Fi, and never forward port 8765 on your router. Tailscale is the safest way to use
the panel from elsewhere.

#### The phone cannot open the `/sync` page

The phone needs to reach the GPU computer just like the panel does: same Wi-Fi network
(with `--control-host 0.0.0.0`) or the Tailscale app signed in to your account. Type
the address exactly as the panel shows it, including `http://` and `:8765`. An address
that starts with `http://localhost` or `http://127.0.0.1` only works on the GPU computer
itself; if the panel cannot show any other address, restart mattecast with one of the
`--control-host` options above.

### Backgrounds

#### My video or GIF background does not load, or plays badly

Supported: JPG, PNG, WebP, BMP, TIFF images; GIF, animated WebP and APNG animations;
MP4, MOV, M4V, WebM, MKV and AVI videos. If a video does not load, convert it to a
standard MP4 with ffmpeg (`sudo apt install ffmpeg`):

```bash
ffmpeg -i input.mov -vf scale=1920:-2 -c:v libx264 -pix_fmt yuv420p -an background.mp4
```

Very long GIFs use a lot of memory; only the first 900 frames are kept. Short loops of
10 to 30 seconds work best.

#### Where are my backgrounds stored, and how do I remove them?

In `~/.local/share/mattecast/backgrounds`. Delete them in the panel (hover a tile, click
×) or in that folder.

### General

#### Can I use it without a capture card?

On the GPU computer itself, yes: mattecast can create a virtual webcam that apps on the
same Linux computer (Zoom for Linux, a browser) can use.

```bash
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="mattecast" exclusive_caps=1
uv sync --extra v4l2
uv run mattecast run --v4l2 /dev/video10 --display none --audio-in none
```

Then pick "mattecast" as the camera in the app. The module has to be loaded again after
each restart. If `v4l2loopback-dkms` fails to build on a new kernel, install a newer
version from the [v4l2loopback project](https://github.com/v4l2loopback/v4l2loopback).
For a different computer you need the capture card.

#### Does my video go anywhere?

No. Everything runs on the GPU computer. The internet is used only once, to download
the models. The control panel is only reachable from other devices if you start it with
`--control-host`.

#### Can I use it for work or commercially?

The mattecast code is MIT licensed, but the MatAnyone 2 model it downloads is licensed
for **non-commercial use only**. Using it in private calls, study and research is fine;
using it as part of a commercial product or service is not allowed without permission
from the model's authors.

#### How do I uninstall it?

```bash
rm -rf ~/mattecast ~/.cache/mattecast ~/.local/share/mattecast
```

If you set up autostart, first run `systemctl --user disable --now mattecast` and delete
`~/.config/systemd/user/mattecast.service`.

## Technical details

<details>
<summary>Pipeline, performance and lip-sync math</summary>

```
frame (BGR) -> GPU, crop/resize to output -> area-downsample to the model size
    -> MatAnyone 2 step()  (fp16, bounded memory of recent frames)
         unseeded: DeepLabV3 person mask -> closing -> seed + warmup
         every N frames: DeepLabV3 at 320p, IoU against the matte -> reseed if drifted
    -> fast color guided filter lifts the matte to full resolution
    -> blur-fusion foreground estimation removes the old background color at edges
    -> composite over the background -> uint8 -> full screen window / v4l2 / file

mic -> 10 ms blocks -> delay line (target follows measured video latency) -> HDMI audio
```

Every video stage keeps only the newest frame, so a slow step drops frames instead of
building up delay. MatAnyone 2 keeps a bounded working memory, so latency stays flat
over long calls.

Measured on an RTX 5090 at 1080p output with the model at 540p: the whole pipeline takes
p50 18 to 21 ms and p95 20 to 24 ms. MatAnyone 2 alone (300 frames):

| Model short side | p50 | p95 | Fits 30 fps? |
| --- | --- | --- | --- |
| 360 | 10.1 ms | 11.0 ms | easily |
| 540 (default) | 14.6 ms | 15.5 ms | yes, with room for the rest |
| 720 | 25.1 ms | 31.8 ms | only just |
| 1080 | 74.2 ms | 78.1 ms | no |

**Screen and audio detection.** A capture card reports a product name over HDMI (EDID).
mattecast reads it from `xrandr --verbose` to find the screen, and from the HDMI audio
port's ELD in `pactl list cards` to find the audio output. The webcam's microphone is
matched by USB vendor id and serial number.

**Lip sync.** For an event at time T, the picture reaches HDMI at `T + L_cam + L_video`
and the sound at `T + L_mic + delay + L_sink`, so

```
delay = L_video + (L_cam - L_mic) + card_lag - L_sink
```

`L_video` (camera read to display flip) is measured on every frame and `L_sink` is
reported by the sound server. `L_cam - L_mic` defaults to 50 ms and is measured by the
flash-and-beep test. `card_lag` defaults to one 60 Hz scanout (16 ms). The microphone
and HDMI run on different clocks; the delay line stretches or squeezes a 10 ms block
by a single sample when needed, which is inaudible.

**Audio routing guard.** PipeWire moves a stream to the default output when its device
disappears. mattecast watches `pactl subscribe` and drops its own stream the moment it
lands anywhere but the chosen device, then reconnects when that device returns.

</details>

## Roadmap

- TensorRT export of MatAnyone 2 for 720p at full speed and lower power.
- Automatic screen detection on Wayland.
- NDI output as a cable-free alternative to the capture card.
- Relighting the person to match the background.
- A published side-by-side comparison with Zoom, macOS and NVIDIA Broadcast.

## License

mattecast's own code is released under the MIT License (see `LICENSE`).

It downloads and runs third-party models with their own terms:

- **MatAnyone 2** code and weights: NTU S-Lab License 1.0, **non-commercial use only**.
  Using mattecast with MatAnyone 2 is therefore non-commercial.
- **DeepLabV3** weights from torchvision: BSD-3-Clause.
- **pyvirtualcam** (only with the optional `v4l2` extra): GPL-2.0.

## Credits

- Peiqing Yang, Shangchen Zhou, Kai Hao, Qingyi Tao: *MatAnyone 2: Scaling Video Matting
  via a Learned Quality Evaluator*, CVPR 2026.
- [Better Backgrounds](https://github.com/cjami/better-backgrounds) by cjami, which showed
  MatAnyone 2 running live and whose benchmark produced the model timings above.
- Marco Forte and François Pitié: *Approximate Fast Foreground Colour Estimation*, ICIP 2021.
- Huikai Wu et al.: *Fast End-to-End Trainable Guided Filter*, CVPR 2018; Kaiming He et
  al.: *Guided Image Filtering*, TPAMI 2013.
