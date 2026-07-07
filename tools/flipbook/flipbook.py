#!/usr/bin/env python3
"""flipbook.py — turn a DiffusionGemma llama-diffusion-cli run into a navigable
flip-book of per-step diffusion frames (plain-text + PNG + an HTML scrubber).

Standalone dev tool. Does NOT import into, or get imported by, the
ComfyUI-DiffusionGemma node pack. See tools/flipbook/README.md.

Usage:
    python3 flipbook.py --prompt "The capital of France is" --steps 12

Or, to only re-parse an already-captured raw stdout file (no GPU/CLI run):
    python3 flipbook.py --replay /path/to/raw_capture.txt --out ./out/replay
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import pty
import re
import struct
import subprocess
import sys
import termios
from pathlib import Path

# ---------------------------------------------------------------------------
# Grounded defaults (see CLAUDE.md task brief / decisions/ for provenance)
# ---------------------------------------------------------------------------
DEFAULT_BINARY = "/srv/dev/llama.cpp-diffusiongemma/build/bin/llama-diffusion-cli"
DEFAULT_GGUF = (
    "/mnt/storage/LLMs/unsloth/diffusiongemma-26B-A4B-it-GGUF/"
    "diffusiongemma-26B-A4B-it-Q8_0.gguf"
)
DEFAULT_STEPS = 48
DEFAULT_CANVAS_LEN = 256
DEFAULT_T_MIN = 0.4
DEFAULT_T_MAX = 0.8
DEFAULT_ENTROPY_BOUND = 0.1
DEFAULT_CONFIDENCE = 0.005

SYNC_START = "\x1b[?2026h"
SYNC_END = "\x1b[?2026l"

# Matches any ANSI CSI sequence: ESC [ params(0x30-0x3f) intermediates(0x20-0x2f) final(0x40-0x7e)
# This single pattern covers cursor moves (\x1b[<N>A/B/C/D), clear-to-EOL (\x1b[K),
# clear-screen (\x1b[J), and the synchronized-update / cursor-visibility private
# modes (\x1b[?2026h, \x1b[?2026l, \x1b[?25l, \x1b[?25h) since '?' falls in the
# parameter-byte range.
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

HEADER_RE = re.compile(
    r"diffusion step:\s*(\d+)\s*/\s*(\d+)\s*\[.*?\]\s*(\d+)%"
)


def strip_ansi(text: str) -> str:
    """Remove all ANSI CSI sequences and bare carriage returns."""
    return ANSI_RE.sub("", text).replace("\r", "")


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    slug = slug[:max_len].strip("-")
    return slug or "prompt"


def compute_temp(step_idx: int, total_steps: int, t_min: float, t_max: float) -> float:
    """Anneal temperature active at a given diffusion step.

    Grounded formula (llama.cpp `diffusion.cpp:532`, do not re-derive):
        t = t_min + (t_max - t_min) * (cur_step / S)
        cur_step = S - step_idx

    `S` (=total_steps) MUST be the run's max_steps arg, parsed from the CLI's own
    step header ("diffusion step: k/S") — never the number of frames actually
    captured. An early-stopped run (e.g. 21 of 48 steps) must still report the
    true, still-hot temperature on its last captured step; normalizing over the
    captured-frame count would hide exactly the under-anneal this overlay exists
    to surface.
    """
    cur_step = total_steps - step_idx
    return t_min + (t_max - t_min) * (cur_step / total_steps)


class Frame:
    def __init__(
        self,
        step: int,
        total_steps: int,
        percent: int,
        temp: float,
        body_lines: list[str],
        entropy_bound: float | None = None,
        confidence: float | None = None,
    ):
        self.step = step
        self.total_steps = total_steps
        self.percent = percent
        self.temp = temp
        self.body_lines = body_lines
        self.entropy_bound = entropy_bound
        self.confidence = confidence

    @property
    def label(self) -> str:
        label = f"step {self.step}/{self.total_steps} · {self.percent}% · temp {self.temp:.3f}"
        if self.entropy_bound is not None and self.confidence is not None:
            label += f" · eb={self.entropy_bound} conf={self.confidence}"
        return label


def _rstrip_blank_lines(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and out[-1].strip() == "":
        out.pop()
    return out


def _lstrip_blank_lines(lines: list[str]) -> list[str]:
    out = list(lines)
    while out and out[0].strip() == "":
        out.pop(0)
    return out


def parse_frames(
    raw: str,
    t_min: float,
    t_max: float,
    entropy_bound: float | None = None,
    confidence: float | None = None,
) -> tuple[list[Frame], list[str]]:
    """Split a raw llama-diffusion-cli stdout capture into per-step Frames.

    Blocks are delimited by the terminal "synchronized update" private-mode
    sequences \\x1b[?2026h (start) ... \\x1b[?2026l (end). Everything before the
    first start marker is a load/log preamble and is discarded. Everything
    after the *last* end marker is the CLI's post-loop plain-text canvas dump
    (no synchronized-update wrapper) plus a `total time:` / `throughput:`
    footer. That tail is a DIFFERENT output path than the per-step ANSI
    blocks — not a diffusion step — so it is deliberately NOT turned into a
    frame (it used to be, and produced a "phantom" final frame that jumped /
    duplicated the last real step). Its `total time` / `throughput` lines are
    still extracted and returned separately as run metadata.

    Returns (frames, meta_lines).
    """
    parts = raw.split(SYNC_START)
    if len(parts) < 2:
        raise ValueError(
            "No synchronized-update blocks (\\x1b[?2026h) found in input — "
            "not a --diffusion-visual capture, or ANSI codes were lost in transit."
        )
    block_parts = parts[1:]  # parts[0] is the preamble before the first block

    frames: list[Frame] = []
    last_remainder = ""

    for bp in block_parts:
        if SYNC_END in bp:
            content, remainder = bp.split(SYNC_END, 1)
        else:
            content, remainder = bp, ""
        last_remainder = remainder

        clean = strip_ansi(content)
        lines = clean.split("\n")
        header_line = lines[0] if lines else ""
        body_lines = lines[1:]
        body_lines = _rstrip_blank_lines(body_lines)

        m = HEADER_RE.search(header_line)
        if not m:
            # Not a recognizable step header — skip rather than fabricate one.
            continue
        step, total, percent = int(m.group(1)), int(m.group(2)), int(m.group(3))
        temp = compute_temp(step, total, t_min, t_max)
        frames.append(Frame(step, total, percent, temp, body_lines, entropy_bound, confidence))

    # Trailing plain-text dump after the last synchronized-update block is NOT
    # a diffusion step — extract only its `total time` / `throughput` footer
    # as metadata, and drop the rest (see docstring above).
    clean = strip_ansi(last_remainder)
    lines = clean.split("\n")
    lines = _lstrip_blank_lines(lines)
    meta = [line for line in lines if line.startswith("total time") or line.startswith("throughput")]

    return frames, meta


# ---------------------------------------------------------------------------
# Running the CLI
# ---------------------------------------------------------------------------

def build_command(args: argparse.Namespace) -> list[str]:
    return [
        args.binary,
        "-m", args.gguf,
        "-p", args.prompt,
        "-n", str(args.canvas_len),
        "-ngl", "99",
        "--diffusion-visual",
        "--diffusion-visual-progress",
        "--diffusion-visual-interval", "1",
        "--diffusion-eb-max-steps", str(args.steps),
        "--diffusion-eb-t-min", str(args.t_min),
        "--diffusion-eb-t-max", str(args.t_max),
        "--diffusion-eb-entropy-bound", str(args.entropy_bound),
        "--diffusion-eb-confidence", str(args.confidence),
        "-no-cnv",
    ]


def run_cli(args: argparse.Namespace) -> str:
    """Run the CLI under a PTY sized wide, so its own terminal-width probe
    reports a wide viewport instead of truncating every captured line.

    llama-diffusion-cli's visual renderer (diffusion-cli.cpp:43-61) queries
    `ioctl(STDOUT_FILENO, TIOCGWINSZ)` and clamps every drawn line to that
    width (diffusion-cli.cpp:128, `ln.resize(cols)`); when stdout is a plain
    pipe (subprocess.PIPE, no TTY) the ioctl reports nothing usable and the
    code falls back to a hardcoded 24x80 (diffusion-cli.cpp:44-45), silently
    cutting off everything past column 80 of the ~256-token canvas. A PTY
    with TIOCSWINSZ set wide fixes this at the source — no env var
    (COLUMNS/TERM) is consulted by that code path, only the ioctl.
    """
    cmd = build_command(args)
    print("+ " + " ".join(cmd), file=sys.stderr)

    # Cols sized to comfortably hold one full canvas line even with no
    # explicit newlines in the generated text (worst case ~4 chars/token).
    pty_cols = max(512, args.canvas_len * 4)
    pty_rows = 64  # generous headroom over the handful of lines actually drawn

    master_fd, slave_fd = pty.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", pty_rows, pty_cols, 0, 0))

    # Disable ONLCR/ECHO on the slave: without this the PTY line discipline
    # inserts its own "\r" before every "\n" (on top of the CLI's own
    # in-band "\r" cursor-return bytes) and would echo back anything written
    # to the master — neither of which the non-PTY capture format had.
    attrs = termios.tcgetattr(slave_fd)
    attrs[1] &= ~termios.ONLCR  # oflag
    attrs[3] &= ~termios.ECHO   # lflag
    termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

    proc = subprocess.Popen(cmd, stdin=slave_fd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
    os.close(slave_fd)

    chunks: list[bytes] = []
    while True:
        try:
            data = os.read(master_fd, 65536)
        except OSError as e:
            if e.errno == errno.EIO:
                break  # child closed its end of the pty — normal PTY-master EOF
            raise
        if not data:
            break
        chunks.append(data)
    os.close(master_fd)
    proc.wait()

    raw = b"".join(chunks).decode("utf-8", errors="replace")
    if proc.returncode != 0:
        print(raw, file=sys.stderr)
        raise SystemExit(
            f"BLOCKED: llama-diffusion-cli exited {proc.returncode} — see stderr above."
        )
    return raw


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_text_frames(frames: list[Frame], out_dir: Path) -> None:
    for i, frame in enumerate(frames):
        path = out_dir / f"step_{i:03d}.txt"
        text = "\n".join(frame.body_lines)
        path.write_text(text + "\n", encoding="utf-8")


def render_png_frames(frames: list[Frame], out_dir: Path) -> bool:
    """Render each frame to a PNG with a fixed canvas size. Returns True if
    Pillow was available and PNGs were written, False if it degraded to
    text-only (caller should fall back to an HTML-of-text scrubber)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(
            "Pillow not available — degrading to text-only frames (no PNGs).",
            file=sys.stderr,
        )
        return False

    font_path = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        if Path(candidate).exists():
            font_path = candidate
            break

    font_size = 16
    header_font_size = 15
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    header_font = (
        ImageFont.truetype(font_path, header_font_size) if font_path else ImageFont.load_default()
    )

    # Fixed frame geometry: measured once across ALL frames so every PNG is
    # the same pixel size (required for a stable scrubbing experience).
    max_lines = max((len(f.body_lines) for f in frames), default=1)
    max_cols = max((len(line) for f in frames for line in f.body_lines), default=1)
    max_cols = max(max_cols, 40)

    bbox = font.getbbox("M")
    char_w = bbox[2] - bbox[0] or 9
    line_h = int(font_size * 1.35)

    margin = 16
    header_h = 40
    img_w = margin * 2 + max_cols * char_w
    img_h = margin * 2 + header_h + max_lines * line_h

    bg = (18, 18, 20)
    fg = (210, 210, 210)
    header_bg = (35, 35, 45)
    header_fg = (120, 200, 255)
    temp_fg = (60, 230, 110)  # sampler-schedule green — the anneal-gauge readout

    for i, frame in enumerate(frames):
        img = Image.new("RGB", (img_w, img_h), bg)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img_w, header_h], fill=header_bg)

        # Header strip: "step k/S · pct% · " in the normal header color, then
        # "temp t.ttt" styled prominently in green (the anneal-gauge readout —
        # flip to the last frame and read whether the run quit hot or cold),
        # then any trailing eb/confidence context in the normal color.
        prefix = f"step {frame.step}/{frame.total_steps} · {frame.percent}% · temp "
        temp_str = f"{frame.temp:.3f}"
        suffix = ""
        if frame.entropy_bound is not None and frame.confidence is not None:
            suffix = f"   eb={frame.entropy_bound} conf={frame.confidence}"

        x, y = margin, 10
        draw.text((x, y), prefix, font=header_font, fill=header_fg)
        x += draw.textlength(prefix, font=header_font)
        draw.text((x, y), temp_str, font=header_font, fill=temp_fg)
        x += draw.textlength(temp_str, font=header_font)
        if suffix:
            draw.text((x, y), suffix, font=header_font, fill=header_fg)

        y = header_h + margin // 2
        for line in frame.body_lines:
            draw.text((margin, y), line, font=font, fill=fg)
            y += line_h

        img.save(out_dir / f"step_{i:03d}.png")

    return True


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DiffusionGemma flip-book — {prompt_escaped}</title>
<style>
  body {{ background: #111214; color: #ddd; font-family: sans-serif; text-align: center; padding: 24px; }}
  h1 {{ font-size: 16px; font-weight: normal; color: #7fc8ff; }}
  #frame {{ max-width: 100%; border: 1px solid #333; background: #121214; }}
  #controls {{ margin-top: 16px; display: flex; align-items: center; justify-content: center; gap: 12px; }}
  #slider {{ width: 60%; }}
  #label {{ font-family: monospace; min-width: 22ch; }}
  button {{ font-size: 16px; padding: 4px 12px; cursor: pointer; }}
  #hint {{ margin-top: 8px; color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>DiffusionGemma flip-book &mdash; prompt: "{prompt_escaped}"</h1>
<img id="frame" src="{first_src}" alt="diffusion frame">
<div id="controls">
  <button id="prev">&#8592;</button>
  <input type="range" id="slider" min="0" max="{max_index}" value="0" step="1">
  <button id="next">&#8594;</button>
  <span id="label"></span>
</div>
<div id="hint">Left/Right arrow keys also step through frames.</div>
<script>
const frames = {frames_json};
const img = document.getElementById('frame');
const slider = document.getElementById('slider');
const label = document.getElementById('label');
let idx = 0;

function show(i) {{
  idx = Math.max(0, Math.min(frames.length - 1, i));
  img.src = frames[idx].file;
  slider.value = idx;
  label.textContent = frames[idx].label;
}}

slider.addEventListener('input', () => show(parseInt(slider.value, 10)));
document.getElementById('prev').addEventListener('click', () => show(idx - 1));
document.getElementById('next').addEventListener('click', () => show(idx + 1));
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') show(idx - 1);
  if (e.key === 'ArrowRight') show(idx + 1);
}});

show(0);
</script>
</body>
</html>
"""

HTML_TEXT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DiffusionGemma flip-book (text-only) — {prompt_escaped}</title>
<style>
  body {{ background: #111214; color: #ddd; font-family: sans-serif; text-align: center; padding: 24px; }}
  h1 {{ font-size: 16px; font-weight: normal; color: #7fc8ff; }}
  pre {{ text-align: left; display: inline-block; background: #121214; border: 1px solid #333;
         padding: 16px; min-width: 60ch; min-height: 20em; white-space: pre; font-size: 15px; }}
  #controls {{ margin-top: 16px; display: flex; align-items: center; justify-content: center; gap: 12px; }}
  #slider {{ width: 60%; }}
  #label {{ font-family: monospace; min-width: 22ch; }}
  button {{ font-size: 16px; padding: 4px 12px; cursor: pointer; }}
</style>
</head>
<body>
<h1>DiffusionGemma flip-book &mdash; prompt: "{prompt_escaped}" (Pillow unavailable: text-only mode)</h1>
<pre id="frame"></pre>
<div id="controls">
  <button id="prev">&#8592;</button>
  <input type="range" id="slider" min="0" max="{max_index}" value="0" step="1">
  <button id="next">&#8594;</button>
  <span id="label"></span>
</div>
<script>
const frames = {frames_json};
const pre = document.getElementById('frame');
const slider = document.getElementById('slider');
const label = document.getElementById('label');
let idx = 0;
let cache = {{}};

async function show(i) {{
  idx = Math.max(0, Math.min(frames.length - 1, i));
  slider.value = idx;
  label.textContent = frames[idx].label;
  if (!(idx in cache)) {{
    const resp = await fetch(frames[idx].file);
    cache[idx] = await resp.text();
  }}
  pre.textContent = cache[idx];
}}

slider.addEventListener('input', () => show(parseInt(slider.value, 10)));
document.getElementById('prev').addEventListener('click', () => show(idx - 1));
document.getElementById('next').addEventListener('click', () => show(idx + 1));
document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') show(idx - 1);
  if (e.key === 'ArrowRight') show(idx + 1);
}});

show(0);
</script>
</body>
</html>
"""


def write_html(frames: list[Frame], out_dir: Path, prompt: str, have_png: bool) -> Path:
    manifest = [
        {
            "file": f"step_{i:03d}.png" if have_png else f"step_{i:03d}.txt",
            "label": frame.label,
        }
        for i, frame in enumerate(frames)
    ]
    prompt_escaped = prompt.replace('"', "&quot;")
    template = HTML_TEMPLATE if have_png else HTML_TEXT_TEMPLATE
    html = template.format(
        prompt_escaped=prompt_escaped,
        first_src=manifest[0]["file"],
        max_index=len(frames) - 1,
        frames_json=json.dumps(manifest),
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Turn a DiffusionGemma llama-diffusion-cli run into a navigable flip-book."
    )
    ap.add_argument("--prompt", required=False, help="Prompt to diffuse (required unless --replay)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="--diffusion-eb-max-steps (default 48)")
    ap.add_argument("--canvas-len", "-n", type=int, default=DEFAULT_CANVAS_LEN, help="-n canvas length (default 256)")
    ap.add_argument("--out", default=None, help="Output dir (default ./out/<slugified-prompt>/)")
    ap.add_argument("--gguf", default=DEFAULT_GGUF, help="Path to GGUF weights")
    ap.add_argument("--binary", default=DEFAULT_BINARY, help="Path to llama-diffusion-cli")
    ap.add_argument("--t-min", type=float, default=DEFAULT_T_MIN, dest="t_min")
    ap.add_argument("--t-max", type=float, default=DEFAULT_T_MAX, dest="t_max")
    ap.add_argument("--entropy-bound", type=float, default=DEFAULT_ENTROPY_BOUND)
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument(
        "--replay",
        default=None,
        help="Skip running the CLI; parse an already-captured raw stdout file instead (dev/test mode).",
    )
    args = ap.parse_args()

    if not args.replay and not args.prompt:
        ap.error("--prompt is required unless --replay is given")

    prompt_for_naming = args.prompt or Path(args.replay).stem
    out_dir = Path(args.out) if args.out else Path(__file__).parent / "out" / slugify(prompt_for_naming)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replay:
        # newline='' is required: the capture uses bare \r as an in-band cursor
        # control byte (terminal carriage-return-to-column-0), not a line
        # ending. Python's universal-newline text mode would otherwise
        # silently rewrite every \r to \n and destroy the block structure.
        with open(args.replay, "r", encoding="utf-8", errors="replace", newline="") as fh:
            raw = fh.read()
    else:
        raw = run_cli(args)
        (out_dir / "_raw_capture.txt").write_text(raw, encoding="utf-8")

    frames, meta = parse_frames(
        raw, args.t_min, args.t_max, args.entropy_bound, args.confidence
    )
    if not frames:
        raise SystemExit("BLOCKED: parsed zero frames — check the raw capture format.")

    render_text_frames(frames, out_dir)
    have_png = render_png_frames(frames, out_dir)
    index_path = write_html(frames, out_dir, args.prompt or prompt_for_naming, have_png)

    print(f"\nWrote {len(frames)} frames to {out_dir}")
    if meta:
        print("Run metadata (post-loop dump, not a diffusion step):")
        for line in meta:
            print(f"  {line}")
    print(f"Open: {index_path}")
    regen = (
        f"python3 {Path(__file__).resolve()} --prompt \"<your prompt>\" --steps {args.steps} "
        f"--canvas-len {args.canvas_len}"
    )
    print(f"Regenerate for a new prompt:\n  {regen}")


if __name__ == "__main__":
    main()
