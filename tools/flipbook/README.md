# flipbook — navigable per-step diffusion viewer

A standalone dev tool. It is **not** part of the ComfyUI-DiffusionGemma node
pack — it is not imported by `__init__.py`, `nodes/`, or anything importable,
so it can't affect the node pack's behavior or dependency surface.

## What it is

`llama-diffusion-cli --diffusion-visual` prints every diffusion step as a
full-canvas ANSI redraw that flies by in the terminal. This tool runs the CLI,
captures that stream, parses it into one frame per real diffusion step
(noise → coherent text), and writes:

- `step_NNN.txt` — plain-text canvas per step
- `step_NNN.png` — the same frame rendered in a monospace font, fixed image
  size across all frames (so scrubbing doesn't jitter)
- `index.html` — a self-contained scrubber: a slider plus left/right
  arrow-key navigation, opens directly in a browser (no server needed)

These are the same per-step frames the future (Advanced) sampler node will
emit live inside ComfyUI once it exists. Until then, this tool is the bridge
to the working `llama-diffusion-cli` (GGUF) backend — see issue #15.

## Run it

```
python3 tools/flipbook/flipbook.py --prompt "The capital of France is" --steps 12
```

Defaults (override as needed): `--steps 48`, `--canvas-len/-n 256`,
`--gguf` → the local Q8_0 GGUF, `--t-min 0.4 --t-max 0.8
--entropy-bound 0.1 --confidence 0.005`. Output goes to
`tools/flipbook/out/<slugified-prompt>/` unless `--out` is given.

The entropy-bound scheduler can stop early via its confidence gate — you may
get fewer frames than `--steps`; that's correct behavior, not a bug.

### Open the scrubber

Open `tools/flipbook/out/<slugified-prompt>/index.html` in any browser
(local file, no server required). Drag the slider or use the Left/Right
arrow keys to step through frames one diffusion step at a time.

### Replay mode (no GPU)

To re-parse an already-captured raw stdout file (e.g. while iterating on the
parser) without touching the GPU:

```
python3 tools/flipbook/flipbook.py --replay /path/to/raw_capture.txt --out ./out/replay
```

## Verified run (2026-07-05)

`--prompt "The capital of France is" --steps 12` produced **8 frames**
(7 diffusion-step blocks + the CLI's final plain-text dump as frame 7, since
the entropy-bound confidence gate stopped the run early at step 6/12).

First frame (step 0/12, 0%) — pure noise:

```
 thought
The user is asking a "The capital of France is is".
    *   ::: the the....

*         : France France
    *                     .
```

Last frame (step 7/12, final, 100%) — converged:

```
thought
The user is asking for the capital of France.
    *   Country: France.
    *   Attribute: Paris.
The capital of France is Paris.The capital of France is Paris.
```

## Notes on the parser

- Blocks are delimited by the terminal "synchronized update" private-mode
  sequences `\x1b[?2026h` (start) / `\x1b[?2026l` (end). Text before the first
  start marker is a load/log preamble (discarded). Text after the *last* end
  marker is the CLI's final plain-text canvas + `total time:`/`throughput:`
  footer — that becomes the final frame.
- All ANSI CSI sequences (`\x1b[K`, `\x1b[<N>A`, `\x1b[?25l/h`, etc.) are
  stripped with one regex (`\x1b\[[0-?]*[ -/]*[@-~]`). Bare `\r` bytes are
  stripped separately.
- **Gotcha we hit and fixed:** reading the captured stream through a normal
  text-mode file open with default `newline` handling silently rewrites the
  in-band `\r` cursor-control bytes to `\n`, destroying the block structure.
  `--replay` opens the file with `newline=''` to avoid this; the live
  `subprocess` path is unaffected because `bytes.decode()` does no such
  translation.
- PNG rendering needs Pillow (present in the system `python3` and the
  ComfyUI venv on this box). If Pillow is unavailable, the tool degrades to
  a text-only `index.html` that fetches the `.txt` frames instead of PNGs,
  rather than failing outright.
