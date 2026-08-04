# Issue #131 rung-1 GGUF probe — 2026-08-04

Rung-1 measurement session per issue #131's "Rung-1 measurement plan" comment
(2026-08-03, final comment on the issue) and the shared runsheet on issue #62
(2026-08-04 protocol comment, §3 runsheet order 3 — this window shared the GPU
slot with the ADR-CDG-012 Q-2 smoke, whose artifacts are the sibling directory
`../2026-08-04-adr-cdg-012-q2-smoke/`).

**Host:** Quadro RTX 8000 (sm_75), 48398 MiB VRAM. GPU verified free at session
start (`nvidia-smi`: 1631 MiB used — the expected resident embedding server,
PID 63519, `llama-server`, unrelated to this probe) and at session end (same
1631 MiB — this probe's process fully released VRAM on exit every run).

## Binary and weights located (no rebuild)

- **Binary:** `/tmp/llamacpp-probe/llama.cpp` at commit `c3fb972` (`git log -1`:
  `c3fb97241295c196e09b783e705e84b96cd1bd74`, 2026-07-04, "Add tool calling") —
  exactly the pin named in #131's rung-0 build readback (PR #24423 tip SHA).
  Both `build/bin/llama-diffusion-cli` and `build-cuda/bin/llama-diffusion-cli`
  present, both 141552 bytes, matching the rung-0 readback's reported binary
  size exactly. `--list-devices` confirms `CUDA0: Quadro RTX 8000 (48398 MiB,
  46603 MiB free)` at first invocation. **No rebuild was performed** — the
  binary was already present and functional.
- **GGUF weights found:** `/mnt/storage/LLMs/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q8_0.gguf`
  (26,878,831,328 bytes, ~25 GiB). **This is Q8_0, not Q4_K_M** — CLAUDE.md's
  "Local run defaults" section names Q4_K_M as the intended local-run quant,
  but only Q8_0 is present on this host (`find` across `/mnt/storage/LLMs`
  turned up no Q4_K_M or other quant of this checkpoint). Q4_K_M is confirmed
  downloadable from the Unsloth repo (HTTP 302 resolve, ~15.65 GiB via HEAD
  request) but was **not downloaded** in this window — that would be a
  scope decision (fetching new weights) beyond "locate what's on this host,"
  and 775 GB free on `/mnt/storage` was not treated as license to expand
  scope without asking. **Every throughput number below is Q8_0, not Q4_K_M**
  — flagged explicitly wherever it appears; do not read it as the audience-
  table Q4_K_M datapoint #131 asked for.
- **#24427 (NVIDIA-orbit PR) is NOT built anywhere on this host.** Rung-0's
  readback only named the #24423 pin (`c3fb972`); no checkout, branch, or
  binary for #24427 exists (`git branch -a` on the only llama.cpp checkout
  with the pinned commit shows only `master` and `pr-24423`). Measurements
  that name #24427 specifically (the visibility-tax entropy-readback probe,
  and the #24427 leg of the cross-engine conformance/determinism comparisons)
  are **not executed** in this window — see "Not done" below.

## Measurements run (numbered per #131's rung-1 plan)

All five commands + full stdout are in this directory's `.log` files;
`measurements.log` collects the grep'd numeric lines in one place.

**(1) Visibility tax (#24427 entropy-readback on/off step time) — NOT DONE.**
#24427 is not built on this host (see above). No number to report; not
fabricated.

**(2) #24423 visual-server frame capture vs Tier-0 contract — DONE.**
`m2_visual_capture_seed7.log` (`--diffusion-visual --diffusion-visual-progress
--diffusion-visual-interval 1`, seed 7, same prompt/knobs as all runs below).
Captured via `script -qc` to preserve the terminal redraw stream. Mechanical
finding: every redrawn frame shows the **argmax canvas token stream and the
step progress bar only** — no per-position entropy, no top-k, no full
distribution appear anywhere in the captured stream. This directly confirms
the conformance-matrix claim banked on #131 ("argmax canvas only... delta =
widen the stream schema, example-layer"): the Tier-0 contract's entropy/top-k
fields have zero exposure surface in `--diffusion-visual` mode as built.

**(3) Conformance run (CDG / #24423 / #24427, same prompt/seed/knobs) —
PARTIAL.** Only the #24423 leg is run here (see "Not done" for why CDG and
#24427 legs are absent). #24423 data, prompt `"The lighthouse keeper counted
every wave before dawn."`, `-n 256`, entropy-bound knobs from CLAUDE.md
(`max_steps=48 t=[0.4,0.8] entropy_bound=0.1 confidence=0.005`), Q8_0:

| seed | steps-to-convergence | wall time | tok/s (canvas) | in-step parallel tok/s |
|---|---|---|---|---|
| 7 (run A) | 17/48 | 3638.04 ms | 70.4 | 1196 |
| 7 (run B, repeat) | 17/48 | 3431.75 ms | 74.6 | 1268 |
| 13 | 23/48 | 4609.91 ms | 55.5 | 1277 |
| 21 | 19/48 | 3819.41 ms | 67.0 | 1273 |

Seed 7's two independent invocations converged at the identical step count
(17) with byte-identical final text ("...Elias didn't count them for the")
— see measurement (4) below for the formal determinism check this also
corroborates.

**(4) Determinism probes — DONE for #24423 vs itself; NOT DONE for #24423
vs CDG (CDG leg absent).**

- **Same explicit seed, run twice** (`--seed 7`, `m5_q8_0_seed7.log` vs
  `m4_determinism_seed7_run2.log`): `diff` of the two logs (stripped of
  timestamp/total-time/throughput lines, which vary run-to-run by wall
  clock) → **0 lines of difference.** Byte-identical step-by-step output.
- **No `--seed` flag passed, run twice** (`m4b_noseed_run1.log` vs
  `m4b_noseed_run2.log`): `diff` → **0 lines of difference.** This
  reproduces, on real hardware, the conformance-matrix finding already
  banked on #131 ("inline-review bug: no seed → identical outputs") — the
  CLI does not appear to draw a fresh nondeterministic seed when none is
  given; two unseeded runs of the same prompt/knobs are byte-identical to
  each other, which is the opposite of CDG's own seeding contract ("no-seed
  = honest nondeterminism," CLAUDE.md grounded facts). This is now a
  **CONFIRMED** finding, not just a code-read hypothesis.

**(5) Q4_K_M tok/s on sm_75 — SUBSTITUTED WITH Q8_0, NOT Q4_K_M.** No
Q4_K_M weights are on this host (see above). The table under measurement (3)
above is the closest available number: **Q8_0, canvas throughput 55.5–74.6
tok/s across 3 seeds (mean ~65.8 tok/s), in-step-parallel 1196–1277 tok/s**,
entropy-bound convergence at 17–23/48 steps depending on seed. Q8_0 is
~2x the byte size of Q4_K_M and generally the slower of the two on
memory-bandwidth-bound decode, so **this number should not be read as a
proxy for the Q4_K_M audience-table datapoint #131 asked for** — it is
informative only as "Q8_0 on Turing works and is fast enough to be usable,"
not as the specific quant point named.

## Not done (named explicitly, not glossed over)

- **#24427 build/checkout does not exist on this host.** Rung-0 only pinned
  and built #24423. Building #24427 fresh was out of scope for this window
  (rung-1 is a measurement pass against what rung-0 built, per the task's own
  framing: "do not rebuild unless the binary is genuinely absent AND the
  rung-0 readback names the build recipe" — rung-0 never named a #24427
  recipe, so there is nothing to reproduce here).
- **CDG (transformers/diffusers) leg of the conformance run was not
  executed.** The sibling Q-2 smoke lane (same GPU window, artifacts in
  `../2026-08-04-adr-cdg-012-q2-smoke/`) hit a **CUDA OOM inside
  `encode_sequence`** under bf16 CPU-spill (45.33/47.26 GiB in use before the
  OOM'ing allocation), confirming this host's bf16 CDG path runs at the edge
  of its VRAM margin. Attempting a fresh bf16 CDG generation call here would
  risk the same failure mode and is a materially larger, separate piece of
  work (full transformers/diffusers load, ~48 GB-class) than the
  GGUF-engine-focused scope this issue's rung-1 plan targets — CDG's role in
  measurement (3) is as the reference point for the self-conditioning-fidelity
  H0, not something this window's GPU budget was sized for. Flagging rather
  than attempting and risking an ungrounded partial result.
- **Q4_K_M weights not downloaded.** Confirmed available upstream
  (`unsloth/diffusiongemma-26B-A4B-it-GGUF`, ~15.65 GiB) but fetching new
  weights is a scope expansion beyond "locate what's on this host" and was
  not taken unilaterally.

## Artifact inventory (this directory)

- `README.md` — this file.
- `measurements.log` — grep'd numeric readouts, chronological, from every run below.
- `smoke_test_seed7.log` — first sanity run (Q8_0, seed 7, full stdout incl. `--list-devices` confirmation).
- `m5_q8_0_seed7.log`, `m5_q8_0_seed13.log`, `m5_q8_0_seed21.log` — measurement (3)/(5) throughput table, one log per seed.
- `m4_determinism_seed7_run2.log` — measurement (4) same-seed repeat run.
- `m4_diff_run1_vs_run2.txt` — diff output (empty — byte-identical).
- `m4b_noseed_run1.log`, `m4b_noseed_run2.log` — measurement (4) no-seed repeat runs.
- `m4b_diff_noseed.txt` — diff output (empty — byte-identical, confirms the no-seed-determinism bug).
- `m2_visual_capture_seed7.log` — measurement (2) `--diffusion-visual` frame-capture stream (raw terminal capture via `script`).
- `nvidia_smi_final.txt` — GPU memory state at session end (1631 MiB used — matches session-start baseline, confirming clean VRAM release).

Not committed or pushed per instructions — raw artifacts only, left for the
operator/next session to disposition.
