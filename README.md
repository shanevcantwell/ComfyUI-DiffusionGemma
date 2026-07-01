# ComfyUI-DiffusionGemma

A ComfyUI node pack for **DiffusionGemma** — text generation by *uniform-state
discrete diffusion*, exposed as a ComfyUI graph you can watch, instrument, and
take apart.

> ### ⚠️ Status: aspirational (design-only, no working nodes yet)
>
> As of 2026-06-30 this repo is **framing, not function**. What exists is the
> decision record and the build plan — no node in here runs in ComfyUI yet.
> The `__init__.py` registers **zero** nodes on purpose; installing this pack
> today adds nothing to your node menu. The first working nodes land in
> **[Phase 1](plan.md)** (`DGemmaLoader` + `DGemmaSampler`, prompt-in →
> text-out). Until then, read this as a spec you could build from — or watch it
> get built.

## What it is (the idea)

DiffusionGemma doesn't autoregress. It starts from a fixed **256-token canvas of
random vocabulary tokens** and iteratively refines it: an entropy-bound sampler
commits the lowest-entropy positions under a budget and re-noises the rest, step
after step, until it stabilizes. There is **no sigma schedule and no latent
space** — the "schedule" is a per-step *temperature + entropy-budget* trajectory,
and the working state is a discrete token canvas.

ComfyUI's whole sampling ecosystem (`KSampler`, `BasicScheduler`, RES4LYF, the
solver zoo) is built on `SIGMAS` and `LATENT` — continuous Gaussian diffusion.
DiffusionGemma's loop has no input of that shape. So this pack keeps RES4LYF's
**node topology** (schedule → constraints → sampler → options chain) but swaps
every socket **payload** for entropy-native types. See
**[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)**.

The payoff is **instrumentation**: per-step entropy heatmaps and the
commit-per-step avalanche curve (`DGemmaTrace`) — watching a discrete diffusion
denoise happen on your own runs. This is the on-ramp that didn't exist.

## Where the design lives

| Doc | What it holds |
|-----|---------------|
| **[plan.md](plan.md)** | The 6-phase build roadmap. *What to do next.* |
| **[decisions/](decisions/)** | ADRs — *why* the load-bearing choices were made. |
| **[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)** | Native socket types instead of reusing `SIGMAS`/`LATENT`. |
| **[ADR-CDG-002](decisions/adr-cdg-002-transformers-streamer-access-path.md)** | transformers + `TextDiffusionStreamer` as the access path. |
| **[loose-ends.md](loose-ends.md)** | Tactical decisions below the ADR bar. |

## Relationship to RES4LYF

This pack **steals RES4LYF's shape and rejects its substrate.** RES4LYF honestly
reuses `SIGMAS`/`LATENT` because it *is* genuinely sigma/latent-based. This pack
is not — so reusing those types would be a literal instance of the "lying sigmas"
trap RES4LYF jokingly named, but unintentional and load-bearing. The node graph
here reflects the real substrate, which is what makes it teachable rather than a
disguise. (See ADR-CDG-001.)

## Install (once nodes exist)

Not yet. When Phase 1 lands: clone into `ComfyUI/custom_nodes/` and restart
ComfyUI. Requires `transformers` with DiffusionGemma support; runs the reference
HF path (see ADR-CDG-002 — deliberately the slower route, chosen for per-step
access).
