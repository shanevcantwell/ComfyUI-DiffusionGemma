# Contributing

This project is an **instrument for exploring** how DiffusionGemma thinks — how
meaning anneals out of a canvas of noise. Contributions in that spirit are very
welcome, and "contribution" here is broad: a question, a screenshot of a strange
commit-front, a knob setting that did something surprising, a half-formed idea,
or code.

## Come talk first — Discussions is open

The **[Discussions](../../discussions)** tab is the front door. Use it for:

- **Show & tell** — a trace, a heatmap, an animation, an alliteration that
  annealed cleanly (or fell apart interestingly).
- **Ideas & questions** — *"what if the schedule…", "why does confidence…",
  "how would I…"*. Nothing is too half-baked; taking half-baked ideas seriously
  is what this project is *for*.
- **Findings** — you watched the commit-front do something the docs don't
  explain. That's the good stuff. Post it.

The instrument exists to surface questions, so bringing one is using it exactly
right.

## Issues vs Discussions

- **Discussions** — ideas, questions, findings, show & tell. Start here when in
  doubt.
- **Issues** — reproducible bugs and tracked work. If something's broken, an
  issue with the graph, the knob values, and what you saw vs. expected is
  perfect.

## Pull requests

PRs are welcome. A few things keep the pack coherent — the full picture is in
**[ARCHITECTURE.md](ARCHITECTURE.md)**, but in short:

- **Keep the split.** The ComfyUI node files stay thin adapters; the real logic
  lives in the engine layer and runs without ComfyUI present, so it can be
  developed and tested from a bare script.
- **Payloads mean what they say.** A socket carries real, canonical data — no
  repurposing a tensor to smuggle something it isn't.
- **Behavior changes carry a test.** New behavior comes with coverage; the
  engine is testable without a running ComfyUI.
- **Keep the canvas light.** On-graph notes are glances, not documentation —
  depth belongs in the docs.

Don't sweat perfection: open a draft, start a discussion, and we'll shape it
together.

## The one ethos

Sell **instrumentability, not speed.** This will never out-run an autoregressive
model, and that's fine — the value is *watching the machine think*. Everything
here optimizes for legibility of the process over throughput of the product.
