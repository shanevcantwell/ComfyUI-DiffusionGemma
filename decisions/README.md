# Decision Records

Architecture and engineering decisions for the DiffusionGemma ComfyUI node pack
(working repo name: `ComfyUI-DiffusionGemma` — rename freely).

These records capture **what** was decided and **why**, so the reasoning lives
next to the code instead of in memory. Tactical decisions that don't meet the
ADR bar live in [`../loose-ends.md`](../loose-ends.md). The build roadmap lives
in [`../plan.md`](../plan.md).

| Handle      | Title                                                    | Status   | Date       |
|-------------|----------------------------------------------------------|----------|------------|
| ADR-CDG-001 | [Native socket types instead of reusing SIGMAS/LATENT](adr-cdg-001-native-socket-types.md) | accepted | 2026-06-30 |
| ADR-CDG-002 | [transformers + TextDiffusionStreamer as access path](adr-cdg-002-transformers-streamer-access-path.md) | accepted | 2026-06-30 |
