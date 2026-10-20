# Phase-specific test scope

This clarification resolves a self-review finding before the D2 baseline. The full NODE-* and SECURITY-01 groups in verification.md are not partial passes at earlier phases. They first gate the phase where their complete behavior exists, then remain cumulative requirements.

| ID | First phase | Required scope |
|---|---|---|
| CORE-SECURITY-01 | I1 | CPU/native-core bounds and allocation checks, malformed tiny GGUF fixtures, invalid shapes/types, session ownership, ASan/UBSan. Worker/protocol fuzzing is not falsely claimed here. |
| BASIC-01 | I2 | Real-weight prompt-to-text, seeded ordinary generation, Tier-0 entropy and commit trace, clean versus raw IDs, frame-to-image rendering and partial result structure through the minimal client. No claim of full current-node, controls or cache-node parity. |
| WORKER-01 | I3 | Local protocol round-trip, malformed/truncated frames, byte limits, worker death, separate cancellation, bounded backpressure and load/unload lifetime. Run before treating the persistent worker as a usable integration boundary. |

I3 runs the complete TRACE-* groups, including constrained and temperature-shaped stages. I4 runs all eight NODE-* groups, not just the five added after the basic checkpoint. I5 runs the complete SECURITY-01 group against the finished product and repeats every cumulative group.

I1's SCHED-* tests are isolated sampler/control mathematics; I3/I4 exercise the actual adapter and graph wiring. An isolated passing unit test never stands in for its later real-node test. The I2 client can use the C facade directly while the persistent worker arrives at I3; the facade is the same engine boundary, not a second inference algorithm.
