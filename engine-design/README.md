# DiffusionGemma native engine: design and implementation workflows

Status: design package, not an implemented or released inference engine.
Date: 2026-10-19. Branch: `design/dg-engine-20261019`.

## Product contract

Build a text-only, instrumentable native GGUF engine using llama.cpp/ggml, with a stable CDG adapter. A generation CLI is a consumer, not the engine API. Public distribution waits until every currently registered CDG node and its supported modes can execute on the native path. Earlier basic-generation and limited-node phases are experimental HITL checkpoints, not releases. Preserve the Transformers reference path.

Image conditioning has its own plan and implementation workflow, disabled in this work. No image inference code, upstream submission, release publication, existing ADR rewrite, or changes to the existing runtime are authorized by this design package.

## Scope and authority

The baseline is CDG `fed377afc7b54f03cb7faa4dd798c80c15279d8a`. A code harness must compare this baseline with the actual checkout before implementation; later changes require a recorded requirement delta, not silent adoption. Do not disturb other worktrees, uncommitted files, research branches, or rescue lineage. This branch alone receives the design work.

Design snapshots D0-D3 are not implementation gates I0-I5. Tags identify immutable design snapshots only. Technical promotion requires executable evidence and separate review sessions, not operator approval of unfamiliar C++. HITL checkpoints are for hands-on usefulness and research behavior; the operator retains a veto but is not the technical gatekeeper. No claim of independent review is made for this author's own self-review.

## Deliverable map

- `requirements.json`: frozen nodes, release constraints, runtime capabilities.
- `sources.md`: inspected source revisions, documentation, unresolved source conflicts.
- `architecture.md`: ownership, graphs, stepping, transport, CDG integration.
- `decisions.md`: design choices and upstream patch boundaries.
- `verification.md`: requirement-to-test matrix, oracles, failure and release criteria.
- `workflows/text.json`: text implementation phases and review contracts.
- `workflows/image.json`: independent image plan, disabled and unimplemented.
- `tools/workflow.py`: harness-neutral planning and evidence-gate utility.
- `contract/dgemma-engine.h`: proposed C contract; declarations, not a working library.
- `tests/`: executable checks of the workflow and contract, not engine inference.
- `validation.md`: tests actually run, unresolved gates, handoff.

## Environment boundary

This chat environment has Git, CMake, and a C/C++ compiler, but no accessible NVIDIA GPU and no direct GitHub DNS/network route. GitHub API operations use the connected account. A real metadata-backed worktree is based on the exact CDG commit; only this new package is materialized locally. It is not a full source checkout. Model-free workflow and header tests can run here. The existing CDG suite, llama.cpp builds, numerical parity, GPU performance, and real ComfyUI/MCP tests must run in the code harness. Missing dependencies or skipped tests cannot become passing release evidence.

The small GitHub workflow on this branch creates design-snapshot tags only. It does not publish binaries or run code on the operator's GPUs.
