# D3 validation and handoff record

Status: design/workflow package validated; engine implementation and release qualification NOT RUN.

## What was actually executed

Local Linux environment: Python 3.13.5, GCC/G++ 14.2.0, CMake 3.31.6. `cmake --build` succeeded, and `ctest --output-on-failure --no-tests=error -V` passed all three registered groups: C11 header smoke, C++17 header smoke, and 44 Python workflow-policy tests. No tests were skipped. The two compiled executables only instantiate contract types; they do not link a native engine or execute inference. Raw output is banked in `evidence/ctest.log`; toolchain output is in `evidence/toolchain.txt`.

GitHub Actions independently executed the same source-level checks at D2 commit `274a65a2beda2992da4139fc6c82786a249e38f1`; run `39376595417` completed successfully. The existing repository leak-guard also passed at D2. This is a second execution environment, not an independent architectural review. Source blobs of all 18 D2 package files were compared against GitHub's tree and matched; `evidence/d2-source-blobs.json` records that immutable source set.

https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/actions/runs/39376595417

The branch-specific snapshot workflow tests the exact pushed files before creating D2/D3 tags, and refuses to move an existing tag. D0/D1 tags mark documentation snapshots only. The action never publishes a binary or an upstream PR.

## What has not been executed

No native DiffusionGemma library, worker or CDG backend has been implemented here. No llama.cpp build, HF/GGUF numerical parity run, CUDA test, current CDG fast/live/e2e suite, real ComfyUI workflow, MCP parity run, fresh-install test or engine benchmark was run. No engine test-coverage percentage is claimed. I0-I5 are implementation work packets with detailed mandatory tests, not completed gates.

The local Git worktree is metadata-backed at the exact CDG baseline and materializes this new package only. Direct GitHub DNS/network access and an NVIDIA GPU were unavailable. The original CDG checkout was not hydrated, so this is not evidence that its existing tests pass. Git data writes through the connected account preserve the base tree and change only this package and its phase-tag workflow. The code harness should fetch the actual remote branch normally rather than treat the partial local checkout as a complete repository.

## Self-review findings and disposition

| Finding | Disposition |
|---|---|
| C-style zero initialization triggered C++ missing-field warnings under `-Werror` | Fixed with language-specific initialization; C11 and C++17 rerun successfully. |
| Early checkpoint required complete later-node/security tests | Fixed in the phase manifest; BASIC-01, CORE-SECURITY-01 and WORKER-01 separate early scopes. All full node tests first gate I4, then remain cumulative. |
| A JSON list at the receipt CLI caused an attribute error | Rejected explicitly with a structured BLOCKED diagnostic; a regression test was added. |
| Source maps disagreed only in local JSON formatting and one paragraph order | Local files reconciled to the exact remote D2 Git blob hashes before the handoff package. |
| Generated receipts could superficially look like real evidence | Test fixtures explicitly identify themselves as synthetic policy tests; none is banked as an engine-phase PASS. |
| Independent reviewer identity cannot be authenticated by a string field | Checker limitations stated; real role dispatch and review provenance remain harness responsibilities. |

The design has author self-review only. No separate model-semantics or systems/security review session was available here. I0 begins with those independent review assignments and remaining source verification, rather than inheriting a fictitious approval. Source conflicts, numerical thresholds, supported node modes and ABI details remain explicitly gated technical work, not questions the operator is expected to answer.

## Release and scope status

The eight registered node classes are frozen in requirements.json; every supported mode still needs source extraction and real execution at I0/I4. `initial_platforms` is the desired build matrix, not four certified runtime combinations. Certify only measured host/device pairs; target the operator's RTX-8000 and RTX-3090 environments first. Image conditioning remains a separate disabled workflow. Existing runtime files, existing ADRs, main, and the project's license are not changed by this package. Public release remains blocked until I5 and is never automatically authorized by the receipt checker.
