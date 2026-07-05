// web/live_view.js — DiffusionGemma live per-step view (plan.md Phase 3 (a)).
//
// `DGemmaSampler`'s own node body pushes one custom event per denoising step
// via `PromptServer.instance.send_sync` (nodes/sampler.py, event name
// `DGEMMA_STEP_EVENT = "dgemma.sampler.step"`) — ComfyUI gives a node's
// outputs to downstream sockets only once its FUNCTION returns, so there is
// no mechanism for a node to stream per-step state through a socket while
// its own loop is still running. The live view is therefore a feature of the
// sampling node itself, not a downstream consumer node (plan.md Phase 3, the
// (a)/(b) split rationale; (b) is `DGemmaTrace`, unaffected by this file).
//
// Registration-ordering note (load-bearing, not a style choice):
// `api.addEventListener` MUST run inside `setup()` (fires once at app load),
// not lazily on first use. Grounded against the real (non-shim) `ComfyApi`
// bundle class (`comfyui_frontend_package==1.45.20`,
// `static/assets/api-*.js`): the websocket message handler's fallback case
// for an event name that was never registered via `addEventListener` does
// `if (this._registered.has(t.type)) dispatchEvent(...); else if (!reported)
// throw Error(...)` — caught and logged, not fatal, but the message itself
// is silently dropped. Registering any later than `setup()` risks losing the
// first several per-step pushes of every run to that drop path. See
// CLAUDE.md's grounded facts / this repo's loose-ends.md for the full
// grounding session.
//
// Rendering approach (implementer's call, per plan.md step 7 — not further
// specified there): a litegraph `onDrawForeground` overlay + `setDirtyCanvas`
// rather than a DOM overlay. This is the idiom `loose-ends.md`'s own
// graduation-trigger note anticipates ("addEventListener + setDirtyCanvas"),
// and it needs no extra widget wiring on the node — the state to render is
// small (a handful of numeric fields per step), so a text line drawn
// directly on the node body is the cheapest correct rendering, matching the
// same "cheapest correct" discipline `nodes/trace.py`'s STRING summary uses.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DGEMMA_STEP_EVENT = "dgemma.sampler.step";
const DGEMMA_SAMPLER_NODE_TYPE = "DGemmaSampler";

app.registerExtension({
    name: "DiffusionGemma.LiveView",

    // Attach the per-step overlay renderer to every `DGemmaSampler` node
    // instance's prototype. This runs once per node *type* registration,
    // not per node instance — cheap, and independent of how many
    // DGemmaSampler nodes exist on the graph.
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== DGEMMA_SAMPLER_NODE_TYPE) {
            return;
        }

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const result = onDrawForeground?.apply(this, arguments);

            const live = this._dgemmaLiveState;
            if (live && !(this.flags && this.flags.collapsed)) {
                const label =
                    `canvas ${live.canvas_idx} · step ${live.step_idx} · ` +
                    `t=${Number(live.t).toFixed(3)} · temp=${Number(live.temperature).toFixed(3)} · ` +
                    `committed=${(Number(live.committed_fraction) * 100).toFixed(1)}%`;

                ctx.save();
                ctx.font = "12px monospace";
                ctx.fillStyle = "#0f0";
                ctx.textAlign = "left";
                ctx.fillText(label, 8, this.size[1] - 8);
                ctx.restore();
            }

            return result;
        };
    },

    async setup() {
        // Fires once at app load — every graph queued afterward is covered.
        api.addEventListener(DGEMMA_STEP_EVENT, (event) => {
            const payload = event.detail;
            if (!payload || payload.node === undefined || payload.node === null) {
                return;
            }

            const node = app.graph.getNodeById(String(payload.node));
            if (!node) {
                return; // Not on this client's graph (e.g. a different tab/session watching the queue).
            }

            node._dgemmaLiveState = payload;
            node.setDirtyCanvas(true, false);
        });
    },
});
