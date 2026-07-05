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
//
// Placement (operator finding, 2026-07-05, live-GUI session): the first cut
// drew at a fixed offset from the node's bottom edge, which landed under the
// node box and collided with the `thinking` widget. Fixed: the y-start is
// computed from the ACTUAL widget stack — litegraph records `widget.last_y`
// on each widget as it draws (the standard extension idiom for "where do the
// widgets end"), so the overlay starts below the deepest widget. The event
// handler also grows the node once per live run when there is no reserved
// space below the widgets (growth target = `computeSize()` height — the
// minimal size that fits inputs + widgets — plus the fixed live area; that
// sum is also the cap, and an already-larger node is left alone). The label
// word-wraps to the node's width and clips to the node body, so the readout
// sits inside the box, below the last widget, at any node width.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DGEMMA_STEP_EVENT = "dgemma.sampler.step";
const DGEMMA_SAMPLER_NODE_TYPE = "DGemmaSampler";

const LIVE_LINE_HEIGHT = 14; // px per wrapped text line (12px monospace + leading)
const LIVE_MAX_LINES = 4; // sane cap — wrapped lines beyond this are dropped
const LIVE_AREA_HEIGHT = LIVE_MAX_LINES * LIVE_LINE_HEIGHT + 12; // reserved space below widgets
const LIVE_PAD_X = 8;
const LIVE_PAD_Y = 6;

// Bottom of the widget stack, from litegraph's own per-widget draw
// bookkeeping (`widget.last_y`, set on each widget as it is drawn). Falls
// back to the node's minimal computed height minus the reserved live area
// when widgets haven't drawn yet (a frame arriving before the first paint).
function widgetStackBottom(node) {
    const widgetHeight = (window.LiteGraph && window.LiteGraph.NODE_WIDGET_HEIGHT) || 20;
    let bottom = 0;
    for (const w of node.widgets || []) {
        if (typeof w.last_y === "number") {
            bottom = Math.max(bottom, w.last_y + (w.computedHeight || widgetHeight));
        }
    }
    if (!bottom) {
        bottom = Math.max(0, node.computeSize()[1] - LIVE_AREA_HEIGHT);
    }
    return bottom;
}

// Greedy word-wrap against the node's inner width, capped at LIVE_MAX_LINES.
function wrapLabel(ctx, text, maxWidth) {
    const words = text.split(" ");
    const lines = [];
    let line = "";
    for (const word of words) {
        const candidate = line ? line + " " + word : word;
        if (line && ctx.measureText(candidate).width > maxWidth) {
            lines.push(line);
            line = word;
            if (lines.length >= LIVE_MAX_LINES) {
                line = "";
                break;
            }
        } else {
            line = candidate;
        }
    }
    if (line && lines.length < LIVE_MAX_LINES) {
        lines.push(line);
    }
    return lines;
}

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

                const yStart = widgetStackBottom(this) + LIVE_PAD_Y;
                const maxWidth = this.size[0] - 2 * LIVE_PAD_X;

                ctx.save();
                // Clip to the node body so wrapped text can never paint
                // outside the box or over neighboring canvas content.
                ctx.beginPath();
                ctx.rect(0, 0, this.size[0], this.size[1]);
                ctx.clip();

                ctx.font = "12px monospace";
                ctx.fillStyle = "#0f0"; // keep the green — it's the spot-it-live cue
                ctx.textAlign = "left";
                const lines = wrapLabel(ctx, label, maxWidth);
                for (let i = 0; i < lines.length; i++) {
                    ctx.fillText(lines[i], LIVE_PAD_X, yStart + (i + 1) * LIVE_LINE_HEIGHT);
                }
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

            // Reserve space below the widget stack while a run is live:
            // computeSize() is the minimal height fitting inputs + widgets,
            // so minimal + LIVE_AREA_HEIGHT is both the growth target and
            // its own cap — an already-larger node is left alone.
            const needed = node.computeSize()[1] + LIVE_AREA_HEIGHT;
            if (node.size[1] < needed) {
                node.setSize([node.size[0], needed]);
            }

            node.setDirtyCanvas(true, false);
        });
    },
});
