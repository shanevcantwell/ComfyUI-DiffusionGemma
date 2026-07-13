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
// Rendering approach — ROUND 2 (operator regression finding, 2026-07-05):
// the readout is a LAYOUT-PARTICIPATING CUSTOM WIDGET in `node.widgets`, not
// `onDrawForeground` paint. Two earlier attempts failed the same way:
// foreground paint at a fixed offset collided with the widget stack, and
// growing the node didn't help because ComfyUI's widget layout hands ALL
// surplus node height to resizable widgets — the multiline `prompt`
// textarea absorbed the added space and the paint stayed buried.
//
// Grounded against the installed frontend bundle
// (`comfyui_frontend_package==1.45.20`, `static/assets/api-DzWNw5Ki.js`),
// the same way the addEventListener idiom was grounded:
// - Layout: the widget-arrange pass does
//   `if (e.computeSize) { t = e.computeSize()[1] + 4; e.computedHeight = t }
//    else if (e.computeLayoutSize) { a.push({minHeight, prefHeight, w: e}) }`
//   and then distributes ONLY the remaining space among the
//   `computeLayoutSize` (resizable, e.g. textarea) widgets via
//   `distributeSpace(Math.max(0, r), s)`. A widget exposing `computeSize`
//   therefore has fixed, reserved height the textarea structurally cannot
//   swallow — the exact fix for the regression.
// - Draw: `drawWidgets` does `typeof s.draw === "function" ?
//   s.draw(e, this, l, i, a, t)` — i.e. custom widgets are drawn via
//   `draw(ctx, node, widgetWidth, y, H, lowQuality)` with `y` assigned by
//   the layout pass (`for (let e of o) e.y = l, l += e.computedHeight`).
// - Inclusion: `getLayoutWidgets()` filters only `hidden` widgets;
//   `isWidgetVisible` only excludes `hidden`/`advanced` — a plain object
//   widget participates.
// - Registration: `node.addCustomWidget(w)` exists
//   (`addCustomWidget(e){this.widgets||=[]; ...; this.widgets.push(t)}`).
// - Serialization safety: save does `if (r.serialize === false) continue`
//   and restore does `if (n.serialize !== false)` — `serialize: false`
//   keeps this widget out of `widgets_values` in BOTH directions, so it
//   can never shift the real widgets' saved values by index.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DGEMMA_STEP_EVENT = "dgemma.sampler.step";
const DGEMMA_SAMPLER_NODE_TYPE = "DGemmaSampler";
const LIVE_WIDGET_NAME = "dgemma_live_view";

const LIVE_LINE_HEIGHT = 14; // px per wrapped text line (12px monospace + leading)
const LIVE_MAX_LINES = 4; // reserved lines; wrapped overflow beyond this is dropped
const LIVE_AREA_HEIGHT = LIVE_MAX_LINES * LIVE_LINE_HEIGHT + 4; // = 60px reserved by layout
const LIVE_PAD_X = 8;

function formatLabel(live) {
    return (
        `canvas ${live.canvas_idx} · step ${live.step_idx} · ` +
        `t=${Number(live.t).toFixed(3)} · temp=${Number(live.temperature).toFixed(3)} · ` +
        `committed=${(Number(live.committed_fraction) * 100).toFixed(1)}%`
    );
}

// Greedy word-wrap against the widget's inner width, capped at LIVE_MAX_LINES.
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

// A read-only, layout-participating widget owning the live readout's lines.
// `computeSize` (fixed height) is what makes the reservation real — see the
// bundle grounding in the header comment. `serialize: false` keeps it out of
// `widgets_values` on both save and restore.
function createLiveWidget() {
    return {
        type: LIVE_WIDGET_NAME,
        name: LIVE_WIDGET_NAME,
        value: null, // latest per-step payload; null = idle
        serialize: false,
        options: {},
        computeSize(width) {
            return [width ?? 0, LIVE_AREA_HEIGHT];
        },
        draw(ctx, node, widgetWidth, y) {
            const label = this.value ? formatLabel(this.value) : "live: (idle)";
            const maxWidth = widgetWidth - 2 * LIVE_PAD_X;

            ctx.save();
            // Clip to this widget's own reserved rect — wrapped text can
            // never paint over neighboring widgets or outside the node.
            ctx.beginPath();
            ctx.rect(0, y, widgetWidth, LIVE_AREA_HEIGHT);
            ctx.clip();

            ctx.font = "12px monospace";
            ctx.fillStyle = "#0f0"; // keep the green — it's the spot-it-live cue
            ctx.textAlign = "left";
            const lines = wrapLabel(ctx, label, maxWidth);
            for (let i = 0; i < lines.length; i++) {
                ctx.fillText(lines[i], LIVE_PAD_X, y + (i + 1) * LIVE_LINE_HEIGHT - 3);
            }
            ctx.restore();
        },
    };
}

function findLiveWidget(node) {
    return (node.widgets || []).find((w) => w.name === LIVE_WIDGET_NAME);
}

app.registerExtension({
    name: "DiffusionGemma.LiveView",

    // Append the live widget to every DGemmaSampler instance as it is
    // created — it lands after the node-def widgets (prompt, seed, ...,
    // thinking), i.e. at the bottom of the widget stack, with its own
    // layout-reserved lines ("explicitly add an empty line to the bottom",
    // as a widget that owns those lines). The idle text makes the reserved
    // space visibly intentional rather than a mystery gap.
    nodeCreated(node) {
        if (node.comfyClass !== DGEMMA_SAMPLER_NODE_TYPE) {
            return;
        }
        if (findLiveWidget(node)) {
            return; // already attached (defensive: configure/reload paths)
        }
        const widget = createLiveWidget();
        if (typeof node.addCustomWidget === "function") {
            node.addCustomWidget(widget);
        } else {
            node.widgets ||= [];
            node.widgets.push(widget);
        }
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

            const widget = findLiveWidget(node);
            if (!widget) {
                return;
            }
            widget.value = payload;
            node.setDirtyCanvas(true, false);
        });
    },
});
