// End-to-end timings from Python: one statement, from call to result, on plain
// NumPy and after hotkernels.install(), at each size that was recorded. Both
// axes are logarithmic, so a fixed cost per call shows as a flat start and the
// cost per element as the slope that follows.
//
// Data: data/python.json, the delivery tool's recorded result. Rows whose
// kernel is null are calls that reached a hook and were left to NumPy; for
// those the figure shows what the detour cost.
// Attributes: data-statement="x[mask]" picks the statement shown first;
// data-data="float64, half selected" narrows it to one kind of input.

import { h, s, loadData, seg, stat, fmtX, fmtNs, fmtN, scale } from "./util.js";

const GROUPS = [
  { value: "taken", label: "Calls a kernel takes" },
  { value: "passed", label: "Calls left to NumPy" },
];
const DEFAULT_STATEMENT = "x[mask]";
const DEFAULT_DATA = "float64, half selected";
// Drawing area, in viewBox units.
const W = 640, H = 310, LEFT = 62, RIGHT = 22, TOP = 18, BOTTOM = 42;
// Smallest vertical distance between two dots, in viewBox units, that leaves room for a label between them.
const MIN_LABEL_GAP = 30;

export const css = `
.pt-np { stroke: var(--np); fill: var(--np); }
.pt-hk { stroke: var(--hot); fill: var(--hot); }
.pt-line { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
.pt-gap { stroke: var(--rule-strong); stroke-dasharray: 3 3; }
.pt-x { font-size: 12.5px; font-weight: 700; font-variant-numeric: tabular-nums; paint-order: stroke; stroke: var(--bg-raised); stroke-width: 4px; }
.pt-axis text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.pt-axis line { stroke: var(--rule); }
.pt-title { font-size: 11px; fill: var(--muted); }
.pt-select { flex: 1 1 14rem; min-width: 0; }
/* Below this width the labels are too small to read, so the stage scrolls sideways instead. */
.w-stage > svg.pt-chart { min-width: 27rem; }
`;

// A round time for an axis tick: "100 ns", "1 µs", "10 ms".
function tickNs(ns) {
  if (ns < 1e3) return `${ns} ns`;
  if (ns < 1e6) return `${ns / 1e3} µs`;
  if (ns < 1e9) return `${ns / 1e6} ms`;
  return `${ns / 1e9} s`;
}

// "+0.8 ns (+1.3%)": what a hook cost a call it did not serve.
function detour(row) {
  const d = row.hotkernels_ns - row.numpy_ns;
  const sign = d < 0 ? "−" : "+";
  const abs = Math.abs(d);
  const pct = (abs / row.numpy_ns) * 100;
  return { short: `${sign}${pct.toFixed(1)}%`, long: `${sign}${fmtNs(abs)} (${sign}${pct.toFixed(1)}%)` };
}

function draw(rows, taken) {
  const svg = s("svg", { class: "pt-chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Time per call against element count, NumPy and hotkernels, both axes logarithmic" });
  const times = rows.flatMap((r) => [r.numpy_ns, r.hotkernels_ns]);
  const y0 = 10 ** Math.floor(Math.log10(Math.min(...times) * 0.8));
  const y1 = 10 ** Math.ceil(Math.log10(Math.max(...times) * 1.25));
  const ns = rows.map((r) => r.n);
  const x = scale([Math.min(...ns) / 2.2, Math.max(...ns) * 2.2], [LEFT, W - RIGHT], true);
  const y = scale([y0, y1], [H - BOTTOM, TOP], true);

  const axis = s("g", { class: "pt-axis" });
  for (let t = y0; t <= y1 * 1.0001; t *= 10) {
    axis.append(s("line", { x1: LEFT, x2: W - RIGHT, y1: y(t), y2: y(t) }),
      s("text", { x: LEFT - 8, y: y(t) + 4, "text-anchor": "end" }, tickNs(Math.round(t))));
  }
  for (const n of ns) {
    axis.append(s("line", { x1: x(n), x2: x(n), y1: H - BOTTOM, y2: H - BOTTOM + 5 }),
      s("text", { x: x(n), y: H - BOTTOM + 19, "text-anchor": "middle" }, fmtN(n)));
  }
  axis.append(s("text", { x: (LEFT + W - RIGHT) / 2, y: H - 6, "text-anchor": "middle" }, "elements"),
    s("text", { x: LEFT, y: TOP - 6, "text-anchor": "start" }, "time per call"));
  svg.append(axis);

  const path = (field) => rows.map((r, i) => `${i ? "L" : "M"}${x(r.n).toFixed(1)} ${y(r[field]).toFixed(1)}`).join(" ");
  for (const r of rows) {
    svg.append(s("line", { class: "pt-gap", x1: x(r.n), x2: x(r.n), y1: y(r.numpy_ns), y2: y(r.hotkernels_ns) }));
  }
  svg.append(s("path", { class: "pt-line pt-np", d: path("numpy_ns") }), s("path", { class: "pt-line pt-hk", d: path("hotkernels_ns") }));
  rows.forEach((r, i) => {
    const last = i === rows.length - 1 && rows.length > 1;
    // The label sits between the two dots when there is room, and above both when there is not.
    const upper = Math.min(y(r.numpy_ns), y(r.hotkernels_ns)), lower = Math.max(y(r.numpy_ns), y(r.hotkernels_ns));
    const labelY = lower - upper >= MIN_LABEL_GAP ? (upper + lower) / 2 + 4 : upper - 11;
    svg.append(
      s("circle", { class: "pt-np", cx: x(r.n), cy: y(r.numpy_ns), r: 4.5 }, s("title", {}, `NumPy, ${fmtN(r.n)} elements: ${fmtNs(r.numpy_ns)}`)),
      s("circle", { class: "pt-hk", cx: x(r.n), cy: y(r.hotkernels_ns), r: 4.5 }, s("title", {}, `hotkernels, ${fmtN(r.n)} elements: ${fmtNs(r.hotkernels_ns)}`)),
      s("text", {
        class: "pt-x", x: x(r.n) + (last ? -10 : 10), "text-anchor": last ? "end" : "start",
        y: labelY,
      },taken ? fmtX(r.numpy_ns / r.hotkernels_ns) : detour(r).short),
    );
  });
  return svg;
}

export default async function mount(el) {
  const { meta, rows } = await loadData("python");
  const key = (r) => `${r.statement}\u0000${r.data}`;
  const inGroup = (group) => rows.filter((r) => (r.kernel !== null) === (group === "taken"));

  const wantStatement = el.dataset.statement || DEFAULT_STATEMENT;
  const wantData = el.dataset.data || (el.dataset.statement ? null : DEFAULT_DATA);
  const first = rows.find((r) => r.statement === wantStatement && (!wantData || r.data === wantData))
    || rows.find((r) => r.statement === wantStatement) || rows[0];
  let group = first.kernel !== null ? "taken" : "passed";
  let current = key(first);

  const select = h("select", { class: "w-select pt-select", "aria-label": "Statement", onchange: () => { current = select.value; render(); } });
  const stage = h("div", { class: "w-stage" });
  const readout = h("div", { class: "w-readout" });
  const legend = h("div", { class: "w-legend" },
    h("span", {}, h("i", { class: "np" }), "plain NumPy"), h("span", {}, h("i", { class: "hot" }), "after hotkernels.install()"));

  function fillSelect() {
    const seen = new Map();
    for (const r of inGroup(group)) if (!seen.has(key(r))) seen.set(key(r), `${r.statement} · ${r.data}`);
    if (!seen.has(current)) current = seen.keys().next().value;
    select.replaceChildren(...[...seen].map(([value, label]) => h("option", { value, selected: value === current }, label)));
    select.value = current;
  }

  function render() {
    const taken = group === "taken";
    const shown = inGroup(group).filter((r) => key(r) === current).sort((a, b) => a.n - b.n);
    stage.replaceChildren(draw(shown, taken));
    readout.replaceChildren(...shown.map((r) => {
      const tile = stat(`${fmtN(r.n)} elements: ${fmtNs(r.numpy_ns)} → ${fmtNs(r.hotkernels_ns)}`, taken ? "hot" : "");
      tile.set(taken ? fmtX(r.numpy_ns / r.hotkernels_ns) : detour(r).long);
      return tile;
    }));
  }

  el.append(
    h("div", { class: "w-controls" },
      seg(GROUPS, group, (v) => { group = v; fillSelect(); render(); }),
      select),
    stage, legend, readout,
    h("p", { class: "w-note" },
      `${meta.cpu}, NumPy ${meta.numpy}, Python ${meta.python}. One Python statement from call to result, the best of 9 alternating batches. `
      + `The machine was in use: load average ${meta.load_average} when the run started.`),
  );
  fillSelect();
  render();
}
