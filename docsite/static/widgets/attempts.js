// Every optimisation attempt on record, kernel by kernel.
//
// The data is the kernels' attempts logs, parsed when the site is built
// (data/attempts.json). For the chosen kernel the chart follows the version
// that was kept: each kept attempt multiplies the cumulative speedup over the
// first version by the ratio `judge compare` printed for it. An attempt that
// was reverted shows as a branch cut off the line, at the height it would
// have reached. An attempt that was never compared with its predecessor (a
// first version, or one screened outside the judge) has no ratio and is a
// tick on the bottom edge. The list underneath has the log entry of each.
//
// data-kernel: the kernel shown first (default "argmax_f32").

import { h, s, loadData, stat, fmtX, scale } from "./util.js";

const DEFAULT_KERNEL = "argmax_f32";
// Chart size in SVG units, and the margins around the plot area.
const WIDTH = 720;
const HEIGHT = 250;
const MARGIN = { left: 48, right: 16, top: 16, bottom: 34 };
// The vertical axis is padded by this factor beyond the lowest and highest point.
const Y_PAD = 1.1;
// Speedups an axis tick may sit on, most wanted first. A tick is drawn when
// it is inside the chart's range and far enough from the ticks already chosen.
const TICK_CANDIDATES = [1, 2, 5, 10, 20, 50, 100, 0.5, 3, 30, 1.5, 7, 15, 1.2, 1.1, 0.9, 1.05, 0.8, 1.3, 4, 0.95, 1.02, 0.7, 0.3];
const MAX_TICKS = 6;
// Smallest vertical distance between two axis ticks, in SVG units.
const MIN_TICK_GAP = 22;
// A compare line that starts like this was not a `judge compare` run against the version before.
const NOT_COMPARED = /^\s*(not run|n\/a)/i;
// What `judge compare` said, in words.
const VERDICT_TEXT = {
  faster: "significantly faster",
  slower: "significantly slower",
  same: "no significant difference",
  none: "not compared",
};

export const css = `
.w .at-top { font-size: .84rem; color: var(--muted); margin: 0 0 .9rem; }
.at-top b { color: var(--ink); }
.w-stage svg.at-chart { width: 100%; min-width: 34rem; max-width: none; }
.at-chart .line { fill: none; stroke: var(--hot); stroke-width: 2.5; stroke-linejoin: round; }
.at-chart .unit { stroke: var(--rule-strong); stroke-dasharray: 2 4; }
.at-chart .stub { stroke-width: 1.5; stroke-dasharray: 4 3; fill: none; }
.at-pt { cursor: pointer; outline: none; }
.at-pt .hit { fill: transparent; }
.at-pt .dot.kept { fill: var(--hot); stroke: var(--bg-raised); stroke-width: 2; }
.at-pt .dot.cut { fill: var(--bg-raised); stroke-width: 2; }
.at-pt .tick { stroke: var(--muted); stroke-width: 2.5; stroke-linecap: round; }
.at-pt.slower .dot.cut, .at-pt.slower .stub { stroke: var(--bad); }
.at-pt.same .dot.cut, .at-pt.same .stub, .at-pt.faster .dot.cut, .at-pt.faster .stub { stroke: var(--muted); }
.at-pt .ring { fill: none; stroke: transparent; stroke-width: 2; }
.at-pt.sel .ring, .at-pt:focus-visible .ring { stroke: var(--ink); }
.w .at-list { list-style: none; margin: 1rem 0 0; padding: 0; border-top: 1px solid var(--rule); }
.w .at-list li { border-bottom: 1px solid var(--rule); margin: 0; }
.at-row { display: grid; grid-template-columns: 2rem 1fr; gap: .1rem .6rem; align-items: baseline; width: 100%; text-align: left;
  font: inherit; color: inherit; background: transparent; border: 0; padding: .55rem .4rem; cursor: pointer; }
.at-row:hover, .at-list li.sel .at-row { background: var(--panel); }
.at-row .num { font-family: var(--mono); font-size: .78rem; color: var(--hot); grid-row: span 2; }
.at-row .title { font-weight: 600; line-height: 1.35; }
.at-row .meta { font-size: .78rem; color: var(--muted); display: flex; flex-wrap: wrap; gap: .3rem .6rem; align-items: baseline; }
.at-badge { font-size: .68rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; padding: .08rem .45rem; border-radius: 5px; background: var(--panel); color: var(--muted); border: 1px solid var(--rule-strong); }
.at-badge.kept { background: var(--hot-soft); color: var(--hot); border-color: var(--hot); }
.at-ratio { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.w .at-detail { display: none; margin: 0; padding: .2rem .6rem 1rem 3rem; font-size: .88rem; line-height: 1.55; }
.at-list li.sel .at-detail { display: block; }
.at-detail dt { font-size: .68rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); margin-top: .7rem; }
.at-detail dd { margin: .15rem 0 0; overflow-wrap: anywhere; }
.at-detail code, .at-row code { white-space: normal; }
@media (max-width: 40rem) { .w .at-detail { padding-left: .6rem; } }
`;

// Whether an attempt's ratio is a `judge compare` result against the version
// before it. A log's first attempt has no predecessor, so any ratio on it is
// against something else (usually naive C).
function trusted(attempt) {
  if (attempt.first || attempt.verdict === "none" || attempt.speedup === null) return false;
  return !NOT_COMPARED.test(attempt.compare.replace(/<[^>]+>/g, ""));
}

// What `judge compare` said about an attempt, in words.
function verdictText(attempt) {
  return VERDICT_TEXT[trusted(attempt) ? attempt.verdict : "none"];
}

// "1.021" or "13.3": a ratio with the precision the judge prints.
function ratioText(x) {
  return x < 2 ? x.toFixed(3) : x.toPrecision(3);
}

// Places each attempt on the chart: kept ones on the cumulative line, reverted
// ones where they would have taken it, and the rest without a height.
function layout(kernel) {
  let cumulative = 1;
  const points = kernel.attempts.map((attempt) => {
    if (!trusted(attempt)) return { attempt, kind: "tick", before: cumulative, y: null };
    if (attempt.kept) {
      const before = cumulative;
      cumulative *= attempt.speedup;
      return { attempt, kind: "kept", before, y: cumulative };
    }
    return { attempt, kind: "cut", before: cumulative, y: cumulative * attempt.speedup };
  });
  return { points, cumulative };
}

// Axis ticks inside [lo, hi] that do not crowd each other; `y` maps a speedup to its height.
function ticks(lo, hi, y) {
  const chosen = [];
  for (const t of TICK_CANDIDATES) {
    if (chosen.length >= MAX_TICKS) break;
    if (t < lo || t > hi) continue;
    if (chosen.every((c) => Math.abs(y(c) - y(t)) >= MIN_TICK_GAP)) chosen.push(t);
  }
  return chosen;
}

function drawChart(kernel, onSelect) {
  const { points, cumulative } = layout(kernel);
  const heights = [1, ...points.map((p) => p.before), ...points.filter((p) => p.y !== null).map((p) => p.y)];
  const lo = Math.min(...heights) / Y_PAD;
  const hi = Math.max(...heights) * Y_PAD;
  const plot = { left: MARGIN.left, right: WIDTH - MARGIN.right, top: MARGIN.top, bottom: HEIGHT - MARGIN.bottom };
  const y = scale([lo, hi], [plot.bottom, plot.top], true);
  const count = points.length;
  const x = (i) => plot.left + ((i + 0.5) / count) * (plot.right - plot.left);

  const svg = s("svg", { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, class: "at-chart", role: "group",
    "aria-label": `Cumulative speedup of ${kernel.name} over its first version, attempt by attempt` });

  const axis = s("g", { class: "axis" });
  for (const t of ticks(lo, hi, y)) {
    axis.append(
      s("line", { x1: plot.left, x2: plot.right, y1: y(t), y2: y(t), class: t === 1 ? "unit" : "" }),
      s("text", { x: plot.left - 8, y: y(t) + 4, "text-anchor": "end", "font-size": 11 }, `${t}x`));
  }
  points.forEach((p, i) => {
    axis.append(s("text", { x: x(i), y: HEIGHT - 10, "text-anchor": "middle", "font-size": 11 }, p.attempt.n));
  });
  axis.append(s("text", { x: plot.left, y: HEIGHT - 10, "text-anchor": "end", "font-size": 11, dx: -8 }, "attempt"));
  svg.append(axis);

  // The kept line: flat until a kept attempt, then a step to the new level.
  let path = `M ${plot.left} ${y(1)}`;
  points.forEach((p, i) => {
    path += ` H ${x(i)}`;
    if (p.kind === "kept") path += ` V ${y(p.y)}`;
  });
  path += ` H ${plot.right}`;
  svg.append(s("path", { class: "line", d: path }));

  const groups = points.map((p, i) => {
    const a = p.attempt;
    const label = `Attempt ${a.n}: ${a.kept ? "kept" : "reverted"}, ${verdictText(a)}${trusted(a) ? `, ${ratioText(a.speedup)}x` : ""}`;
    const g = s("g", { class: `at-pt ${trusted(a) ? a.verdict : "none"}`, tabindex: 0, role: "button", "aria-label": label,
      onclick: () => onSelect(i, true), onfocus: () => onSelect(i, false),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(i, true); } } });
    const cx = x(i);
    if (p.kind === "tick") {
      g.append(
        s("rect", { class: "hit", x: cx - 12, y: plot.bottom - 22, width: 24, height: 26 }),
        s("circle", { class: "ring", cx, cy: plot.bottom - 6, r: 9 }),
        s("line", { class: "tick", x1: cx, x2: cx, y1: plot.bottom - 10, y2: plot.bottom - 2 }));
    } else {
      const cy = y(p.y);
      if (p.kind === "cut") g.append(s("line", { class: "stub", x1: cx, x2: cx, y1: y(p.before), y2: cy }));
      g.append(
        s("circle", { class: "hit", cx, cy, r: 13 }),
        s("circle", { class: "ring", cx, cy, r: 9 }),
        s("circle", { class: `dot ${p.kind}`, cx, cy, r: 5 }));
    }
    svg.append(g);
    return g;
  });
  return { svg, groups, cumulative };
}

function attemptRow(attempt, onSelect) {
  const meta = h("span", { class: "meta" },
    h("span", { class: `at-badge ${attempt.kept ? "kept" : ""}` }, attempt.kept ? "kept" : "reverted"),
    h("span", {}, verdictText(attempt)),
    trusted(attempt) && h("span", { class: "at-ratio" },
      `${ratioText(attempt.speedup)}x [${ratioText(attempt.lo)} – ${ratioText(attempt.hi)}]`));
  const row = h("button", { type: "button", class: "at-row", "aria-expanded": "false", onclick: onSelect },
    h("span", { class: "num" }, attempt.n), h("span", { class: "title", html: attempt.title }), meta);
  const detail = h("dl", { class: "at-detail" });
  for (const [label, html] of [["Idea", attempt.idea], ["check", attempt.check], ["compare", attempt.compare], ["Notes", attempt.notes]]) {
    if (html) detail.append(h("dt", {}, label), h("dd", { html }));
  }
  return h("li", {}, row, detail);
}

export default async function mount(el) {
  const data = await loadData("attempts");
  for (const k of data.kernels) k.attempts[0].first = true;
  const wanted = el.dataset.kernel || DEFAULT_KERNEL;
  let kernel = data.kernels.find((k) => k.name === wanted) || data.kernels[0];
  let selected = 0;
  let groups = [];

  const select = h("select", { class: "w-select", "aria-label": "Kernel",
    onchange: () => { kernel = data.kernels[select.selectedIndex]; show(); } });
  for (const k of data.kernels) {
    const kept = k.attempts.filter((a) => a.kept).length;
    select.append(h("option", { selected: k === kernel },
      `${k.name}: ${k.attempts.length} attempts, ${kept} kept${k.assembly ? "" : " (ships as C)"}`));
  }

  const stage = h("div", { class: "w-stage" });
  const list = h("ol", { class: "at-list", onkeydown: onListKey });
  const total = stat("attempts");
  const kept = stat("kept", "hot");
  const reverted = stat("reverted");
  const gain = stat("kept gains multiplied, over the first version", "hot");

  function choose(index, scroll) {
    selected = index;
    groups.forEach((g, i) => g.classList.toggle("sel", i === index));
    [...list.children].forEach((li, i) => {
      li.classList.toggle("sel", i === index);
      li.firstChild.setAttribute("aria-expanded", String(i === index));
    });
    if (scroll) list.children[index]?.scrollIntoView({ block: "nearest", behavior: "auto" });
  }

  function onListKey(event) {
    const step = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const next = Math.min(kernel.attempts.length - 1, Math.max(0, selected + step));
    choose(next, false);
    list.children[next].firstChild.focus();
  }

  function show() {
    const chart = drawChart(kernel, choose);
    groups = chart.groups;
    stage.replaceChildren(chart.svg);
    list.replaceChildren(...kernel.attempts.map((a, i) => attemptRow(a, () => choose(i, false))));
    const keptCount = kernel.attempts.filter((a) => a.kept).length;
    total.set(String(kernel.attempts.length));
    kept.set(String(keptCount));
    reverted.set(String(kernel.attempts.length - keptCount));
    gain.set(fmtX(chart.cumulative));
    // Open on the attempt that moved the line most.
    const gains = kernel.attempts.map((a) => (a.kept && trusted(a) ? a.speedup : 0));
    choose(Math.max(0, gains.indexOf(Math.max(...gains))), false);
  }

  el.append(
    h("p", { class: "at-top" }, "All ", h("b", {}, String(data.kernels.length)), " kernels: ",
      h("b", {}, String(data.total)), " attempts, ", h("b", {}, String(data.kept)), " kept, ",
      h("b", {}, String(data.reverted)), " reverted."),
    h("div", { class: "w-controls" }, h("span", { class: "w-label" }, "Kernel"), select),
    stage,
    h("div", { class: "w-legend" },
      h("span", {}, h("i", { class: "hot" }), "kept: the line steps to the new speedup"),
      h("span", {}, h("i", { class: "bad" }), "reverted, slower"),
      h("span", {}, h("i", {}), "reverted, no significant difference, or never compared (tick)")),
    h("div", { class: "w-readout" }, total, kept, reverted, gain),
    list,
  );
  show();
}
