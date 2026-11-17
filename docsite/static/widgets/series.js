// A line chart with a logarithmic x-axis, drawn from data written on the
// element itself. Used for figures such as time per element against array
// size, where the steps in the line are the edges of the caches.
//
// Attributes:
//   data-series       JSON: [{"label": "hotkernels", "kind": "hot" | "np" | "cc", "points": {"x": [...], "y": [...]}}, ...]
//                     Points may also be a list of [x, y] pairs, but not in a page's Markdown: two opening
//                     square brackets in a row start a template expression there and the build fails.
//   data-per-element  if present, plot y / x instead of y (total time -> time per element)
//   data-logy         if present, the y-axis is logarithmic; otherwise linear from 0
//   data-xlabel       caption of the x-axis (default "elements")
//   data-ylabel       caption of the y-axis, also used as the unit when a point is read out
//   data-marks        JSON: [{"x": 32768, "label": "L1 data cache, 128 KB"}, ...] dashed vertical lines
// Pointing at a dot, or tabbing to it, shows its value under the chart.

import { h, s, fmtN, scale } from "./util.js";

// Drawing area, in viewBox units.
const W = 680, H = 340, LEFT = 58, RIGHT = 20, TOP = 26, BOTTOM = 44;
const KINDS = { hot: "var(--hot)", np: "var(--np)", cc: "var(--cc)" };
// Most x-axis labels that fit; beyond this every other one is dropped.
const MAX_X_LABELS = 12;

export const css = `
.sr-line { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
.sr-dot { stroke: var(--bg-raised); stroke-width: 1.5; cursor: pointer; outline: none; }
.sr-dot:hover, .sr-dot:focus { stroke: var(--ink); stroke-width: 2; }
.sr-axis text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.sr-axis line { stroke: var(--rule); }
.sr-mark line { stroke: var(--muted); stroke-dasharray: 4 4; }
.sr-mark text { font-size: 10.5px; fill: var(--muted); }
.sr-read { min-height: 1.3em; }
/* Below this width the labels are too small to read, so the stage scrolls sideways instead. */
.w-stage > svg.sr-chart { min-width: 27rem; }
`;

function attrJSON(el, name, fallback) {
  const text = el.getAttribute(name);
  if (!text) return fallback;
  try { return JSON.parse(text); } catch (e) { throw new Error(`${name} is not valid JSON: ${e.message}`); }
}

// Points as [x, y] pairs, from either form data-series accepts.
function pairs(points) {
  return Array.isArray(points) ? points : points.x.map((x, i) => [x, points.y[i]]);
}

// 1048576 -> "1M", 4096 -> "4K"; anything that is not a round binary multiple in full.
function compact(n) {
  if (n >= 1 << 20 && n % (1 << 20) === 0) return `${n >> 20}M`;
  if (n >= 1024 && n % 1024 === 0) return `${n >> 10}K`;
  return fmtN(n);
}

// A value with about three significant digits and no trailing zeros.
function num(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return fmtN(Math.round(v));
  return String(Number(v.toPrecision(a < 1 ? 2 : 3)));
}

// Round steps for a linear axis from 0 to `top`.
function linearTicks(top) {
  const raw = top / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((m) => m >= raw);
  const ticks = [];
  for (let t = 0; t <= top + step * 1e-9; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}

// Powers of ten covering [lo, hi], with 2 and 5 added when there are few decades.
function logTicks(lo, hi) {
  const ticks = [];
  const few = Math.log10(hi / lo) < 2.5;
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    for (const m of few ? [1, 2, 5] : [1]) {
      const t = m * 10 ** p;
      if (t >= lo * 0.999 && t <= hi * 1.001) ticks.push(Number(t.toPrecision(12)));
    }
  }
  return ticks;
}

export default async function mount(el) {
  const perElement = el.hasAttribute("data-per-element");
  const logY = el.hasAttribute("data-logy");
  const xLabel = el.dataset.xlabel || "elements";
  const yLabel = el.dataset.ylabel || "";
  const marks = attrJSON(el, "data-marks", []);
  const series = attrJSON(el, "data-series", []).map((sr) => ({
    ...sr,
    color: KINDS[sr.kind] || "var(--ink)",
    points: pairs(sr.points).map(([x, y]) => [x, perElement ? y / x : y]),
  }));
  if (!series.length) throw new Error("series: data-series is empty");

  const xs = [...new Set(series.flatMap((sr) => sr.points.map((p) => p[0])))].sort((a, b) => a - b);
  const ys = series.flatMap((sr) => sr.points.map((p) => p[1]));
  const x = scale([xs[0] / 1.5, xs[xs.length - 1] * 1.5], [LEFT, W - RIGHT], true);
  let y, yTicks;
  if (logY) {
    const lo = 10 ** Math.floor(Math.log10(Math.min(...ys) * 0.9));
    const hi = 10 ** Math.ceil(Math.log10(Math.max(...ys) * 1.1));
    y = scale([lo, hi], [H - BOTTOM, TOP], true);
    yTicks = logTicks(lo, hi);
  } else {
    yTicks = linearTicks(Math.max(...ys) * 1.08);
    y = scale([0, yTicks[yTicks.length - 1]], [H - BOTTOM, TOP]);
  }

  const svg = s("svg", { class: "sr-chart", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${yLabel || "value"} against ${xLabel}: ${series.map((sr) => sr.label).join(", ")}` });

  const axis = s("g", { class: "sr-axis" });
  for (const t of yTicks) {
    axis.append(s("line", { x1: LEFT, x2: W - RIGHT, y1: y(t), y2: y(t) }),
      s("text", { x: LEFT - 8, y: y(t) + 4, "text-anchor": "end" }, num(t)));
  }
  const skip = xs.length > MAX_X_LABELS ? 2 : 1;
  xs.forEach((v, i) => {
    axis.append(s("line", { x1: x(v), x2: x(v), y1: H - BOTTOM, y2: H - BOTTOM + 5 }));
    if (i % skip === 0) axis.append(s("text", { x: x(v), y: H - BOTTOM + 19, "text-anchor": "middle" }, compact(v)));
  });
  axis.append(s("text", { x: (LEFT + W - RIGHT) / 2, y: H - 6, "text-anchor": "middle" }, xLabel));
  if (yLabel) axis.append(s("text", { x: LEFT, y: TOP - 11, "text-anchor": "start" }, yLabel));
  svg.append(axis);

  marks.forEach((mark, i) => {
    const mx = x(mark.x);
    const right = mx > (LEFT + W - RIGHT) / 2;
    svg.append(s("g", { class: "sr-mark" },
      s("line", { x1: mx, x2: mx, y1: TOP, y2: H - BOTTOM }),
      s("text", { x: mx + (right ? -6 : 6), y: TOP + 11 + (i % 2) * 14, "text-anchor": right ? "end" : "start" }, mark.label)));
  });

  const read = h("p", { class: "w-note sr-read", "aria-live": "polite" }, "Point at a dot to read its value.");
  const show = (sr, [px, py]) => {
    read.replaceChildren(h("b", {}, sr.label), ` at ${fmtN(px)} ${xLabel}: ${num(py)}${yLabel ? ` ${yLabel}` : ""}`);
  };

  for (const sr of series) {
    const d = sr.points.map(([px, py], i) => `${i ? "L" : "M"}${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join(" ");
    svg.append(s("path", { class: "sr-line", d, style: `stroke:${sr.color}` }));
  }
  // Dots go on top of every line, so one series does not hide another's points.
  for (const sr of series) {
    for (const point of sr.points) {
      svg.append(s("circle", {
        class: "sr-dot", cx: x(point[0]), cy: y(point[1]), r: 4.5, tabindex: 0, style: `fill:${sr.color}`,
        onmouseenter: () => show(sr, point), onfocus: () => show(sr, point), onclick: () => show(sr, point),
      }, s("title", {}, `${sr.label}, ${fmtN(point[0])}: ${num(point[1])}`)));
    }
  }

  el.append(
    h("div", { class: "w-stage" }, svg),
    h("div", { class: "w-legend" }, series.map((sr) => h("span", {}, h("i", { style: `background:${sr.color}` }), sr.label))),
    read,
  );
}
