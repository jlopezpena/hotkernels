// Latency against throughput: why a sum is split over several accumulators.
//
// The byte-sum kernel adds each 16-byte vector into an accumulator with one
// instruction. The result of that instruction is ready three cycles later, and
// the next add into the same accumulator has to wait for it. That is a
// dependency chain: with one accumulator the loop does one vector per three
// cycles, however fast everything else is. With k accumulators there are k
// independent chains, so k vectors per three cycles, until something else runs
// out: the core loads three vectors per cycle and no more.
//
//   cycles per vector = max(3 / k, 1 / 3)
//
// The three measured points are from kernels/sum_u8/attempts.md, attempt 2.
//
// Attributes:
//   data-k  initial number of accumulators: 1, 2, 4, 8 or 16 (default 1)

import { h, s, seg, stat } from "./util.js";

// Cycles from an accumulator going into the add until its result can be used.
const LATENCY = 3;
// Vectors the core can load per cycle.
const LOADS_PER_CYCLE = 3;
const VECTOR_BYTES = 16;
// Clock of an M1 Pro performance core, for converting cycles to time.
const GHZ = 3.2;
const CHOICES = [1, 2, 4, 8, 16];
// Cycles per vector measured inside the first-level cache, by accumulator count.
const MEASURED = [[4, 0.75], [8, 0.37], [16, 0.333]];
// Accumulator rows drawn before the rest are summarised in a line of text.
const MAX_ROWS = 8;
// Cycles shown on the timeline.
const CYCLES = 12;

export const css = `
.chain-parts { display: flex; flex-wrap: wrap; gap: 1rem 1.5rem; align-items: flex-start; }
.chain-parts > div:first-child { flex: 1 1 24rem; min-width: 0; }
.chain-parts > div:last-child { flex: 0 1 19rem; min-width: 15rem; }
.chain-title { font-size: .74rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .5rem; }
.chain-timeline { min-width: 30rem; }
.chain-box { fill: var(--hot-soft); stroke: var(--hot); stroke-width: 1; }
.chain-load { stroke: var(--np); stroke-width: 1.5; }
.chain-model { fill: none; stroke: var(--hot); stroke-width: 2; }
.chain-limit { stroke: var(--np); stroke-width: 1.2; stroke-dasharray: 4 4; }
.chain-now { fill: var(--hot); }
.chain-measured { fill: var(--bg-raised); stroke: var(--ink); stroke-width: 1.6; }
.chain-grid { stroke: var(--rule); stroke-width: 1; }
`;

const cyclesPerVector = (k) => Math.max(LATENCY / k, 1 / LOADS_PER_CYCLE);

function timeline(k) {
  const W = 640, LEFT = 64, RIGHT = 8, TOP = 26, ROW = 26, BOX = 20;
  const rows = Math.min(k, MAX_ROWS);
  const more = k - rows;
  const axisY = TOP + rows * ROW + 6;
  const H = axisY + 34 + (more ? 18 : 0);
  const x = (cycle) => LEFT + (cycle / CYCLES) * (W - LEFT - RIGHT);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chain-timeline", role: "img",
    "aria-label": `Timeline of adds into ${k} accumulator${k === 1 ? "" : "s"} over ${CYCLES} cycles` });

  for (let c = 0; c <= CYCLES; c++) {
    svg.append(s("line", { class: "chain-grid", x1: x(c), x2: x(c), y1: TOP - 4, y2: axisY }));
    svg.append(s("text", { class: "muted", x: x(c), y: axisY + 14, "text-anchor": "middle", "font-size": 11 }, c));
  }
  svg.append(s("text", { class: "muted", x: (LEFT + W - RIGHT) / 2, y: axisY + 29, "text-anchor": "middle", "font-size": 11 }, "cycles"));

  // Loads arrive three per cycle, whatever the adds are doing.
  svg.append(s("text", { class: "muted", x: LEFT - 8, y: 14, "text-anchor": "end", "font-size": 11 }, "loads"));
  for (let j = 0; j / LOADS_PER_CYCLE < CYCLES; j++) {
    const at = x(j / LOADS_PER_CYCLE);
    svg.append(s("line", { class: "chain-load", x1: at, x2: at, y1: 6, y2: 16 }));
  }

  for (let r = 0; r < rows; r++) {
    svg.append(s("text", { class: "muted", x: LEFT - 8, y: TOP + r * ROW + BOX / 2 + 4, "text-anchor": "end", "font-size": 11 }, `acc ${r}`));
  }
  // Vector j is loaded at cycle j/3 and added into accumulator j mod k,
  // starting when both the vector and that accumulator's last result are ready.
  const free = new Array(k).fill(0);
  for (let j = 0; j / LOADS_PER_CYCLE < CYCLES; j++) {
    const acc = j % k;
    const start = Math.max(j / LOADS_PER_CYCLE, free[acc]);
    free[acc] = start + LATENCY;
    if (acc >= rows || start >= CYCLES) continue;
    const y = TOP + acc * ROW;
    svg.append(s("rect", { class: "chain-box", x: x(start) + 1, y, width: x(start + LATENCY) - x(start) - 2, height: BOX, rx: 4 }));
    svg.append(s("text", { x: x(start) + 8, y: y + BOX / 2 + 4, "font-size": 11 }, `vector ${j}`));
  }
  if (more) {
    svg.append(s("text", { class: "muted", x: LEFT, y: H - 4, "font-size": 11 }, `+ ${more} more accumulators, the same pattern`));
  }
  return svg;
}

function chart(k) {
  const W = 300, H = 230, L = 46, R = 14, T = 14, B = 40;
  const lx = (v) => L + (Math.log2(v) / Math.log2(16)) * (W - L - R);
  const ly = (v) => T + ((Math.log(4) - Math.log(v)) / (Math.log(4) - Math.log(0.25))) * (H - T - B);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Cycles per vector against the number of accumulators: the model and three measurements" });

  for (const v of [0.25, 0.5, 1, 2, 4]) {
    svg.append(s("line", { class: "chain-grid", x1: L, x2: W - R, y1: ly(v), y2: ly(v) }));
    svg.append(s("text", { class: "muted", x: L - 7, y: ly(v) + 4, "text-anchor": "end", "font-size": 11 }, v));
  }
  for (const v of CHOICES) {
    svg.append(s("text", { class: "muted", x: lx(v), y: H - B + 16, "text-anchor": "middle", "font-size": 11 }, v));
  }
  svg.append(s("text", { class: "muted", x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11 }, "accumulators"));
  svg.append(s("text", { class: "muted", x: 12, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 11,
    transform: `rotate(-90 12 ${(T + H - B) / 2})` }, "cycles per vector"));

  const limit = 1 / LOADS_PER_CYCLE;
  svg.append(s("line", { class: "chain-limit", x1: L, x2: W - R, y1: ly(limit), y2: ly(limit) }));
  svg.append(s("text", { x: L + 4, y: ly(limit) - 6, "font-size": 11, style: "fill: var(--np)" }, "load limit: 3 vectors per cycle"));

  const points = [];
  for (let i = 0; i <= 64; i++) {
    const v = 2 ** (i / 16);
    points.push(`${lx(v).toFixed(1)},${ly(cyclesPerVector(v)).toFixed(1)}`);
  }
  svg.append(s("polyline", { class: "chain-model", points: points.join(" ") }));
  for (const [mk, cycles] of MEASURED) {
    svg.append(s("circle", { class: "chain-measured", cx: lx(mk), cy: ly(cycles), r: 5 }));
  }
  svg.append(s("circle", { class: "chain-now", cx: lx(k), cy: ly(cyclesPerVector(k)), r: 4 }));

  svg.append(s("circle", { class: "chain-measured", cx: W - R - 78, cy: T + 8, r: 4.5 }));
  svg.append(s("text", { x: W - R - 68, y: T + 12, "font-size": 11 }, "measured"));
  svg.append(s("line", { class: "chain-model", x1: W - R - 84, x2: W - R - 72, y1: T + 26, y2: T + 26 }));
  svg.append(s("text", { x: W - R - 68, y: T + 30, "font-size": 11 }, "max(3/k, ⅓)"));
  return svg;
}

export default async function mount(el) {
  let k = CHOICES.includes(Number(el.dataset.k)) ? Number(el.dataset.k) : 1;

  const timelineBox = h("div", { class: "w-stage" });
  const chartBox = h("div", { class: "w-stage" });
  const cycles = stat("cycles per vector", "hot");
  const bytes = stat("bytes per cycle");
  const time = stat(`ns per byte at ${GHZ} GHz`);
  const limit = stat("what limits it");
  const explain = h("div", { class: "w-note", "aria-live": "polite" });

  function render() {
    const c = cyclesPerVector(k);
    const chainBound = LATENCY / k > 1 / LOADS_PER_CYCLE;
    timelineBox.replaceChildren(timeline(k));
    chartBox.replaceChildren(chart(k));
    cycles.set(c >= 1 ? c.toFixed(2).replace(/\.?0+$/, "") : c.toFixed(3));
    bytes.set((VECTOR_BYTES / c).toFixed(1).replace(/\.0$/, ""));
    time.set((c / VECTOR_BYTES / GHZ).toFixed(4));
    limit.set(chainBound ? "the chain" : "the load ports");
    explain.textContent = chainBound
      ? `Each accumulator is busy for ${LATENCY} cycles per add, so ${k} accumulator${k === 1 ? " takes" : "s take"} ${k} vector${k === 1 ? "" : "s"} every ${LATENCY} cycles. Loads arrive three per cycle and queue up behind the adds.`
      : `${k} accumulators could take ${k} vectors every ${LATENCY} cycles, but only ${LOADS_PER_CYCLE * LATENCY} arrive in that time. The adds now wait for the loads: every accumulator has idle gaps.`;
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Accumulators"),
      seg(CHOICES.map((v) => ({ value: v, label: String(v) })), k, (v) => { k = Number(v); render(); })),
    h("div", { class: "chain-parts" },
      h("div", {}, h("div", { class: "chain-title" }, "One add is three cycles long"), timelineBox),
      h("div", {}, h("div", { class: "chain-title" }, "Model and measurement"), chartBox)),
    explain,
    h("div", { class: "w-readout" }, cycles, bytes, time, limit),
    h("div", { class: "w-note" }, "Measured on an Apple M1 Pro inside the first-level cache (sum_u8, attempt 2). Beyond that cache every variant ran at the same speed: memory is the limit."),
  );
  render();
}
