// How the judge decides whether an edit is "significantly faster".
//
// It follows judge/bench.py: 30 samples of each version, taken alternately;
// the speedup is the ratio of the two median times; 2,000 bootstrap resamples
// draw sample indices in pairs (the same indices for both versions, because
// sample i of each was taken at nearly the same moment and shares whatever
// noise the machine had then); the 95% interval runs from the 2.5th to the
// 97.5th percentile of the resampled ratios; a difference is significant when
// that interval does not include 1.0.
//
// The timings here are simulated. The old version takes 100 units and the new
// one 100 / (true speedup). Sample i of both is multiplied by one shared
// factor, the drift of the machine, and each by a factor of its own. The
// shared noise is twice the independent noise.
//
// Attributes: data-speedup (default 1.03) and data-noise (percent, default 2)
// set where the two sliders start.

import { h, s, range, button, stat, rng, scale } from "./util.js";

const SAMPLES = 30;
const RESAMPLES = 2000;
const INTERVAL = [0.025, 0.975];
const OLD_TIME = 100;
const SHARED_TO_INDEPENDENT = 2;
const REPEATS = 200;
// Measurements run between pauses during "Repeat", so the page stays responsive.
const REPEATS_PER_FRAME = 10;
const BINS = 41;
// Drawing areas, in viewBox units.
const W = 640, STRIP_H = 170, HIST_H = 190, LEFT = 46, RIGHT = 16;
// Room the two labels of the histogram need beside their lines, in viewBox units.
const ONE_TAG_WIDTH = 92, POINT_TAG_WIDTH = 104;

const FASTER = "faster", SLOWER = "slower", SAME = "same";

export const css = `
.bs-old { fill: var(--cc); stroke: var(--cc); }
.bs-new { fill: var(--hot); stroke: var(--hot); }
.bs-trace { fill: none; stroke-width: 1.2; opacity: .45; }
.bs-axis text, .bs-cap { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.bs-axis line { stroke: var(--rule); }
.bs-bar { fill: var(--rule-strong); }
.bs-bar.in { fill: var(--hot); opacity: .75; }
.bs-one { stroke: var(--ink); stroke-width: 1.5; stroke-dasharray: 5 4; }
.bs-point { stroke: var(--hot); stroke-width: 2; }
.bs-tag { font-size: 11px; font-weight: 650; paint-order: stroke; stroke: var(--bg-raised); stroke-width: 4px; }
.bs-tag.hot { fill: var(--hot); }
.bs-plots { display: grid; gap: .6rem; }
/* Below this width the labels are too small to read, so the stage scrolls sideways instead. */
.bs-plots svg { min-width: 27rem; }
.w-verdict.bs-neutral { color: var(--ink); }
`;

// A standard normal draw (Box–Muller) from a uniform generator.
function normal(random) {
  const u = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

// One timing run: SAMPLES interleaved samples of the old and the new version.
function measure(random, speedup, noise) {
  const old = new Float64Array(SAMPLES), fresh = new Float64Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) {
    const machine = Math.exp(SHARED_TO_INDEPENDENT * noise * normal(random));
    old[i] = OLD_TIME * machine * Math.exp(noise * normal(random));
    fresh[i] = (OLD_TIME / speedup) * machine * Math.exp(noise * normal(random));
  }
  return { old, fresh };
}

// Median of a typed array; sorts it in place.
function median(values) {
  values.sort();
  const mid = values.length >> 1;
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

// The judge's statistic: ratio of medians, with a paired bootstrap interval.
function analyse({ old, fresh }, random) {
  const point = median(old.slice()) / median(fresh.slice());
  const ratios = new Float64Array(RESAMPLES);
  const a = new Float64Array(SAMPLES), b = new Float64Array(SAMPLES);
  for (let r = 0; r < RESAMPLES; r++) {
    for (let i = 0; i < SAMPLES; i++) {
      const j = Math.floor(random() * SAMPLES);
      a[i] = old[j];
      b[i] = fresh[j];
    }
    ratios[r] = median(a) / median(b);
  }
  const sorted = ratios.slice().sort();
  const at = (q) => {
    // Linear interpolation between order statistics, as np.percentile does.
    const position = q * (RESAMPLES - 1), below = Math.floor(position);
    return sorted[below] + (sorted[Math.min(below + 1, RESAMPLES - 1)] - sorted[below]) * (position - below);
  };
  const lo = at(INTERVAL[0]), hi = at(INTERVAL[1]);
  return { point, lo, hi, ratios, verdict: lo > 1 ? FASTER : hi < 1 ? SLOWER : SAME };
}

const x3 = (v) => `${v.toFixed(3)}x`;

function verdictText(result) {
  const numbers = `${x3(result.point)} [${result.lo.toFixed(3)} – ${result.hi.toFixed(3)}]`;
  if (result.verdict === FASTER) return { kind: "good", text: `significantly faster, ${numbers}: the edit is kept` };
  if (result.verdict === SLOWER) return { kind: "bad", text: `significantly slower, ${numbers}: reverted` };
  return { kind: "bs-neutral", text: `no significant difference, ${numbers}: reverted` };
}

// The 30 samples of each version in the order they were taken.
function drawStrip({ old, fresh }) {
  const svg = s("svg", { viewBox: `0 0 ${W} ${STRIP_H}`, role: "img",
    "aria-label": "The thirty samples of the old and the new version, in the order they were taken" });
  const all = [...old, ...fresh];
  const min = Math.min(...all), max = Math.max(...all), pad = (max - min) * 0.12 || 1;
  const x = scale([0, SAMPLES - 1], [LEFT + 8, W - RIGHT - 8]);
  const y = scale([min - pad, max + pad], [STRIP_H - 30, 20]);
  const axis = s("g", { class: "bs-axis" });
  for (const t of [min, (min + max) / 2, max]) {
    axis.append(s("line", { x1: LEFT, x2: W - RIGHT, y1: y(t), y2: y(t) }),
      s("text", { x: LEFT - 6, y: y(t) + 4, "text-anchor": "end" }, t.toFixed(1)));
  }
  for (const i of [0, 9, 19, 29]) axis.append(s("text", { x: x(i), y: STRIP_H - 12, "text-anchor": "middle" }, i + 1));
  axis.append(s("text", { x: LEFT, y: 11 }, "time per call of each sample, in the order they were taken (sample 1 to 30)"));
  svg.append(axis);
  for (const [values, cls] of [[old, "bs-old"], [fresh, "bs-new"]]) {
    const d = Array.from(values, (v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
    svg.append(s("path", { class: `bs-trace ${cls}`, d }));
    values.forEach((v, i) => svg.append(s("circle", { class: cls, cx: x(i), cy: y(v), r: 3.2 })));
  }
  return svg;
}

// The resampled ratios, with the 95% interval filled in.
function drawHistogram(result) {
  const svg = s("svg", { viewBox: `0 0 ${W} ${HIST_H}`, role: "img",
    "aria-label": `Histogram of ${RESAMPLES} resampled speedups; the 95% interval is ${result.lo.toFixed(3)} to ${result.hi.toFixed(3)}` });
  let min = Math.min(1, ...result.ratios), max = Math.max(1, ...result.ratios);
  const pad = (max - min) * 0.06 || 0.01;
  min -= pad; max += pad;
  const base = HIST_H - 30, top = 30;
  const x = scale([min, max], [LEFT, W - RIGHT]);
  const counts = new Array(BINS).fill(0);
  const width = (max - min) / BINS;
  for (const r of result.ratios) counts[Math.min(BINS - 1, Math.floor((r - min) / width))]++;
  const y = scale([0, Math.max(...counts)], [base, top]);

  const axis = s("g", { class: "bs-axis" }, s("line", { x1: LEFT, x2: W - RIGHT, y1: base, y2: base }));
  for (let i = 0; i <= 4; i++) {
    const t = min + pad + ((max - min - 2 * pad) * i) / 4;
    axis.append(s("text", { x: x(t), y: base + 16, "text-anchor": "middle" }, x3(t)));
  }
  axis.append(s("text", { x: LEFT, y: 11 }, `${RESAMPLES.toLocaleString("en-US")} resampled speedups; filled bars are the 95% interval`));
  svg.append(axis);

  counts.forEach((count, i) => {
    const centre = min + (i + 0.5) * width;
    const inside = centre >= result.lo && centre <= result.hi;
    svg.append(s("rect", { class: `bs-bar${inside ? " in" : ""}`, x: x(min + i * width) + 0.5,
      width: Math.max(1, x(min + (i + 1) * width) - x(min + i * width) - 1), y: y(count), height: base - y(count) }));
  });
  // The two labels go on opposite sides of their lines, each on the side away
  // from the other line, unless that would push one out of the drawing.
  const side = (at, wantLeft, width) => {
    const left = wantLeft ? at - width >= 2 : at + width > W - 2;
    return { x: at + (left ? -5 : 5), "text-anchor": left ? "end" : "start" };
  };
  const above = result.point >= 1;
  svg.append(
    s("line", { class: "bs-one", x1: x(1), x2: x(1), y1: top - 8, y2: base }),
    s("text", { class: "bs-tag", y: top - 1, ...side(x(1), above, ONE_TAG_WIDTH) }, "1.0: no change"),
    s("line", { class: "bs-point", x1: x(result.point), x2: x(result.point), y1: top + 8, y2: base }),
    s("text", { class: "bs-tag hot", y: top + 16, ...side(x(result.point), !above, POINT_TAG_WIDTH) }, `measured ${x3(result.point)}`),
  );
  return svg;
}

export default async function mount(el) {
  let speedup = Number(el.dataset.speedup) || 1.03;
  let noisePercent = Number(el.dataset.noise) || 2;
  let seed = 1;
  let busy = false;

  const strip = h("div", { class: "w-stage" });
  const histogram = h("div", { class: "w-stage" });
  const verdict = h("p", { class: "w-verdict", "aria-live": "polite" });
  const shares = {
    [FASTER]: stat("significantly faster: kept", "good"),
    [SAME]: stat("no significant difference"),
    [SLOWER]: stat("significantly slower", "bad"),
  };
  const repeatNote = h("p", { class: "w-note" });

  function clearShares() {
    for (const tile of Object.values(shares)) tile.set("–");
    repeatNote.textContent = `Press “Repeat ${REPEATS} times” to see how often each verdict comes out at these settings.`;
  }

  function run() {
    const random = rng(seed * 7919 + 17);
    const sample = measure(random, speedup, noisePercent / 100);
    const result = analyse(sample, random);
    strip.replaceChildren(drawStrip(sample));
    histogram.replaceChildren(drawHistogram(result));
    const { kind, text } = verdictText(result);
    verdict.className = `w-verdict ${kind}`;
    verdict.textContent = text;
  }

  async function repeat() {
    if (busy) return;
    busy = true;
    again.disabled = many.disabled = true;
    const tally = { [FASTER]: 0, [SAME]: 0, [SLOWER]: 0 };
    const settings = [speedup, noisePercent];
    for (let done = 0; done < REPEATS; done++) {
      const random = rng((seed + done + 1) * 104729 + 3);
      tally[analyse(measure(random, speedup, noisePercent / 100), random).verdict]++;
      if ((done + 1) % REPEATS_PER_FRAME === 0) {
        for (const [name, tile] of Object.entries(shares)) tile.set(`${((tally[name] / (done + 1)) * 100).toFixed(1)}%`);
        repeatNote.textContent = `${done + 1} of ${REPEATS} measurements…`;
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (settings[0] !== speedup || settings[1] !== noisePercent) break;   // a slider moved: stop counting
      }
    }
    busy = false;
    again.disabled = many.disabled = false;
    if (settings[0] !== speedup || settings[1] !== noisePercent) { clearShares(); return; }
    repeatNote.textContent = `Share of ${REPEATS} separate measurements, each with a true speedup of ${speedup.toFixed(2)}x and ${noisePercent}% noise.`
      + (speedup === 1 ? " The edit changed nothing, and the rule still kept it this often." : "");
  }

  const again = button("Measure again", () => { seed += 1; run(); }, "primary");
  const many = button(`Repeat ${REPEATS} times`, repeat);
  const changed = () => { clearShares(); run(); };

  el.append(
    h("div", { class: "w-controls" },
      range({ label: "True speedup", min: 0.9, max: 1.3, step: 0.01, value: speedup,
        format: (v) => `${v.toFixed(2)}x`, onInput: (v) => { speedup = v; changed(); } }),
      range({ label: "Noise", min: 0.5, max: 10, step: 0.5, value: noisePercent,
        format: (v) => `${v}%`, onInput: (v) => { noisePercent = v; changed(); } }),
      h("span", { class: "w-spacer" }), again, many),
    h("div", { class: "bs-plots" }, strip, histogram),
    h("div", { class: "w-legend" },
      h("span", {}, h("i", { class: "cc" }), "old version"), h("span", {}, h("i", { class: "hot" }), "new version"),
      h("span", {}, `${SAMPLES} samples of each, taken alternately`)),
    verdict,
    h("div", { class: "w-readout" }, Object.values(shares)),
    repeatNote,
  );
  clearShares();
  run();
}
