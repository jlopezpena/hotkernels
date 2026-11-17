// The headline chart: how many times faster each kernel is at 1,048,576
// elements, per NumPy call, against NumPy or against the same loop compiled
// with clang -O3. Bars are on a logarithmic scale that starts at 1x. Where a
// kernel is timed in two cases, the solid bar ends at the lower figure and a
// lighter extension reaches the higher one.
//
// Data: data/headline.json, read from the README's results table by the build.
// Attributes: data-against="numpy" (default) or "c" picks the initial comparison.

import { h, loadData, seg, fmtX } from "./util.js";

// Candidate ends of the scale; the first one at or above the largest value is used.
const TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
const MODES = [
  { value: "numpy", label: "vs NumPy", field: "vs_numpy" },
  { value: "c", label: "vs clang -O3", field: "vs_c" },
];

export const css = `
.sp-chart { display: grid; grid-template-columns: minmax(7rem, max-content) minmax(8rem, 1fr) max-content; column-gap: .9rem; row-gap: .3rem; align-items: center; }
.sp-row { display: contents; }
.sp-label { font-size: .8rem; line-height: 1.3; max-width: 17rem; }
.sp-label code { font-size: .78rem; background: none; border: 0; padding: 0; white-space: normal; }
.sp-label span { color: var(--muted); margin-left: .45rem; white-space: nowrap; }
.sp-track { position: relative; height: 1.05rem; }
.sp-axis { height: 1.2rem; }
.sp-tick { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule); transition: left .35s; }
.sp-axis .sp-tick { background: none; width: auto; transform: translateX(-50%); font-size: .7rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.sp-axis .sp-tick:first-child { transform: none; }
.sp-bar, .sp-ext { position: absolute; top: 2px; bottom: 2px; background: var(--hot); transition: width .35s, left .35s, opacity .2s; }
.sp-bar { left: 0; border-radius: 3px 0 0 3px; min-width: 2px; }
.sp-bar.whole { border-radius: 3px; }
.sp-ext { opacity: .38; border-radius: 0 3px 3px 0; }
.sp-value { font-size: .8rem; font-variant-numeric: tabular-nums; white-space: nowrap; text-align: right; min-width: 6.5rem; }
.sp-value.words { color: var(--muted); }
@media (max-width: 40rem) {
  .sp-chart { grid-template-columns: minmax(0, 1fr) max-content; row-gap: .1rem; }
  .sp-label { grid-column: 1 / -1; max-width: none; margin-top: .45rem; }
  .sp-axis-label { display: none; }
  .sp-value { min-width: 0; }
}
`;

// "17.3x – 41.7x", or "29.0x" when the two ends are the same.
function rangeText(r) {
  return r.lo === r.hi ? fmtX(r.lo) : `${fmtX(r.lo)} – ${fmtX(r.hi)}`;
}

export default async function mount(el) {
  const rows = await loadData("headline");
  let mode = MODES.find((m) => m.value === el.dataset.against) || MODES[0];

  const axis = h("div", { class: "sp-track sp-axis", "aria-hidden": "true" });
  const chart = h("div", { class: "sp-chart" },
    h("div", { class: "sp-row" }, h("div", { class: "sp-label sp-axis-label" }), axis, h("div", { class: "sp-value" })));

  const parts = rows.map((row) => {
    const grid = h("div", { class: "sp-grid" });
    const bar = h("div", { class: "sp-bar" });
    const ext = h("div", { class: "sp-ext" });
    const value = h("div", { class: "sp-value" });
    chart.appendChild(h("div", { class: "sp-row" },
      h("div", { class: "sp-label" }, h("code", {}, row.call), h("span", {}, row.elements)),
      h("div", { class: "sp-track" }, grid, bar, ext),
      value));
    return { row, grid, bar, ext, value };
  });

  function update() {
    const top = Math.max(...rows.map((r) => (r[mode.field] ? r[mode.field].hi : 1)));
    const end = TICKS.find((t) => t >= top) || top;
    const ticks = TICKS.filter((t) => t <= end);
    // Position on the logarithmic scale, as a percentage of the track.
    const at = (x) => (Math.log(Math.max(x, 1)) / Math.log(end)) * 100;

    axis.replaceChildren(...ticks.map((t) => h("span", { class: "sp-tick", style: `left:${at(t)}%` }, `${t}x`)));
    for (const { row, grid, bar, ext, value } of parts) {
      grid.replaceChildren(...ticks.map((t) => h("span", { class: "sp-tick", style: `left:${at(t)}%` })));
      const r = row[mode.field];
      if (!r) {
        bar.style.opacity = ext.style.opacity = "0";
        bar.style.width = ext.style.width = "0";
        value.textContent = row.vs_c_text;
        value.classList.add("words");
        continue;
      }
      const ranged = r.hi !== r.lo;
      bar.style.opacity = "";
      bar.style.width = `${at(r.lo)}%`;
      bar.classList.toggle("whole", !ranged);
      ext.style.opacity = ranged ? "" : "0";
      ext.style.left = `${at(r.lo)}%`;
      ext.style.width = `${at(r.hi) - at(r.lo)}%`;
      value.textContent = rangeText(r);
      value.classList.remove("words");
    }
    chart.setAttribute("aria-label", `Speedup of each kernel ${mode.label}, at 1,048,576 elements`);
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Times faster"),
      seg(MODES, mode.value, (v) => { mode = MODES.find((m) => m.value === v); update(); })),
    h("div", { class: "w-stage" }, chart),
    h("p", { class: "w-note" }, "1,048,576 elements, Apple M1 Pro. A lighter extension is the more favourable of two timed cases."),
  );
  update();
}
