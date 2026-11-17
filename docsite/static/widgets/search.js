// A branchless binary search run for several needles in step, as the
// searchsorted kernel does it (kernels/searchsorted_f64/arm64/kernel.s).
//
// Every needle keeps an index `base`. For h = 16, 8, 4, 2, 1 each one probes
// a[base + h - 1] and adds h to its base when that element is below its key;
// a last probe of a[base] adds one more. The step size depends only on the
// array's length, so all searches take the same number of steps and the same
// h at each one. The figure shows four needles advancing together over one
// sorted array of 32 values, and underneath what that does to the memory
// loads: one search at a time has to wait for each load before it knows the
// address of the next, while searches in step have one load each in flight
// at once.

import { h, button, stat, rng, reducedMotion } from "./util.js";

// Length of the sorted array. A power of two, so the first step always starts from base 0.
const N = 32;
// Searches run together.
const NEEDLES = 4;
// One colour per needle, from the stylesheet.
const COLOURS = ["--hot", "--np", "--good", "--cc"];
// Probes per search: one per halving, then the final probe of a[base].
const STEPS = Math.log2(N) + 1;
// Pause between showing the probes of a step and moving the bases, and between steps when playing.
const MOVE_DELAY_MS = 650;
const PLAY_INTERVAL_MS = 1500;
// An index label is drawn under every this-many cells.
const INDEX_EVERY = 4;

export const css = `
.sr-rows { display: flex; flex-direction: column; gap: .45rem; min-width: max-content; }
.sr-row { display: flex; align-items: center; gap: .7rem; }
.sr-key { flex: none; width: 8.2rem; font-family: var(--mono); font-size: .78rem; font-weight: 700; color: var(--c); line-height: 1.3; }
.sr-key small { display: block; font-weight: 500; color: var(--muted); font-size: .68rem; min-height: 1.3em; }
.sr-row .cell.small { width: 1.2rem; }
.sr-row .cell.probe { background: var(--c); border-color: var(--c); color: var(--bg-raised); font-weight: 700; opacity: 1; }
.sr-row .cell.base { box-shadow: inset 3px 0 0 var(--c); }
.sr-row .cell.answer { outline: 2px solid var(--c); outline-offset: 1px; font-weight: 700; opacity: 1; }
.sr-index { display: flex; gap: 3px; margin-left: 8.9rem; font-family: var(--mono); font-size: .62rem; color: var(--muted); }
.sr-index span { width: 1.2rem; flex: none; text-align: center; }
.w .sr-step { font-size: .86rem; margin: 0 0 .9rem; min-height: 2.6em; }
.sr-step b { color: var(--hot); }
.sr-timelines { display: grid; gap: 1rem 2rem; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); margin-top: 1.2rem; }
.sr-tl h4 { font-size: .74rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .5rem; }
.w .sr-tl p { font-size: .78rem; color: var(--muted); margin: .45rem 0 0; }
.sr-seq { display: flex; flex-wrap: wrap; gap: 2px; }
.sr-par { display: flex; gap: 6px; }
.sr-par > div { display: flex; flex-direction: column; gap: 2px; }
.sr-box { width: .8rem; height: .8rem; border-radius: 2px; background: var(--c); opacity: .3; transition: opacity .2s, outline-color .2s; outline: 2px solid transparent; outline-offset: 0; }
.sr-box.done { opacity: 1; }
.sr-box.now { opacity: 1; outline-color: var(--ink); }
.sr-results { display: flex; flex-wrap: wrap; gap: .4rem .6rem; margin-top: .9rem; font-family: var(--mono); font-size: .78rem; }
.sr-results span { padding: .2rem .55rem; border-radius: 6px; border: 1px solid var(--rule-strong); color: var(--c); }
.sr-results span b { color: var(--good); }
.sr-results span b.bad { color: var(--bad); }
`;

// First index i with a[i] >= key: np.searchsorted(a, key, side="left").
function lowerBound(a, key) {
  let i = 0;
  while (i < a.length && a[i] < key) i++;
  return i;
}

// A sorted array of distinct two-digit values.
function makeArray(random) {
  const a = [];
  let v = 1 + Math.floor(random() * 3);
  for (let i = 0; i < N; i++) {
    a.push(v);
    v += 1 + Math.floor(random() * 3);
  }
  return a;
}

// Keys to search for: some are elements of the array, some fall between or outside them.
function makeKeys(a, random) {
  const keys = [];
  for (let j = 0; j < NEEDLES; j++) {
    const exact = random() < 0.35;
    keys.push(exact ? a[Math.floor(random() * N)] : a[0] - 1 + Math.floor(random() * (a[N - 1] - a[0] + 3)));
  }
  return keys;
}

export default function mount(el) {
  let seed = 7;
  let a, keys, bases, step, probes, moves, pending, timer, playing;

  const stepText = h("p", { class: "sr-step", "aria-live": "polite" });
  const rows = h("div", { class: "sr-rows" });
  const seq = h("div", { class: "sr-seq" });
  const par = h("div", { class: "sr-par" });
  const results = h("div", { class: "sr-results" });
  const stepButton = button("Step", () => { stop(); advance(); }, "primary");
  const playButton = button("Play", () => (playing ? stop() : play()));

  function reset(newData) {
    stop();
    clearTimeout(timer);
    if (newData) {
      const random = rng(seed++);
      a = a || makeArray(random);
      keys = makeKeys(a, random);
    }
    bases = keys.map(() => 0);
    step = 0;
    probes = null;
    moves = null;
    pending = false;
    render();
  }

  // The step size of step k: 16, 8, 4, 2, 1, and 1 again for the final probe of a[base].
  function stepSize(k) {
    return k === STEPS - 1 ? 1 : N >> (k + 1);
  }

  function advance() {
    if (step >= STEPS || pending) return;
    const last = step === STEPS - 1;
    const size = stepSize(step);
    probes = bases.map((b) => (last ? b : b + size - 1));
    moves = probes.map((p, j) => a[p] < keys[j]);
    pending = true;
    render();
    const finish = () => {
      bases = bases.map((b, j) => b + (moves[j] ? size : 0));
      step++;
      pending = false;
      render();
      if (step >= STEPS) stop();
    };
    if (reducedMotion) finish();
    else timer = setTimeout(finish, MOVE_DELAY_MS);
  }

  function play() {
    if (step >= STEPS) reset(false);
    playing = setInterval(advance, PLAY_INTERVAL_MS);
    playButton.textContent = "Pause";
    advance();
  }

  function stop() {
    clearInterval(playing);
    playing = null;
    playButton.textContent = "Play";
  }

  function describe() {
    if (step === 0 && !pending) {
      return h("span", {}, "Four searches over the same sorted array, each starting with ", h("code", {}, "base = 0"),
        ". Press Step: every search makes its next probe at once.");
    }
    const k = pending ? step : step - 1;
    const size = stepSize(k);
    const last = k === STEPS - 1;
    const what = last
      ? ["every needle probes ", h("code", {}, "a[base]"), " and adds 1 to its base if that element is below its key."]
      : ["every needle probes ", h("code", {}, `a[base + ${size - 1}]`), ` (h = ${size}) and adds ${size} to its base if that element is below its key.`];
    const done = step >= STEPS && !pending ? [" Done: each base is now the answer."] : [];
    return h("span", {}, h("b", {}, `Step ${k + 1} of ${STEPS}: `), ...what, " The step size is the same for all of them.", ...done);
  }

  function renderRows() {
    rows.replaceChildren();
    // Elements still in play for a search: [base, base + len), where len halves at each step.
    const len = step >= STEPS ? 0 : N >> Math.min(step, STEPS - 1);
    const finished = step >= STEPS && !pending;
    keys.forEach((key, j) => {
      const cells = h("div", { class: "cells nowrap" });
      for (let i = 0; i < N; i++) {
        const classes = ["cell", "small"];
        const inPlay = i >= bases[j] && i < bases[j] + Math.max(len, 1);
        if (finished || !inPlay) classes.push("dim");
        if (i === bases[j] && !finished) classes.push("base");
        if (probes && probes[j] === i && (pending || !finished)) classes.push("probe");
        if (finished && i === bases[j]) classes.push("answer");
        cells.appendChild(h("span", { class: classes.join(" ") }, a[i]));
      }
      let message = "";
      if (probes && (pending || step > 0)) {
        const value = a[probes[j]];
        const size = stepSize(pending ? step : step - 1);
        message = moves[j] ? `${value} < ${key}: base += ${size}` : `${value} ≥ ${key}: base stays`;
      }
      const baseText = bases[j] >= N ? `base = ${N} (end)` : `base = ${bases[j]}`;
      rows.appendChild(h("div", { class: "sr-row", style: `--c: var(${COLOURS[j]})` },
        h("div", { class: "sr-key" }, `key ${key}`, h("small", {}, baseText), h("small", {}, message)),
        cells));
    });
    const index = h("div", { class: "sr-index", "aria-hidden": "true" });
    for (let i = 0; i < N; i++) index.appendChild(h("span", {}, i % INDEX_EVERY === 0 ? i : ""));
    rows.appendChild(index);
  }

  // State of the box for one probe: already made, being made now, or still to come.
  function boxClass(k) {
    const current = pending ? step : -1;
    if (k === current) return "sr-box now";
    return k < step ? "sr-box done" : "sr-box";
  }

  function renderTimelines() {
    seq.replaceChildren();
    par.replaceChildren();
    for (let j = 0; j < NEEDLES; j++) {
      for (let k = 0; k < STEPS; k++) seq.appendChild(h("span", { class: boxClass(k), style: `--c: var(${COLOURS[j]})` }));
    }
    for (let k = 0; k < STEPS; k++) {
      const column = h("div", {});
      for (let j = 0; j < NEEDLES; j++) column.appendChild(h("span", { class: boxClass(k), style: `--c: var(${COLOURS[j]})` }));
      par.appendChild(column);
    }
  }

  function renderResults() {
    results.replaceChildren();
    if (step < STEPS || pending) return;
    keys.forEach((key, j) => {
      const expected = lowerBound(a, key);
      const ok = bases[j] === expected;
      results.appendChild(h("span", { style: `--c: var(${COLOURS[j]})` },
        `key ${key} → index ${bases[j]} `, h("b", { class: ok ? "" : "bad" }, ok ? "✓" : `✗ (expected ${expected})`)));
    });
  }

  function render() {
    stepText.replaceChildren(describe());
    renderRows();
    renderTimelines();
    renderResults();
    stepButton.disabled = step >= STEPS || pending;
  }

  const stepsStat = stat("probes per search, whatever the key");
  const oneStat = stat("load in flight, one needle at a time", "np");
  const manyStat = stat("loads in flight, all needles in step", "hot");
  stepsStat.set(String(STEPS));
  oneStat.set("1");
  manyStat.set(String(NEEDLES));

  el.append(
    h("div", { class: "w-controls" },
      stepButton, playButton, button("Reset", () => reset(false)), button("New needles", () => reset(true))),
    stepText,
    h("div", { class: "w-stage" }, rows),
    results,
    h("div", { class: "sr-timelines" },
      h("div", { class: "sr-tl" }, h("h4", {}, "One needle at a time"), seq,
        h("p", {}, `${NEEDLES * STEPS} loads end to end. Each probe's address depends on the compare before it, so nothing can start early.`)),
      h("div", { class: "sr-tl" }, h("h4", {}, "All needles in step"), par,
        h("p", {}, `${STEPS} rounds of ${NEEDLES} loads. The loads of a round do not depend on each other, so their cache misses are in flight together.`))),
    h("div", { class: "w-readout" }, stepsStat, oneStat, manyStat),
    h("p", { class: "w-note" }, "The kernel advances 16, 64 or 128 needles at once depending on the array's length."),
  );

  reset(true);
}
