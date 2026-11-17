// Helpers shared by the interactive figures: building elements, the standard
// controls, number formatting, and loading the data files the build writes.

const SVG_NS = "http://www.w3.org/2000/svg";

function fill(node, attrs, children) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.setAttribute("class", value);
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.appendChild(typeof child === "object" ? child : document.createTextNode(String(child)));
  }
  return node;
}

// h("div", {class: "x", onclick: fn}, child, "text", [more])
export function h(tag, attrs, ...children) {
  return fill(document.createElement(tag), attrs, children);
}

// The same for SVG elements.
export function s(tag, attrs, ...children) {
  return fill(document.createElementNS(SVG_NS, tag), attrs, children);
}

// Fetches data/<name>.json, which the build writes from the repository.
export async function loadData(name) {
  const base = document.body.dataset.root || "";
  const response = await fetch(`${base}data/${name}.json?v=${document.body.dataset.version || ""}`);
  if (!response.ok) throw new Error(`data/${name}.json: ${response.status}`);
  return response.json();
}

// A segmented control. options: [{value, label}], onChange(value).
export function seg(options, value, onChange) {
  const el = h("div", { class: "w-seg", role: "group" });
  const set = (v, notify) => {
    for (const b of el.children) b.classList.toggle("on", b.dataset.value === String(v));
    if (notify) onChange(v);
  };
  for (const option of options) {
    el.appendChild(h("button", { type: "button", "data-value": option.value, onclick: () => set(option.value, true) }, option.label));
  }
  set(value, false);
  el.set = (v) => set(v, false);
  return el;
}

// A slider with its value shown. onInput(number); format(number) -> text.
export function range({ label, min, max, step = 1, value, format = String, onInput }) {
  const out = h("output", {}, format(value));
  const input = h("input", {
    type: "range", min, max, step, value, "aria-label": label,
    oninput: () => { const v = Number(input.value); out.textContent = format(v); onInput(v); },
  });
  const el = h("label", { class: "w-range" }, h("span", { class: "w-label" }, label), input, out);
  el.set = (v) => { input.value = v; out.textContent = format(v); };
  return el;
}

export function button(label, onClick, cls = "") {
  return h("button", { type: "button", class: `w-btn ${cls}`.trim(), onclick: onClick }, label);
}

// A number with a caption, for the row under a figure. kind: "", "hot", "np", "good", "bad".
export function stat(label, kind = "") {
  const value = h("span", { class: "w-stat-v" }, "–");
  const el = h("div", { class: `w-stat ${kind}`.trim() }, value, h("span", { class: "w-stat-l" }, label));
  el.set = (text) => { value.textContent = text; };
  return el;
}

// Calls back whenever the reader switches between light and dark, for
// figures that draw with colours read from the stylesheet.
export function onTheme(callback) {
  document.addEventListener("themechange", callback);
}

// A colour from the stylesheet, e.g. cssColor("--hot").
export function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// A small seeded random generator (mulberry32), so a figure shows the same
// thing on every load until the reader asks for another draw.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// "23.4x": a speedup, with the precision the judge prints.
export function fmtX(x) {
  if (x === null || x === undefined) return "–";
  return `${x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)}x`;
}

// A duration in nanoseconds, in the largest unit that keeps it above 1.
export function fmtNs(ns) {
  if (ns < 1e3) return `${ns < 10 ? ns.toFixed(1) : ns.toFixed(0)} ns`;
  if (ns < 1e6) return `${(ns / 1e3).toPrecision(3)} µs`;
  return `${(ns / 1e6).toPrecision(3)} ms`;
}

// 1048576 -> "1,048,576"
export function fmtN(n) {
  return n.toLocaleString("en-US");
}

// Maps a value from one interval onto another; with log = true the input scale is logarithmic.
export function scale([d0, d1], [r0, r1], log = false) {
  const f = log ? Math.log : (v) => v;
  return (v) => r0 + ((f(v) - f(d0)) / (f(d1) - f(d0))) * (r1 - r0);
}
