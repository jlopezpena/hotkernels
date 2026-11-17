// What every page runs: the theme switch, the table-of-contents highlight,
// copy buttons on code, and the loaders for diagrams, maths and interactive
// figures. Each loader only fetches its library when the page has something
// for it to do.

const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";
const KATEX_URL = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.mjs";
const THEME_KEY = "theme";
const THEME_EVENT = "themechange";

const root = document.documentElement;
const version = document.body.dataset.version || "";

// ------------------------------------------------------------------ theme

function setTheme(theme) {
  root.dataset.theme = theme;
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* private window: the choice lasts for the page */ }
  document.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: theme }));
}

document.querySelector(".theme-toggle")?.addEventListener("click", () => {
  setTheme(root.dataset.theme === "dark" ? "light" : "dark");
});

// --------------------------------------------------------------- diagrams

// Reads a colour from the stylesheet, so diagrams follow the theme.
function color(name) {
  return getComputedStyle(root).getPropertyValue(name).trim();
}

function mermaidConfig() {
  return {
    startOnLoad: false,
    theme: "base",
    fontFamily: getComputedStyle(root).getPropertyValue("--sans"),
    themeVariables: {
      fontSize: "14px",
      background: color("--bg-raised"),
      primaryColor: color("--panel"),
      primaryTextColor: color("--ink"),
      primaryBorderColor: color("--rule-strong"),
      secondaryColor: color("--bg-raised"),
      secondaryTextColor: color("--ink"),
      secondaryBorderColor: color("--rule-strong"),
      tertiaryColor: color("--bg-raised"),
      tertiaryTextColor: color("--ink"),
      tertiaryBorderColor: color("--rule"),
      lineColor: color("--muted"),
      textColor: color("--ink"),
      mainBkg: color("--panel"),
      nodeBorder: color("--rule-strong"),
      clusterBkg: color("--bg"),
      clusterBorder: color("--rule"),
      titleColor: color("--muted"),
      edgeLabelBackground: color("--bg-raised"),
      actorBkg: color("--panel"),
      actorBorder: color("--rule-strong"),
      actorTextColor: color("--ink"),
      actorLineColor: color("--rule-strong"),
      signalColor: color("--muted"),
      signalTextColor: color("--ink"),
      labelBoxBkgColor: color("--panel"),
      labelBoxBorderColor: color("--rule-strong"),
      labelTextColor: color("--ink"),
      loopTextColor: color("--muted"),
      noteBkgColor: color("--hot-soft"),
      noteBorderColor: color("--hot"),
      noteTextColor: color("--ink"),
      activationBkgColor: color("--hot-soft"),
      activationBorderColor: color("--hot"),
      sequenceNumberColor: color("--bg"),
    },
    flowchart: { curve: "basis", padding: 14, htmlLabels: true, useMaxWidth: true },
    sequence: { useMaxWidth: true, mirrorActors: false, messageMargin: 34 },
  };
}

async function drawDiagrams() {
  const blocks = [...document.querySelectorAll("pre.mermaid")];
  if (!blocks.length) return;
  const { default: mermaid } = await import(MERMAID_URL);
  for (const block of blocks) block.dataset.source = block.textContent;
  const draw = async () => {
    for (const block of blocks) {
      block.removeAttribute("data-processed");
      block.textContent = block.dataset.source;
    }
    mermaid.initialize(mermaidConfig());
    await mermaid.run({ nodes: blocks });
  };
  await draw();
  document.addEventListener(THEME_EVENT, draw);
}

// ------------------------------------------------------------------ maths

async function typesetMaths() {
  const spans = document.querySelectorAll(".math");
  if (!spans.length) return;
  const { default: katex } = await import(KATEX_URL);
  for (const span of spans) {
    katex.render(span.textContent, span, { displayMode: span.classList.contains("block"), throwOnError: false });
  }
}

// ---------------------------------------------------------------- widgets

const injectedStyles = new Set();

// A widget is a module in widgets/ whose default export mounts it into an
// element. It may also export `css`, which is added to the page once.
async function mountWidget(el) {
  const name = el.dataset.widget;
  el.classList.add("w");
  el.innerHTML = '<p class="w-loading">Loading figure…</p>';
  try {
    const module = await import(`./widgets/${name}.js?v=${version}`);
    if (module.css && !injectedStyles.has(name)) {
      injectedStyles.add(name);
      document.head.appendChild(Object.assign(document.createElement("style"), { textContent: module.css }));
    }
    el.innerHTML = "";
    await module.default(el);
  } catch (error) {
    console.error(`widget ${name}`, error);
    el.innerHTML = '<p class="w-error">This figure could not be loaded.</p>';
  }
}

// ------------------------------------------------- small page conveniences

function addCopyButtons() {
  for (const pre of document.querySelectorAll("pre.code")) {
    const button = Object.assign(document.createElement("button"), { className: "copy", type: "button", textContent: "Copy" });
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.querySelector("code").textContent);
        button.textContent = "Copied";
      } catch (e) {
        button.textContent = "Select and copy";
      }
      setTimeout(() => { button.textContent = "Copy"; }, 1600);
    });
    pre.appendChild(button);
  }
}

// Marks the table-of-contents entry of the section being read.
function followHeadings() {
  const links = new Map([...document.querySelectorAll(".toc a")].map((a) => [a.getAttribute("href").slice(1), a]));
  if (!links.size) return;
  const headings = [...links.keys()].map((id) => document.getElementById(id)).filter(Boolean);
  const mark = () => {
    const line = window.scrollY + window.innerHeight * 0.25;
    let current = headings[0];
    for (const heading of headings) if (heading.offsetTop <= line) current = heading;
    for (const [id, a] of links) a.classList.toggle("active", current && id === current.id);
  };
  window.addEventListener("scroll", mark, { passive: true });
  mark();
}

addCopyButtons();
followHeadings();
document.querySelectorAll("[data-widget]").forEach(mountWidget);
drawDiagrams().catch((error) => console.error("diagrams", error));
typesetMaths().catch((error) => console.error("maths", error));
