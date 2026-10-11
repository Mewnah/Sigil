#!/usr/bin/env node
/**
 * Documentation checks (dependency free):
 *
 *   node docs/check-docs.mjs              relative links, anchors and table shapes
 *   node docs/check-docs.mjs --no-todo    also fail while a `TODO(P7-numbers)` placeholder is left
 *
 * What it checks, over README.md, ARCHITECTURE.md, the current part of CHANGELOG.md (everything above the
 * first released version) and every Markdown file under docs/ (including manual/, adr/ and benchmarks/):
 *
 * - every relative link resolves to a file in the repository, and every `#anchor` into a Markdown file
 *   matches a heading there (the in-app Help reader and GitHub both slug headings the github-slugger way:
 *   lower case, punctuation removed, spaces to hyphens, repeated headings numbered -1, -2, ...);
 * - every GFM table row has as many cells as its header (a `|` inside a code span splits a cell and has to
 *   be written `\|`);
 * - with --no-todo, no `TODO(P7-numbers)` placeholder remains outside a code span (the P7 documentation
 *   pass leaves one wherever a measured number or a chosen default belongs; the final integrate fills them);
 * - facts that must agree with the code (the rules at the end of this file, each with a self-test that runs on
 *   every invocation): the speech-to-text UI labels against the English strings, the CPU Streamer's thread rule
 *   against `ort_session.rs`, models no routing row lists against `stt-routing.json`, the headless commands
 *   against `cli/mod.rs`, and the pieces of documentation a review asked for.
 *
 * Links inside fenced code blocks and code spans are ignored. Exit code 1 when anything is found.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const noTodo = process.argv.includes("--no-todo");

function walk(dir) {
  const out = [];
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) out.push(...walk(p));
    else if (ent.name.endsWith(".md")) out.push(p);
  }
  return out;
}

const files = [
  ...["README.md", "ARCHITECTURE.md", "CHANGELOG.md"].map((f) => path.join(root, f)),
  ...walk(path.join(root, "docs")),
]
  .filter((f) => fs.existsSync(f))
  .map((f) => path.relative(root, f).split(path.sep).join("/"));

const texts = new Map();
function read(rel) {
  if (!texts.has(rel)) {
    let t = fs.readFileSync(path.join(root, rel), "utf8").replace(/\r\n/g, "\n");
    if (rel === "CHANGELOG.md") {
      // Entries of released versions name documents that were later merged or renamed (see its intro).
      const cut = t.search(/^## \[\d/m);
      if (cut > 0) t = t.slice(0, cut);
    }
    texts.set(rel, t);
  }
  return texts.get(rel);
}

/** github-slugger for the headings of these documents (letters, digits, `_`, `-` and spaces survive). */
function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, "")
    .replace(/ /g, "-");
}

function headingText(raw) {
  return raw
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/[`*]/g, "")
    .replace(/<[^>]+>/g, "")
    .trim();
}

const anchorCache = new Map();
function anchors(rel) {
  if (anchorCache.has(rel)) return anchorCache.get(rel);
  const set = new Set();
  const seen = new Map();
  let fence = false;
  for (const line of read(rel).split("\n")) {
    if (/^\s*```/.test(line)) fence = !fence;
    if (fence) continue;
    const m = /^#{1,6}\s+(.*?)\s*#*\s*$/.exec(line);
    if (!m) continue;
    const base = slugify(headingText(m[1]));
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    set.add(n === 0 ? base : `${base}-${n}`);
  }
  anchorCache.set(rel, set);
  return set;
}

function tableCells(line) {
  let t = line.trim();
  if (t.startsWith("|")) t = t.slice(1);
  if (t.endsWith("|") && !t.endsWith("\\|")) t = t.slice(0, -1);
  let n = 1;
  for (let i = 0; i < t.length; i++) {
    if (t[i] === "\\") i++;
    else if (t[i] === "|") n++;
  }
  return n;
}

const problems = [];
let todos = 0;

for (const rel of files) {
  const lines = read(rel).split("\n");
  let fence = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      fence = !fence;
      continue;
    }
    if (fence) continue;
    const prose = line.replace(/`[^`]*`/g, "");

    // Links.
    const re = /\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
    let m;
    while ((m = re.exec(prose))) {
      const href = m[1];
      if (/^(https?:|mailto:|data:)/i.test(href)) continue;
      const [filePart, anchor] = href.split("#");
      const target = filePart === "" ? rel : path.posix.normalize(path.posix.join(path.posix.dirname(rel), decodeURI(filePart)));
      if (filePart !== "" && !fs.existsSync(path.join(root, target))) {
        problems.push(`${rel}:${i + 1}: missing file ${href}`);
        continue;
      }
      if (anchor && target.endsWith(".md") && files.includes(target) && !anchors(target).has(decodeURIComponent(anchor))) {
        problems.push(`${rel}:${i + 1}: missing anchor ${href}`);
      }
    }

    // Placeholders.
    if (/TODO\(P7-numbers\)/.test(prose)) {
      todos += (prose.match(/TODO\(P7-numbers\)/g) ?? []).length;
      if (noTodo) problems.push(`${rel}:${i + 1}: placeholder left: ${prose.trim().slice(0, 110)}`);
    }
  }

  // Tables: a header row followed by a separator row.
  fence = false;
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*```/.test(lines[i])) fence = !fence;
    if (fence) continue;
    if (/^\s*\|/.test(lines[i]) && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      const head = tableCells(lines[i]);
      if (tableCells(lines[i + 1]) !== head) problems.push(`${rel}:${i + 2}: table separator has ${tableCells(lines[i + 1])} cells, the header ${head}`);
      let j = i + 2;
      while (j < lines.length && /^\s*\|/.test(lines[j])) {
        const c = tableCells(lines[j]);
        if (c !== head) problems.push(`${rel}:${j + 1}: table row has ${c} cells, the header ${head}: ${lines[j].slice(0, 80)}`);
        j++;
      }
      i = j - 1;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Facts: statements the P7 review found wrong against the code. Every rule reads the number or label it checks
// from the source (or the English strings), so the docs fail when either side moves, and every rule has a
// self-test (the sentence the review caught must be flagged, the corrected one must pass), so a rule cannot go
// silently dead.
// ---------------------------------------------------------------------------------------------------------------

function readSource(rel) {
  try {
    return fs.readFileSync(path.join(root, rel), "utf8").replace(/\r\n/g, "\n");
  } catch {
    return null;
  }
}

/**
 * Logical units of a Markdown text: a table row, a bullet or numbered item with its continuation lines, a heading
 * or a paragraph (hard-wrapped lines joined by a space). A table row carries `ctx`, the prose unit above the table
 * under the same heading, because a table's lead-in sentence is where its claim is made.
 */
function unitsOf(text) {
  const out = [];
  let cur = null;
  let fence = false;
  let lastProse = "";
  const flush = () => {
    if (!cur) return;
    out.push(cur);
    if (!cur.table && !cur.heading) lastProse = cur.text;
    cur = null;
  };
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {
      flush();
      fence = !fence;
      continue;
    }
    if (fence) continue;
    if (line.trim() === "") {
      flush();
      continue;
    }
    const heading = /^#{1,6}\s/.test(line);
    const table = /^\s*\|/.test(line);
    const item = /^\s*(?:[-*]|\d+\.)\s/.test(line);
    if (heading || table || item || !cur || cur.table) {
      flush();
      if (heading) lastProse = "";
      cur = { line: i + 1, text: line.trim(), table, heading, ctx: table ? lastProse : "" };
      if (heading || table) flush();
    } else {
      cur.text += " " + line.trim();
    }
  }
  flush();
  return out;
}

/** What the rules compare the docs with, read from the code once. `null` entries are reported as problems. */
function loadFacts() {
  const facts = { errors: [] };
  const fail = (m) => facts.errors.push(m);

  const ort = readSource("src-tauri/src/services/neural_audio/ort_session.rs");
  const num = (name) => {
    const m = ort && new RegExp(`pub const ${name}: (?:usize|u32) = (\\d+);`).exec(ort);
    return m ? Number(m[1]) : null;
  };
  facts.threads = { min: num("STREAMER_MIN_THREADS"), max: num("STREAMER_MAX_THREADS"), spare: num("STREAMER_SPARE_CORES") };
  if (Object.values(facts.threads).some((v) => v === null)) fail("cannot read STREAMER_MIN_THREADS / STREAMER_MAX_THREADS / STREAMER_SPARE_CORES from ort_session.rs");

  const en = readSource("public/i18n/en/translation.json");
  try {
    const stt = JSON.parse(en).stt;
    const need = (k) => {
      if (typeof stt[k] !== "string") fail(`public/i18n/en/translation.json has no stt.${k}`);
      return stt[k] ?? "";
    };
    facts.labels = {
      hardware: [need("local_tier_auto").replace(/\s*\(recommended\)\s*$/i, ""), need("local_tier_cpu"), need("local_tier_gpu")],
      twoPass: need("local_two_pass"),
      interim: need("field_enable_interim_results"),
    };
  } catch {
    fail("cannot read the stt strings of public/i18n/en/translation.json");
    facts.labels = { hardware: [], twoPass: "", interim: "" };
  }

  // Catalog speech models that no routing row lists: installable, but no route ever plans them.
  const routing = readSource("src-tauri/model-catalog/stt-routing.json");
  const gguf = readSource("src-tauri/model-catalog/stt-gguf.json");
  facts.unrouted = [];
  try {
    const routed = new Set(JSON.stringify(JSON.parse(routing).routing).match(/stt\/[a-z0-9._-]+/g));
    for (const e of JSON.parse(gguf).entries) {
      if (!routed.has(e.id)) facts.unrouted.push(String(e.title).split(/[\s(]/)[0]);
    }
  } catch {
    fail("cannot read src-tauri/model-catalog/stt-routing.json / stt-gguf.json");
  }

  const cli = readSource("src-tauri/src/services/cli/mod.rs");
  // clap derive: the variants of `enum HeadlessCommand` (EpProbe, GpuLoad, ...) are the kebab-case commands.
  const variants = cli && /enum HeadlessCommand \{\n([\s\S]*?)\n\}/.exec(cli);
  facts.headless = variants ? [...variants[1].matchAll(/^ {4}([A-Z][A-Za-z]*)\b/gm)].map((m) => m[1].replace(/(?<!^)([A-Z])/g, "-$1").toLowerCase()) : [];
  if (!facts.headless.length) fail("cannot read the headless commands from src-tauri/src/services/cli/mod.rs");
  return facts;
}

const LABEL_FILES = (rel) => rel === "README.md" || rel === "CHANGELOG.md" || rel === "docs/OPERATOR_REFERENCE.md" || rel.startsWith("docs/manual/");
const ANY_FILE = () => true;
const ROUTE_FILES = (rel) => rel === "README.md" || rel === "docs/OPERATOR_REFERENCE.md" || rel === "docs/benchmarks/SPEECH_BENCHMARKS.md" || rel.startsWith("docs/manual/");

const RULES = [
  {
    // The speech-to-text inspector says "Processor only" / "Graphics card" / "Two-pass accuracy" / "Enable interim results".
    name: "ui-labels",
    applies: LABEL_FILES,
    check(unit, f) {
      const t = unit.text.replace(/^\|\s*\*\*[^*|]+\*\*\s*\|/, "|"); // a tier name opening a table row is not a UI label
      const options = f.labels.hardware;
      for (const m of t.matchAll(/\*\*Hardware\s*(?:→|->)\s*([^*]+)\*\*/g)) {
        if (!options.includes(m[1].trim())) return `**Hardware → ${m[1].trim()}** is not an option (the options are ${options.map((o) => `"${o}"`).join(", ")})`;
      }
      if (/Hardware\s*(?:→|->)\s*\*{0,2}(?:GPU|CPU only)\b/.test(t)) return `Hardware has no "GPU" / "CPU only" option (use ${options.map((o) => `"${o}"`).join(", ")})`;
      if (/\*\*Hardware\*\*|Hardware\s*(?:→|->)/.test(t) && /\*\*(?:GPU|CPU only)\*\*/.test(t)) {
        return `names a Hardware option **GPU** / **CPU only** (the options are ${options.map((o) => `"${o}"`).join(", ")})`;
      }
      if (/\*\*Two passes?\*\*/i.test(t)) return `the checkbox is called "${f.labels.twoPass}"`;
      if (/\*\*Interim results\*\*/.test(t) && /(?:next time speech-to-text starts|live model|Nemotron|Parakeet)/.test(t)) {
        return `the speech-to-text checkbox is called "${f.labels.interim}" ("Interim results" is the translation panels' label)`;
      }
      return null;
    },
    facts: { labels: { hardware: ["Best for my PC", "Processor only", "Graphics card"], twoPass: "Two-pass accuracy", interim: "Enable interim results" } },
    bad: [
      "- **Speech-to-text:** leave **Hardware** on **Best for my PC** (or **CPU only**); turn **Two passes** off if the processor is busy.",
      "a GPU final model (**Hardware → GPU**); a bigger translation model",
      "If frame rate comes first, set **Hardware → CPU only**.",
      "- **Interim results** (default on; applies the next time speech-to-text starts): shows words while you are still speaking.",
    ],
    good: [
      "| **CPU only** | No NVIDIA GPU | Speech-to-text runs on the processor (**Hardware → Processor only** pins it there). | Nothing |",
      "If frame rate comes first, set **Hardware → Processor only**; **Hardware → Graphics card** shares the card.",
      "turn **Two-pass accuracy** off; **Enable interim results** (default on; applies the next time speech-to-text starts)",
      "Tick **Interim results** in the Automatic or AI model panel to turn interim translations on.",
    ],
  },
  {
    // ort_session.rs: streamer_threads_for(cores) = clamp(cores - SPARE, MIN, MAX); the game keeps SPARE cores only from MIN + SPARE cores up.
    name: "streamer-threads",
    applies: ANY_FILE,
    check(unit, f) {
      if (!/Streamer/.test(unit.text)) return null;
      const { min, max, spare } = f.threads;
      const t = unit.text;
      if (/all (?:of )?(?:your |the )?(?:physical )?(?:processor )?cores (?:but|except) two|all but two/i.test(t)) {
        const floor = new RegExp(`(?:never fewer than|at least|no fewer than)\\D{0,40}\\b${min}\\b|within ${min} to ${max}\\b`, "i");
        const ceil = new RegExp(`(?:never more than|nor more than|at most|no more than)\\D{0,40}\\b${max}\\b|within ${min} to ${max}\\b`, "i");
        if (!floor.test(t) || !ceil.test(t)) return `"all cores but two" without its bounds: the Streamer takes never fewer than ${min} and never more than ${max} threads (ort_session.rs)`;
      }
      if (/\b(?:keeps?|leaves?|leaving) (?:the game |it )?(?:two|2) (?:spare |free )?(?:processor )?cores\b/i.test(t)) {
        const full = min + spare;
        const qualifier = new RegExp(
          `\\b(?:${full}|six) or more physical cores|fewer than ${full} physical cores|${min}-? or ${min + 1}[- ]core|${min} or ${min + 1} physical cores|${min + 1} or fewer`,
          "i",
        );
        if (!qualifier.test(t)) return `"leaves two cores" without the condition: only a PC with ${full} or more physical cores keeps two, a ${min}- or ${min + 1}-core PC keeps fewer (ort_session.rs)`;
      }
      if (/Streamer preset still runs one thread|one ONNX Runtime thread \(the Streamer preset\)|thread-count setting and an H3 live record come before/i.test(t)) {
        return "the CPU Streamer no longer runs one thread (P7-I0: ort_session::vc_threads)";
      }
      return null;
    },
    facts: { threads: { min: 4, max: 6, spare: 2 } },
    bad: [
      "The default Streamer preset uses 4 to 6 of your processor cores (all but two) at below-normal priority, so it keeps up in real time and leaves two cores to the game.",
      "The Streamer preset still runs one thread (P7-B: RTF 2.7 on one thread); a thread-count setting and an H3 live record come before changing it.",
      "On the CPU, the thread policy comes next: the reference configuration with one ONNX Runtime thread (the Streamer preset), two (`protect_game`) and the default pool.",
      "A 6-core PC takes four and leaves two cores to the game, and the Streamer takes all physical cores but two, at least 4.",
    ],
    good: [
      "The Streamer preset runs on all your physical processor cores but two, never fewer than 4 threads and never more than 6; with 6 or more physical cores you keep two cores for the game.",
      "the Streamer preset takes all physical cores but two within 4 to 6; a 6-core PC takes four and leaves two cores to the game, a 4- or 5-core PC also takes four and keeps fewer",
      "the llama-server translator leaves two cores to the game",
    ],
  },
  {
    // A catalog speech model that no routing row lists cannot be what "Best for my PC" / "Hardware" picks.
    name: "unrouted-models",
    applies: ROUTE_FILES,
    check(unit, f) {
      const own = unit.text;
      const ctx = unit.table ? `${unit.ctx} ${own}` : own;
      for (const name of f.unrouted) {
        if (!new RegExp(`\\b${name}\\b`).test(own)) continue;
        const claims = /Best for my PC|\*\*Hardware|Hardware\s*(?:→|->)|route can plan|routable|offered the GPU/i.test(ctx);
        const denies = /not routed|no route plans|never (?:picked|selects|routed|planned)|not routable|no row of the routing table|installable but|catalogued but not routed/i.test(ctx);
        if (claims && !denies) return `${name} is in the catalog but no routing row lists it (stt-routing.json), so Best for my PC / Hardware never plans it: say "installable, not routed"`;
      }
      return null;
    },
    facts: { unrouted: ["Voxtral"] },
    bad: [
      { table: true, ctx: "Opt-in GPU models (the transcribe.cpp runtime pack; **Hardware → Graphics card**, or **Best for my PC** when one fits).", text: "| Voxtral Mini 4B Realtime Q4_K_M | 2.8 GB | about 4.1 GB | Streaming; append-only partials |" },
      "The heavy GPU models (Cohere Transcribe, Qwen3-ASR 1.7B, Voxtral) can be picked by **Best for my PC** when one fits.",
    ],
    good: [
      "Voxtral Mini 4B Realtime can be installed from Models & Runtimes, but no route plans it: **Hardware** never selects it.",
      { table: true, ctx: "Opt-in GPU models that the route can plan (**Hardware → Graphics card**).", text: "| Cohere Transcribe Q4_K | 1.6 GB | about 2 GB | English and European languages only |" },
      "Voxtral is a streaming model.",
    ],
  },
];

function runFactRules(facts) {
  const found = [];
  // Self-tests first: a rule that no longer flags the review's sentence (or flags the corrected one) is itself a problem.
  for (const r of RULES) {
    const f = { ...facts, ...(r.facts ?? {}) };
    const unit = (s) => (typeof s === "string" ? { line: 0, text: s, table: false, ctx: "" } : { line: 0, ...s });
    for (const s of r.bad) if (!r.check(unit(s), f)) found.push(`docs/check-docs.mjs: self-test of rule "${r.name}" no longer flags: ${(typeof s === "string" ? s : s.text).slice(0, 90)}`);
    for (const s of r.good) {
      const m = r.check(unit(s), f);
      if (m) found.push(`docs/check-docs.mjs: self-test of rule "${r.name}" flags a correct sentence (${m}): ${(typeof s === "string" ? s : s.text).slice(0, 90)}`);
    }
  }
  for (const rel of files) {
    const applicable = RULES.filter((r) => r.applies(rel));
    if (!applicable.length) continue;
    for (const unit of unitsOf(read(rel))) {
      for (const r of applicable) {
        const m = r.check(unit, facts);
        if (m) found.push(`${rel}:${unit.line}: ${r.name}: ${m}`);
      }
    }
  }

  // Every `--headless` command of the app is in the developer guide's table.
  const guide = read("docs/DEVELOPER_GUIDE.md");
  for (const cmd of facts.headless) {
    if (!new RegExp(`^\\| \`${cmd}\` \\|`, "m").test(guide)) found.push(`docs/DEVELOPER_GUIDE.md: headless command \`${cmd}\` (src-tauri/src/services/cli/mod.rs) is missing from the headless table`);
  }

  // Pieces the P7 review asked for that must stay.
  const required = [
    ["docs/benchmarks/SPEECH_BENCHMARKS.md", /^### Moonshine and long utterances/m, "the Moonshine long-utterance measurement (the 9 s cut)"],
    ["docs/benchmarks/SPEECH_BENCHMARKS.md", /DEVELOPER_GUIDE\.md#speech-benchmark/, "a link to the speech-benchmark section of the developer guide"],
  ];
  for (const [rel, re, what] of required) if (!re.test(read(rel))) found.push(`${rel}: missing ${what}`);

  return found;
}

const factsLoaded = loadFacts();
problems.push(...factsLoaded.errors.map((e) => `docs/check-docs.mjs: ${e}`));
if (!factsLoaded.errors.length) problems.push(...runFactRules(factsLoaded));

for (const p of problems) console.log(p);
console.log(`${files.length} files checked, ${problems.length} problem(s), ${todos} TODO(P7-numbers) placeholder(s)${noTodo ? " (counted as problems)" : ""}`);
process.exit(problems.length ? 1 : 0);
