/**
 * Interface size makes 1rem = 16px x size (`html { font-size: calc(16px * var(--uiscale)) }`), so a menu keeps its
 * proportions on any monitor only if everything that sizes it is in rem, and everything that decides its layout is
 * measured in the same rem. Two ways to get that wrong:
 *
 * - Tailwind's `sm:` `md:` `lg:` `xl:` `2xl:` are MEDIA queries, and rem in a media query is the initial 16px: they ignore
 *   Interface size (a 1440 px window at 150% is 960 interface px wide and must not get the 1024 px layout). Host UI uses
 *   `ui-sm:` ... `ui-2xl:` and `ui-short:` (style.css), which query the host window in rem.
 * - A width, height, padding or icon size written in px (`w-[400px]`, `width: 380`, `width: "380px"`, `size={14}`) does
 *   not grow with the text it holds.
 *
 * This scans the host UI (`src/core/**`, `src/index.tsx`, `src/style.css`) for both, plus the px sizes of the two vendor
 * stylesheets the host loads (react-toastify, SimpleBar), which `src/style.css` has to override in rem. Stream output
 * (`src/client/**`) and the caption window are NOT scanned: they stay in real pixels. A justified px lives in `ALLOWED`
 * (or `VENDOR_ALLOWED`) with the reason.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";

const SRC = path.resolve(__dirname, "../../..");
const ROOT = path.resolve(SRC, "..");

/** A width prop of Modal.Body / Drawer is rem; a number above this is px. */
const MAX_REM_PROP = 60;

/** Below this a px length is a hairline (a border, a 3 px grip dot, a 1 px guide) and stays px. */
const HAIRLINE_PX = 8;

function walk(dir: string, accept: (name: string) => boolean): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "__tests__" || entry.name === "node_modules" ? [] : walk(full, accept);
    return accept(entry.name) ? [full] : [];
  });
}

const rel = (file: string) => path.relative(SRC, file).split(path.sep).join("/");
const read = (file: string) => fs.readFileSync(file, "utf8");

/** Host UI code: every .ts/.tsx under src/core (tests excluded), and the entry file. */
const codeFiles = [...walk(path.join(SRC, "core"), n => /\.tsx?$/.test(n) && !/\.test\.tsx?$/.test(n)), path.join(SRC, "index.tsx")];
const tsxFiles = codeFiles.filter(f => f.endsWith(".tsx"));
/** Host UI styles: style.css and the CSS modules next to the components. */
const cssFiles = [path.join(SRC, "style.css"), ...walk(path.join(SRC, "core"), n => n.endsWith(".css"))];

/**
 * The justified exceptions. `token` is the matched text (a class, a prop, or `selector { property }` for CSS). Each one
 * needs a reason; an entry that no longer matches anything fails the "no stale entries" test, so the list cannot rot.
 */
const ALLOWED: { file: string; token: string; why: string }[] = [
  { file: "core/ui/uiScale.ts", token: "width: 800", why: "the 800 x 600 OS window the app opens at, in logical px (window size, not drawn UI)" },
  { file: "core/ui/uiScale.ts", token: "height: 600", why: "the 800 x 600 OS window the app opens at, in logical px (window size, not drawn UI)" },
  { file: "style.css", token: ":root { font-size }", why: "the base of the whole scale: 16px x --uiscale" },
  { file: "style.css", token: ".dot { width }", why: "the stream particle (src/client/services/particles adds the class): stream output stays in real pixels" },
  { file: "style.css", token: ".dot { height }", why: "the stream particle (src/client/services/particles adds the class): stream output stays in real pixels" },
  { file: "core/ui/Modal.tsx", token: ", width = 31.25", why: "Modal.Body's width prop is rem (31.25 is the 500 px of Interface size 1), not px" },
  { file: "core/ui/sigil/StatusFooter.tsx", token: "left: x + 14", why: "the action-bar drag ghost sits 14 screen px from the mouse (a cursor offset, not a drawn size)" },
  { file: "core/ui/sigil/StatusFooter.tsx", token: "top: y + 14", why: "the action-bar drag ghost sits 14 screen px from the mouse (a cursor offset, not a drawn size)" },
];

/**
 * Px sizes of a vendor stylesheet that `src/style.css` does not have to override. `token` is `selector { property }` as the
 * vendor writes it.
 */
const VENDOR_ALLOWED: { file: string; token: string; why: string }[] = [
  { file: "simplebar-react/dist/simplebar.min.css", token: ".hs-dummy-scrollbar-size { height }", why: "an invisible element SimpleBar measures the native scrollbar with; never drawn" },
  { file: "simplebar-react/dist/simplebar.min.css", token: ".hs-dummy-scrollbar-size { width }", why: "an invisible element SimpleBar measures the native scrollbar with; never drawn" },
];
/** The vendor stylesheets the host loads (a new `import "pkg/x.css"` has to be added here and its px sizes reviewed). */
const VENDOR_CSS = ["react-toastify/dist/ReactToastify.min.css", "simplebar-react/dist/simplebar.min.css"];
const used = new Set<string>();
const allowed = (file: string, token: string): boolean => {
  const hit = ALLOWED.find(a => a.file === file && a.token === token);
  if (hit) used.add(`${hit.file} ${hit.token}`);
  return hit !== undefined;
};

/** `text` with every comment of a stylesheet blanked (same length and line count, so positions stay right). */
const blankComments = (text: string) => text.replace(/\/\*[\s\S]*?\*\//g, c => c.replace(/[^\n]/g, " "));

/** Every match of `re` in `text` with its line, minus the allow-listed ones. Stylesheets are read without their comments. */
function findings(files: string[], re: RegExp, token: (m: RegExpExecArray) => string | null): string[] {
  const out: string[] = [];
  for (const file of files) {
    const text = file.endsWith(".css") ? blankComments(read(file)) : read(file);
    for (const m of text.matchAll(new RegExp(re.source, re.flags.includes("g") ? re.flags : `${re.flags}g`))) {
      const t = token(m as RegExpExecArray);
      if (t === null || allowed(rel(file), t)) continue;
      out.push(`${rel(file)}:${text.slice(0, m.index).split("\n").length}  ${t}`);
    }
  }
  return out;
}

/** Utilities whose arbitrary value is a length that lays UI out (not colors, radii, shadows, text sizes or timings). */
const SIZING =
  /^(?:min-|max-)?[wh]$|^size$|^basis$|^gap(?:-[xy])?$|^space-[xy]$|^[pm][xytblrse]?$|^-?(?:inset(?:-[xy])?|top|right|bottom|left|start|end)$|^-?translate-[xy]$|^leading$|^tracking$|^indent$|^(?:grid-cols|grid-rows|auto-cols|auto-rows)$|^(?:scroll-[pm][xytblrse]?)$/;
const hasBigPx = (text: string) => [...text.matchAll(/(\d+(?:\.\d+)?)px/g)].some(px => Number(px[1]) > HAIRLINE_PX);

/** CSS properties that size or space something (a px value of one of these does not follow Interface size). */
const LAYOUT_PROPS =
  /^(?:(?:min-|max-)?(?:width|height|inline-size|block-size)|font-size|line-height|gap|row-gap|column-gap|flex-basis|(?:padding|margin|inset)(?:-[a-z-]+)?|top|right|bottom|left|--[\w-]*(?:width|height|size))$/;

const NUM = String.raw`(\d+(?:\.\d+)?)`;
const TICK = "`";
const QUOTE = "[\"'`]";
const NOT_QUOTE = "[^\"'`]";
/** Style-object keys that take a bare number as px (React writes `{ gap: 16 }` as `16px`; line-height and opacity are unitless). */
const PX_KEYS = String.raw`(?:width|minWidth|maxWidth|height|minHeight|maxHeight|fontSize|padding(?:Top|Right|Bottom|Left|Inline|Block)?(?:Start|End)?|margin(?:Top|Right|Bottom|Left|Inline|Block)?(?:Start|End)?|gap|rowGap|columnGap|top|right|bottom|left|inset|flexBasis|borderRadius|border(?:Top|Bottom)(?:Left|Right)Radius)`;
/** Window and element sizes in real px, which a breakpoint in code must not be compared with. */
const WINDOW_SIZE = String.raw`(?:innerWidth|innerHeight|outerWidth|outerHeight|clientWidth|clientHeight|screen\.(?:width|height|availWidth|availHeight))`;

const scans = {
  /**
   * Also max-sm:, min-[640px]: and [@media(...)]: which are media queries too. Not the container variants (`@sm:`,
   * `@min-[20rem]:`): a container query measures in the root's rem, so it follows Interface size (`@[400px]:` is `pxContainerVariants`).
   */
  mediaVariants: (files = codeFiles) =>
    findings(files, /(?<![\w@-])(?:(?:max-|min-)?(?:sm|md|lg|xl|2xl)|(?:max|min)-\[[^\]\s]*\]):(?=[\w[!(-])|\[@media[^\]]*\]:/g, m => m[0]),

  /**
   * Media and container queries that measure in px. `@media (min-width: 40rem)` is no better than px: rem in a media query
   * is the initial 16px. Covers the range syntax too (`(width >= 40rem)`, `(40rem <= width <= 60rem)`), `device-width`,
   * and `@container` with a px length (a rem one follows Interface size, which is how the `ui-*` variants work).
   */
  mediaQueriesInCss: (files = cssFiles) =>
    findings(files, /@media[^{]*\([^()]*\b(?:width|height)\b[^()]*\)|@container[^{]*\d+(?:\.\d+)?px[^){]*\)?/g, m => m[0]),

  /** Tailwind's arbitrary container variants with a px length (`@[400px]:flex`, `@min-[400px]:`): real px, like a px query. */
  pxContainerVariants: (files = codeFiles) =>
    findings(files, /(?<![\w-])(?:[\w-]+:)*@(?:min-|max-)?\[[^\]\s]*\]:/g, m => (hasBigPx(m[0]) ? m[0] : null)),

  /**
   * A breakpoint in code: `matchMedia("(min-width: ...)")` or a window/element size compared with a literal
   * (`window.innerWidth < 1024`). It would decide the layout in real px, whatever Interface size is. Compare with `uiPx(n)`
   * or ask the CSS (`ui-*`) instead.
   */
  jsThresholds: (files = codeFiles) =>
    findings(
      files,
      new RegExp(
        String.raw`matchMedia\(\s*${QUOTE}${NOT_QUOTE}*\b(?:width|height)\b|\b${WINDOW_SIZE}\s*[<>]=?\s*${NUM}|${NUM}\s*[<>]=?\s*[\w$.]*\b${WINDOW_SIZE}`,
        "g",
      ),
      m => {
        const n = m[1] ?? m[2];
        return n === undefined || Number(n) > HAIRLINE_PX ? m[0] : null;
      },
    ),

  arbitraryClasses: (files = codeFiles) =>
    findings(files, /(?<![\w-])(?:[\w-]+:)*(!?-?[a-z][\w-]*)-\[([^\]\s]*)\]/g, m => (SIZING.test(m[1].replace(/^!/, "")) && hasBigPx(m[2]) ? m[0] : null)),

  arbitraryProperties: (files = codeFiles) =>
    findings(
      files,
      /\[(?:min-|max-)?(?:width|height|inline-size|block-size|font-size|padding[\w-]*|margin[\w-]*|gap|top|left|right|bottom|inset[\w-]*):[^\]\s]*\]/g,
      m => (hasBigPx(m[0]) ? m[0] : null),
    ),

  /**
   * A number that ends up as px, in any host code (components, helpers, services; the number may be a literal or a default):
   * - a style/animate object value: `{ width: 380 }`, `{ fontSize: 14 }`, `{ paddingLeft: 24 }`, `{ top: y + 14 }` (React adds
   *   `px` to a bare number), and the same written as a string (`"380px"`, `"calc(100% - 120px)"`, `` `${n}px` ``);
   * - a DOM size attribute (`<svg width={20}>`, `<svg width="20">`), an icon `size={14}`, and a `size = 20` default.
   * A string with a unit ("23.75rem") is not matched. The width props of our own components (`Modal.Body`, `Drawer`) take rem
   * as numbers and are not matched: they are JSX attributes of a component, not of a DOM element.
   */
  pxNumbers: (files = codeFiles) => {
    const DOM = String.raw`<(?:svg|img|canvas|video|iframe)\b[^>]*?\b(?:width|height)=`;
    const re = new RegExp(
      [
        // { width: 380 }, { fontSize: 14 }, { left: x + 14 }, { marginLeft: -2 }
        String.raw`\b${PX_KEYS}\s*:\s*(?:[A-Za-z_$][\w$.]*\s*[-+*]\s*)?-?${NUM}(?=\s*[,}\n)])`,
        // { width: "380px" }, { maxWidth: "calc(100% - 120px)" }
        String.raw`\b${PX_KEYS}\s*:[^,}\n]*?${QUOTE}${NOT_QUOTE}*?${NUM}\s*px${NOT_QUOTE}*${QUOTE}?`,
        // { width: `${n}px` }, { width: `${380}px` }
        String.raw`\b${PX_KEYS}\s*:[^,}\n]*?${TICK}[^${TICK}]*?\$\{[^}]*\}\s*px[^${TICK}]*${TICK}?`,
        // ({ size = 20 }) => <svg width={size} />
        String.raw`[{,]\s*(?:size|width|height|iconSize)\s*=\s*${NUM}(?=\s*[,}])`,
        // <svg width={20}>, <svg width="20">, <RiX size={14}>, <RiX size="14">
        String.raw`${DOM}\{\s*${NUM}\s*\}`,
        String.raw`${DOM}"${NUM}"`,
        String.raw`\bsize=\{\s*${NUM}\s*\}`,
        String.raw`\bsize="${NUM}"`,
      ].join("|"),
      "g",
    );
    return findings(files, re, m => {
      const n = m.slice(1).find(g => g !== undefined);
      return n === undefined || Number(n) > HAIRLINE_PX ? m[0].replace(/\s+/g, " ").trim() : null;
    });
  },

  /**
   * `Modal.Body`, `Drawer` and `InspectorDrawer` take their widths as rem numbers (`width={26.25}`). A number over 60
   * there is px written the old way (`width={420}` would be a 6720 px modal).
   */
  remProps: (files = codeFiles) =>
    findings(
      files,
      /<(?:Modal\.Body|Drawer|InspectorDrawer)\b[^>]*?\b(?:width|minWidth|maxWidth)=\{\s*(\d+(?:\.\d+)?)\s*\}/g,
      m => (Number(m[1]) > MAX_REM_PROP ? m[0].replace(/\s+/g, " ").trim() : null),
    ),

  pxInCss: (files = cssFiles) => {
    const out: string[] = [];
    for (const file of files) {
      for (const d of cssDeclarations(read(file))) {
        if (!LAYOUT_PROPS.test(d.property) || !hasBigPx(d.value)) continue;
        const token = `${d.selector} { ${d.property} }`;
        if (!allowed(rel(file), token)) out.push(`${rel(file)}  ${token}: ${d.value}`);
      }
    }
    return out;
  },
};

describe("host UI breakpoints and sizes follow the Interface size", () => {
  it("scans the files it is meant to scan", () => {
    const names = codeFiles.map(rel);
    expect(names).toEqual(
      expect.arrayContaining([
        "core/ui/sigil/SigilLayout.tsx",
        "core/ui/Modal.tsx",
        "core/ui/overlay-input.tsx",
        "core/ui/uiScale.ts",
        // components outside core/ui, and helpers in .ts: Tailwind scans them too, and a style object in them is px as well
        "core/services/diagnostics/DiagnosticsPanel.tsx",
        "core/services/readings/ReadingsSection.tsx",
        "index.tsx",
      ]),
    );
    expect(names.some(n => n.startsWith("client/"))).toBe(false);
    expect(tsxFiles.length).toBeGreaterThan(100);
    expect(cssFiles.map(rel)).toEqual(expect.arrayContaining(["style.css", "core/ui/background-input.module.css"]));
  });

  it("uses no sm: md: lg: xl: 2xl: media-query variant (use ui-sm: ... ui-2xl:)", () => {
    expect(scans.mediaVariants()).toEqual([]);
  });

  it("uses no width/height @media query (min-/max-, range syntax) and no px @container query in the host stylesheets", () => {
    expect(scans.mediaQueriesInCss()).toEqual([]);
  });

  it("uses no px arbitrary container variant (@[400px]:, @min-[400px]:)", () => {
    expect(scans.pxContainerVariants()).toEqual([]);
  });

  it("decides no layout in code from a real-px window size (matchMedia width, innerWidth < 1024)", () => {
    expect(scans.jsThresholds()).toEqual([]);
  });

  it("has no arbitrary px length above a hairline in a sizing class (w-[400px], min-w-[200px], gap-[12px], w-[min(240px,...)])", () => {
    expect(scans.arbitraryClasses()).toEqual([]);
  });

  it("has no arbitrary-property px length above a hairline ([width:300px])", () => {
    expect(scans.arbitraryProperties()).toEqual([]);
  });

  it("has no number written as px for a size (width: 380, width: \"380px\", fontSize: 14, paddingLeft: 24, width={380}, size={14}, size = 20)", () => {
    expect(scans.pxNumbers()).toEqual([]);
  });

  it("passes no px number to the rem width props of Modal.Body and Drawer", () => {
    expect(scans.remProps()).toEqual([]);
  });

  it("has no px width, height, padding, margin, offset or font size above a hairline in the host stylesheets", () => {
    expect(scans.pxInCss()).toEqual([]);
  });

  it("has no stale ALLOWED entry (every exception still matches something)", () => {
    used.clear();
    Object.values(scans).forEach(scan => scan());
    const stale = ALLOWED.filter(a => !used.has(`${a.file} ${a.token}`)).map(a => `${a.file} ${a.token}`);
    expect(stale).toEqual([]);
  });
});

describe("the guard itself", () => {
  // Fixtures written to a temp folder: what the scans must catch, and what they must let through.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-layout-guard-"));
  const tsx = path.join(dir, "Bad.tsx");
  const css = path.join(dir, "bad.css");
  fs.writeFileSync(
    tsx,
    [
      `export const A = () => <div className="hidden lg:block md:flex max-sm:p-2 xl:w-1/2 2xl:grid min-[640px]:flex [@media(max-height:680px)]:hidden hover:sm:flex" />;`,
      `export const B = () => <div className="w-[400px] min-w-[200px] gap-[12px] hover:max-w-[320px] w-[min(240px,calc(100vw-32px))] [width:300px] [min-height:120px]" />;`,
      `export const C = () => <div style={{ width: 380, height: 40 }} className="text-[14px] w-[3px] h-[8px] mb-[-1px] rounded-[20px] shadow-[0_0_20px_black] ui-lg:block ui-short:hidden w-[26rem]" />;`,
      `export const D = () => <><svg width={20} height={20} /><RiX size={14} /><RiY size="0.875rem" /><Drawer width={23.75} /><Modal.Body width={26.25}><Modal.Body width={420}><Drawer position="left" minWidth={280} /><motion.div animate={{ width: "23.75rem", height: 0 }} /></>;`,
      // px written as a string or a template, and the other style keys React turns into px
      `export const E = () => <aside animate={{ width: "380px" }} style={{ minWidth: \`\${380}px\`, height: \`\${rows}px\`, maxWidth: "calc(100% - 120px)", fontSize: 14, paddingLeft: 24, top: 40, gap: 16, borderRadius: 12, left: x + 14, flexBasis: 200, marginLeft: -20 }} />;`,
      // what stays: rem strings and templates, hairlines, unitless numbers, percentages
      `export const F = () => <aside style={{ width: "23.75rem", minWidth: \`\${w}rem\`, height: "calc(100% - 2rem)", fontSize: 8, padding: 4, top: 0, gap: "1rem", lineHeight: 20, opacity: 14, zIndex: 100, flex: 1, borderRadius: "0.5rem", maxWidth: "100%", marginLeft: -2 }} />;`,
      `export const G = ({ size = 20, className }) => <svg width="24" height="24" />;`,
      `export const H = ({ size = "1.25rem", width = 6, gap = 0 }) => <><svg width="100%" /><img width={x} /><RiZ size="1rem" /></>;`,
      // a breakpoint decided in code, and one that is not
      `export const I = () => window.matchMedia("(min-width: 1024px)").matches || window.innerWidth < 1024 || 640 > window.innerHeight || el.clientWidth >= 480;`,
      `export const J = () => window.matchMedia("(prefers-color-scheme: dark)").matches || window.innerWidth <= 0 || el.clientWidth > uiPx(300) || window.innerHeight < minHeight;`,
      `export const K = () => <div className="@[400px]:flex @min-[400px]:grid @max-[20rem]:hidden @sm:block @[3px]:p-0" />;`,
    ].join("\n"),
  );
  fs.writeFileSync(
    css,
    [
      `.x { width: 300px; padding: 4px 20px; border: 1px solid red; font-size: 0.75rem; }`,
      `.y { --my-toast-width: 320px; min-height: 2rem; }`,
      `@media (min-width: 640px) { .z { color: red; } }`,
      `/* @media (max-width: 480px) { .w { width: 999px } } */`,
      `@media (width >= 40rem) { .a { color: red; } }`,
      `@media (40rem <= width <= 60rem) { .b { color: red; } }`,
      `@media screen and (max-height: 40rem) { .c { color: red; } }`,
      `@container (min-width: 400px) { .d { color: red; } }`,
      `@container sigil-ui (min-width: 40rem) { .e { color: red; } }`,
      `@media (prefers-color-scheme: dark) { .f { color: red; } }`,
      `@media (hover: hover) { .g { color: red; } }`,
    ].join("\n"),
  );
  afterAll(() => fs.rmSync(dir, { recursive: true, force: true }));
  const tokens = (found: string[]) => found.map(f => f.replace(/^[^ ]+  /, ""));

  it("catches media-query breakpoints but not the ui-* ones", () => {
    expect(tokens(scans.mediaVariants([tsx]))).toEqual(["lg:", "md:", "max-sm:", "xl:", "2xl:", "min-[640px]:", "[@media(max-height:680px)]:", "sm:"]);
  });

  it("catches px lengths above a hairline in sizing classes, and only those", () => {
    expect(tokens(scans.arbitraryClasses([tsx]))).toEqual(["w-[400px]", "min-w-[200px]", "gap-[12px]", "hover:max-w-[320px]", "w-[min(240px,calc(100vw-32px))]"]);
    expect(tokens(scans.arbitraryProperties([tsx]))).toEqual(["[width:300px]", "[min-height:120px]"]);
  });

  it("catches px numbers for widths, heights and icon sizes, not rem strings or component rem props", () => {
    const found = tokens(scans.pxNumbers([tsx]));
    expect(found.slice(0, 4)).toEqual(["width: 380", "height: 40", "<svg width={20}", "size={14}"]);
  });

  it("catches px written as a string or a template, and the other style keys React turns into px", () => {
    const found = tokens(scans.pxNumbers([tsx]));
    expect(found.slice(4)).toEqual([
      'width: "380px"',
      "minWidth: `${380}px`",
      "height: `${rows}px`",
      'maxWidth: "calc(100% - 120px)"',
      "fontSize: 14",
      "paddingLeft: 24",
      "top: 40",
      "gap: 16",
      "borderRadius: 12",
      "left: x + 14",
      "flexBasis: 200",
      "marginLeft: -20",
      // a px default carried in a variable, and a string attribute
      "{ size = 20",
      '<svg width="24"',
    ]);
  });

  it("catches px written for the rem width props of Modal.Body and Drawer", () => {
    expect(tokens(scans.remProps([tsx]))).toEqual(["<Modal.Body width={420}", "<Drawer position=\"left\" minWidth={280}"]);
  });

  it("catches px layout values in a stylesheet, not rem, hairlines or comments", () => {
    expect(tokens(scans.pxInCss([css]))).toEqual([".x { width }: 300px", ".x { padding }: 4px 20px", ".y { --my-toast-width }: 320px"]);
  });

  it("catches width/height media queries (min-/max-, range syntax) and px container queries, not rem ones or other features", () => {
    expect(tokens(scans.mediaQueriesInCss([css]))).toEqual([
      "@media (min-width: 640px)",
      "@media (width >= 40rem)",
      "@media (40rem <= width <= 60rem)",
      "@media screen and (max-height: 40rem)",
      "@container (min-width: 400px)",
    ]);
  });

  it("catches px arbitrary container variants", () => {
    expect(tokens(scans.pxContainerVariants([tsx]))).toEqual(["@[400px]:", "@min-[400px]:"]);
  });

  it("catches a breakpoint decided in code from a real-px window size, not a theme query or a compare with 0 or with uiPx()", () => {
    expect(tokens(scans.jsThresholds([tsx]))).toEqual([
      'matchMedia("(min-width',
      "innerWidth < 1024",
      "640 > window.innerHeight",
      "clientWidth >= 480",
    ]);
  });
});

describe("the ui-* variants and their size container (style.css)", () => {
  const css = read(path.join(SRC, "style.css")).replace(/\/\*[\s\S]*?\*\//g, "");
  const variants = [...css.matchAll(/@custom-variant\s+(ui-[\w-]+)\s*\(([^;]*)\)\s*;/g)].map(m => ({ name: m[1], query: m[2] }));

  it("defines ui-sm/md/lg/xl/2xl at Tailwind's own thresholds, in rem and in ascending order, and ui-short", () => {
    const widths = variants.filter(v => /^ui-(sm|md|lg|xl|2xl)$/.test(v.name));
    expect(widths.map(v => v.name)).toEqual(["ui-sm", "ui-md", "ui-lg", "ui-xl", "ui-2xl"]);
    expect(widths.map(v => v.query)).toEqual([
      "@container sigil-ui (min-width: 40rem)",
      "@container sigil-ui (min-width: 48rem)",
      "@container sigil-ui (min-width: 64rem)",
      "@container sigil-ui (min-width: 80rem)",
      "@container sigil-ui (min-width: 96rem)",
    ]);
    expect(variants.find(v => v.name === "ui-short")?.query).toBe("@container sigil-ui (max-height: 42.5rem)");
  });

  it("makes body the host-only size container they query, and keeps it from collapsing", () => {
    const rule = /html\.host\s+body\s*\{([^}]*)\}/.exec(css.replace(/@apply[^;]*;/g, ""))?.[1] ?? "";
    expect(rule).toMatch(/container-type:\s*size\b/);
    expect(rule).toMatch(/container-name:\s*sigil-ui\b/);
    // size containment ignores the content: html, body and #root must keep their explicit window size.
    const sized = /:root,\s*html,\s*body,\s*#root\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
    expect(sized).toMatch(/width:\s*100vw/);
    expect(sized).toMatch(/height:\s*100vh/);
  });

  it("only uses ui-* variants that exist (a typo such as ui-xxl: silently does nothing)", () => {
    const defined = new Set(variants.map(v => v.name));
    const unknown = findings(codeFiles, /(?<![\w-])((?:ui-[\w-]+)):(?=[\w[!(-])/g, m => (defined.has(m[1]) ? null : m[0]));
    expect(unknown).toEqual([]);
  });

  it("is used: the layouts that depended on lg: and sm: have their ui-* twin", () => {
    const layout = read(path.join(SRC, "core/ui/sigil/SigilLayout.tsx"));
    expect(layout).toMatch(/hidden ui-lg:block/);
  });

  it("is switched on before the host UI first renders: `host` is on <html> ahead of renderView(<SigilRoot />)", () => {
    // Without it `html.host body` is not a container, every ui-* variant silently never matches (the right panel and the
    // action bar logo stay hidden, ui-short:hidden never fires) and nothing fails. The class is a single assignment, so
    // nothing else may overwrite `className` on <html> afterwards.
    const entry = read(path.join(SRC, "index.tsx"));
    const mark = /document\.documentElement\.(?:className\s*=\s*"host"|classList\.add\("host"\))/.exec(entry);
    const render = entry.indexOf("renderView(<SigilRoot />)");
    expect(mark, "index.tsx must put the class `host` on <html>").not.toBeNull();
    expect(render, "index.tsx must render <SigilRoot />").toBeGreaterThan(-1);
    expect(mark!.index).toBeLessThan(render);
    // the same class name the stylesheet's container rule is written for
    expect(css).toMatch(/html\.host\s+body\s*\{/);
    const overwrites = findings(codeFiles, /documentElement\.className\s*=(?!=)/g, m => m[0]);
    expect(overwrites.map(o => o.replace(/^[^ ]+  /, ""))).toEqual(["documentElement.className ="]);
    expect(overwrites[0]).toMatch(/^index\.tsx:/);
  });
});

describe("vendor stylesheets the host loads (react-toastify, SimpleBar)", () => {
  // Their px sizes cannot be scanned in our own sources, so they are read from the packages: every layout size above a
  // hairline has to be re-declared in rem in style.css (same px at Interface size 1), or be listed with a reason.
  const vendorPath = (name: string) => path.join(ROOT, "node_modules", name);
  const norm = (selector: string) =>
    selector
      .replace(/\.toasts\b/g, "") // the scope style.css puts in front (`.toasts .Toastify__toast-icon`)
      .replace(/::/g, ":")
      .replace(/\s*>\s*/g, " > ")
      .replace(/\s+/g, " ")
      .trim();
  const parts = (selector: string) => selector.split(",").map(norm).filter(Boolean);
  const mine = cssDeclarations(read(path.join(SRC, "style.css")));
  const rows = VENDOR_CSS.flatMap(file =>
    cssDeclarations(read(vendorPath(file)))
      .filter(d => LAYOUT_PROPS.test(d.property) && hasBigPx(d.value))
      .map(d => ({ file, ...d, token: `${d.selector} { ${d.property} }` })),
  );
  const overridden = (row: (typeof rows)[number]) =>
    mine.some(
      o => o.property === row.property && !hasBigPx(o.value) && (row.property.startsWith("--") || parts(o.selector).some(p => parts(row.selector).includes(p))),
    );
  const vendorAllowed = (row: (typeof rows)[number]) => VENDOR_ALLOWED.some(a => a.file === row.file && a.token === row.token);

  it("is every stylesheet from a package that the host imports", () => {
    const imported = [
      ...codeFiles.flatMap(file => [...read(file).matchAll(/^\s*import\s+["']([^"'.][^"']*\.css)["']/gm)].map(m => m[1])),
      ...[...read(path.join(SRC, "style.css")).matchAll(/@import\s+["']([^"']+\.css)["']/g)].map(m => m[1]),
    ];
    expect([...new Set(imported)].sort()).toEqual([...VENDOR_CSS].sort());
  });

  it("reads the px sizes it is meant to review", () => {
    const tokens = rows.map(r => r.token);
    expect(tokens).toEqual(expect.arrayContaining([".Toastify__toast-icon { width }", ".Toastify__spinner { width }", ".simplebar-track.simplebar-vertical { width }"]));
  });

  it("has every px size above a hairline re-declared in rem in style.css (or listed in VENDOR_ALLOWED)", () => {
    const missing = rows.filter(r => !overridden(r) && !vendorAllowed(r)).map(r => `${r.file}  ${r.token}: ${r.value}`);
    expect(missing).toEqual([]);
  });

  it("has no stale VENDOR_ALLOWED entry", () => {
    const stale = VENDOR_ALLOWED.filter(a => !rows.some(r => r.file === a.file && r.token === a.token)).map(a => `${a.file} ${a.token}`);
    expect(stale).toEqual([]);
  });
});

/**
 * The declarations of a stylesheet with the selector of the rule they are in. Comments are removed; at-rules are
 * walked into (`@layer`, `@media`, `@keyframes` steps are reported under their own prelude); `@apply` lines are not
 * declarations.
 */
function cssDeclarations(source: string): { selector: string; property: string; value: string }[] {
  const text = source.replace(/\/\*[\s\S]*?\*\//g, "");
  const out: { selector: string; property: string; value: string }[] = [];
  const stack: string[] = [];
  let buffer = "";
  const flush = () => {
    const decl = /^\s*([a-zA-Z-]+|--[\w-]+)\s*:\s*([\s\S]+?)\s*$/.exec(buffer);
    if (decl && stack.length > 0 && !/^@apply/.test(buffer.trim())) {
      out.push({ selector: stack[stack.length - 1].replace(/\s+/g, " "), property: decl[1], value: decl[2] });
    }
    buffer = "";
  };
  for (const ch of text) {
    if (ch === "{") {
      stack.push(buffer.trim());
      buffer = "";
    } else if (ch === "}") {
      flush();
      stack.pop();
    } else if (ch === ";") {
      flush();
    } else {
      buffer += ch;
    }
  }
  return out;
}
