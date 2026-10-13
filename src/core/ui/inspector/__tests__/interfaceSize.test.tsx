/**
 * The Interface size row of Settings (`InterfaceSizeSection`): what it shows (server-side render, with the English strings
 * the app will have: `public/i18n/en/translation.json` plus this fix's staged `scripts/i18n-staging/fix-A-ui-scale-auto.json`)
 * and what a click on each chip does to the host (`ApiServer.changeScale` / `changeScaleAuto`). The repo has no DOM
 * environment, so the click part runs the components outside React's renderer like `onboarding.wiring.test.tsx` does:
 * `withHooks` gives them hook state and `expand` replaces each function component in a tree with what it renders. The
 * handlers on the resulting `button` elements are the ones a click would call. `useSnapshot` is the identity here, so
 * the same stubbed host state serves both parts.
 */
import fs from "node:fs";
import path from "node:path";
import * as React from "react";
import type { ReactElement, ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import en from "../../../../../public/i18n/en/translation.json";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: vi.fn(() => true), invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/app", () => ({ getVersion: vi.fn(async () => "0.0.0") }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ confirm: vi.fn(async () => true) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { success: vi.fn(), warn: vi.fn(), error: vi.fn() }) }));
vi.mock("@/i18n", () => ({ i18nLanguages: [], loadLanguageFile: vi.fn() }));
vi.mock("@/core/services/diagnostics/DiagnosticsPanel", () => ({ default: () => null }));
vi.mock("@/client", () => ({ useGetState: (select: (state: unknown) => unknown) => select({ canvas: { w: 1920, h: 1080 }, author: "" }) }));
vi.mock("valtio", async (importOriginal) => ({ ...(await importOriginal<object>()), useSnapshot: (state: unknown) => state }));

import Inspector_Settings, { InterfaceSizeSection } from "../inspector_settings";

const ROOT = path.resolve(__dirname, "../../../../..");

type Json = { [key: string]: string | Json };
function merge(base: Json, extra: Json): Json {
  const out: Json = { ...base };
  for (const [k, v] of Object.entries(extra)) {
    const existing = out[k];
    out[k] = typeof v === "object" && typeof existing === "object" ? merge(existing, v) : v;
  }
  return out;
}

/** English as the app has it: the strings that are merged already, plus this fix's staged ones (a no-op once merged). */
function english(): Json {
  const staged = path.join(ROOT, "scripts", "i18n-staging", "fix-A-ui-scale-auto.json");
  return merge(en as Json, fs.existsSync(staged) ? (JSON.parse(fs.readFileSync(staged, "utf8")) as Json) : {});
}

beforeAll(async () => {
  await i18n.use(initReactI18next).init({ lng: "en", resources: { en: { translation: english() } }, interpolation: { escapeValue: false } });
});

afterEach(() => vi.unstubAllGlobals());

/** The host-side globals the row reads: the settings, what Auto picked for the monitor, and the two ways to change the size. */
function mount(opts: { auto?: boolean; manual?: number; detected?: { auto: number; known: boolean } } = {}) {
  const api = {
    state: proxy({ uiScale: opts.manual ?? 1, uiScaleAuto: opts.auto ?? true }),
    uiScale: { detected: proxy(opts.detected ?? { auto: 1.5, known: true }) },
    changeScale: vi.fn(),
    changeScaleAuto: vi.fn(),
  };
  vi.stubGlobal("window", { ApiServer: api, Config: { features: { background_input: false } } });
  return api;
}

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

/** The chips of the rendered row: their label and whether they are lit. */
function chips(markup: string) {
  return [...markup.matchAll(/<button\b([^>]*)>(.*?)<\/button>/g)].map(([, attrs, label]) => ({
    label,
    lit: /\bbtn-active\b/.test(attrs),
    pressed: /aria-pressed="true"/.test(attrs),
  }));
}

describe("what the Interface size row shows", () => {
  it("has the label Interface size and the chips Auto, S, M, L and X, with the size Auto picked on the line below", () => {
    mount({ detected: { auto: 1.5, known: true } });
    const out = html(<InterfaceSizeSection />);
    expect(out).toContain("Interface size");
    expect(chips(out).map(c => c.label)).toEqual(["Auto", "S", "M", "L", "X"]);
    expect(out).toContain("Auto picks 150% for the monitor Sigil is on.");
    expect(out).not.toMatch(/settings\.(ui_scale|field_ui_scale)/); // no raw key
  });

  it("does not put the size in the Auto chip: the chip says Auto and nothing else", () => {
    mount({ detected: { auto: 1.5, known: true } });
    const auto = chips(html(<InterfaceSizeSection />))[0];
    expect(auto.label).toBe("Auto");
    expect(auto.label).not.toContain("%");
  });

  it("lights Auto while Auto is on, whatever the manual size is", () => {
    mount({ auto: true, manual: 1.2 });
    expect(chips(html(<InterfaceSizeSection />)).map(c => c.lit)).toEqual([true, false, false, false, false]);
    expect(chips(html(<InterfaceSizeSection />)).map(c => c.pressed)).toEqual([true, false, false, false, false]);
  });

  it("lights the manual size once Auto is off, and still says what Auto would pick", () => {
    mount({ auto: false, manual: 1.2, detected: { auto: 1.2, known: true } });
    const out = html(<InterfaceSizeSection />);
    expect(chips(out).map(c => c.lit)).toEqual([false, false, false, true, false]);
    expect(out).toContain("Auto picks 120% for the monitor Sigil is on.");
  });

  it("lights no chip for a size that is none of them (a hand-edited 1.1)", () => {
    mount({ auto: false, manual: 1.1 });
    expect(chips(html(<InterfaceSizeSection />)).map(c => c.lit)).toEqual([false, false, false, false, false]);
  });

  it("says so when the monitor could not be read, instead of passing a 100% off as a reading", () => {
    mount({ detected: { auto: 1, known: false } });
    const out = html(<InterfaceSizeSection />);
    expect(out).toContain("Sigil could not read this monitor, so Auto stays at 100%.");
    expect(out).not.toContain("Auto picks");
  });

  it("follows what Auto picked while the page is open", () => {
    const api = mount({ detected: { auto: 1, known: true } });
    expect(html(<InterfaceSizeSection />)).toContain("Auto picks 100% for the monitor Sigil is on.");
    api.uiScale.detected.auto = 1.25;
    expect(html(<InterfaceSizeSection />)).toContain("Auto picks 125% for the monitor Sigil is on.");
  });

  it("lets the chips wrap instead of giving them the 9rem fixed width that five chips overflow", () => {
    mount();
    const out = html(<InterfaceSizeSection />);
    expect(out).not.toContain("field-width");
    const group = /<div role="group"[^>]*>/.exec(out)?.[0] ?? "";
    expect(group).toMatch(/\bflex-wrap\b/);
    expect(group).toMatch(/\bmin-w-0\b/);
    expect(group).toContain('aria-label="Interface size"');
  });
});

// region ---CLICKS---

type AnyElement = ReactElement<Record<string, any>>;

const internals = (React as unknown as { __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: { ReactCurrentDispatcher: { current: unknown } } })
  .__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;

/** Runs `run` with the hooks of one component instance. */
function withHooks<T>(slots: unknown[], run: () => T): T {
  let at = 0;
  const dispatcher = {
    useState: (initial: unknown) => {
      const i = at++;
      if (!(i in slots)) slots[i] = typeof initial === "function" ? (initial as () => unknown)() : initial;
      const set = (value: unknown) => { slots[i] = typeof value === "function" ? (value as (v: unknown) => unknown)(slots[i]) : value; };
      return [slots[i], set];
    },
    useRef: (initial: unknown) => {
      const i = at++;
      if (!(i in slots)) slots[i] = { current: initial };
      return slots[i];
    },
    useMemo: (make: () => unknown) => make(),
    useCallback: (fn: unknown) => fn,
    useContext: (context: { _currentValue: unknown }) => context._currentValue,
    useEffect: () => {},
    useLayoutEffect: () => {},
  };
  const previous = internals.ReactCurrentDispatcher.current;
  internals.ReactCurrentDispatcher.current = dispatcher;
  try {
    return run();
  } finally {
    internals.ReactCurrentDispatcher.current = previous;
  }
}

/** Replaces every function component in `node` by what it renders, down to the DOM elements. */
function expand(node: ReactNode): ReactNode {
  if (Array.isArray(node)) return node.map(expand);
  if (!React.isValidElement(node)) return node;
  const el = node as AnyElement;
  if (typeof el.type === "function") return expand(withHooks([], () => (el.type as (props: unknown) => ReactNode)(el.props)));
  return { ...el, props: { ...el.props, children: expand(el.props.children) } } as AnyElement;
}

/** Every element of `node` (depth first) that `match` accepts. */
function findAll(node: ReactNode, match: (el: AnyElement) => boolean): AnyElement[] {
  if (Array.isArray(node)) return node.flatMap(child => findAll(child, match));
  if (!React.isValidElement(node)) return [];
  const el = node as AnyElement;
  return [...(match(el) ? [el] : []), ...findAll(el.props.children, match)];
}

describe("what a click on a chip does", () => {
  function click(label: string) {
    const buttons = findAll(expand(<InterfaceSizeSection />), el => el.type === "button");
    const chip = buttons.find(b => b.props.children === label);
    if (!chip) throw new Error(`no chip ${label}: ${buttons.map(b => String(b.props.children)).join(", ")}`);
    chip.props.onClick();
  }

  it("has the five chips as buttons", () => {
    mount();
    expect(findAll(expand(<InterfaceSizeSection />), el => el.type === "button").map(b => b.props.children)).toEqual(["Auto", "S", "M", "L", "X"]);
  });

  it("Auto turns Auto on and sets no size", () => {
    const api = mount({ auto: false, manual: 1.4 });
    click("Auto");
    expect(api.changeScaleAuto).toHaveBeenCalledTimes(1);
    expect(api.changeScale).not.toHaveBeenCalled();
  });

  it("S, M, L and X set their size, which turns Auto off, and never ask for Auto", () => {
    const api = mount();
    for (const [label, size] of [["S", 0.8], ["M", 1], ["L", 1.2], ["X", 1.4]] as const) {
      click(label);
      expect(api.changeScale).toHaveBeenLastCalledWith(size);
    }
    expect(api.changeScale).toHaveBeenCalledTimes(4);
    expect(api.changeScaleAuto).not.toHaveBeenCalled();
  });
});

describe("where the row sits in Settings", () => {
  it("is in the Application section, after the theme and before the language", () => {
    mount();
    // The settings panel itself, run with hooks but not expanded: its direct tree names the sections it is made of.
    const settings = withHooks([], () => (Inspector_Settings as unknown as { type: () => ReactNode }).type());
    const labels = findAll(settings, el => el.type === InterfaceSizeSection || typeof el.props.label === "string").map(el =>
      el.type === InterfaceSizeSection ? "interface-size" : String(el.props.label));
    const at = labels.indexOf("interface-size");
    expect(at).toBeGreaterThan(-1);
    expect(labels[at - 1]).toBe("settings.field_app_theme");
    expect(labels[at + 1]).toBe("settings.field_language");
    expect(labels.filter(l => l === "interface-size")).toHaveLength(1);
  });
});

// endregion
