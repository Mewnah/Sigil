import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The OBS strings changed for the stale-page fix (`obs.checklist_browser_mirror`,
 * `obs.browser_source_tips`, `obs.stale_page_hint`) must stay right in every locale, and the manual
 * must not call anything "like Curses" (English wording is in `obsWording.test.ts`).
 */

const ROOT = path.resolve(__dirname, "../../..");
const I18N_DIR = path.join(ROOT, "public/i18n");
const LOCALES = fs.readdirSync(I18N_DIR).filter((dir) => fs.existsSync(path.join(I18N_DIR, dir, "translation.json")));
const obsSection = (locale: string): Record<string, string> =>
  JSON.parse(fs.readFileSync(path.join(I18N_DIR, locale, "translation.json"), "utf8")).obs;
const lineCount = (text: string) => text.split("\n").length;
const URL_KEYS = ["checklist_browser_mirror", "browser_source_tips"] as const;

/** The names that must keep working (the Curses backup file, the legacy variables). */
const FUNCTIONAL_NAMES = /(?:Curses )?\.cursestmp|CURSES_[A-Z0-9_]*\*?/g;

describe("OBS strings in every locale", () => {
  it("covers all the locales we ship", () => {
    expect(LOCALES.length).toBeGreaterThanOrEqual(21);
    expect(LOCALES).toContain("en");
    expect(LOCALES).toContain("zh_cn");
  });

  describe.each(LOCALES)("%s", (locale) => {
    const obs = obsSection(locale);
    const english = obsSection("en");

    it.each(URL_KEYS)("%s gives the plain browser source URL, with no Curses and no code tag", (key) => {
      const value = obs[key]!;
      expect(value).toContain("http://localhost:PORT/client");
      expect(value).not.toContain("<code>");
      expect(value).not.toContain("</code>");
      expect(value).not.toMatch(/curses/i);
      // The list keeps its shape (one line per item) in every language.
      expect(lineCount(value)).toBe(lineCount(english[key]!));
    });

    it("stale_page_hint quotes the English on-screen text and the English OBS button, and is one paragraph", () => {
      const hint = obs.stale_page_hint!;
      expect(hint).toContain("connecting...");
      expect(hint).toContain("Refresh cache of current page");
      expect(hint).not.toContain("<code>");
      expect(hint).not.toMatch(/curses/i);
      expect(hint).not.toContain("\n");
      expect(hint.length).toBeGreaterThan(80);
    });
  });

  it("zh mirrors zh_cn for these keys", () => {
    for (const key of [...URL_KEYS, "stale_page_hint"]) expect(obsSection("zh")[key]).toBe(obsSection("zh_cn")[key]);
  });
});

describe("the credit to mmpneo's Curses", () => {
  // It had been translated as a common word ("Flüche", "Maldiciones", ...) in most locales, so the
  // credit no longer named the project it credits. The name stays Curses in every language.
  it.each(LOCALES)("%s names Curses in the About credit and its GitHub link", (locale) => {
    const settings = JSON.parse(fs.readFileSync(path.join(I18N_DIR, locale, "translation.json"), "utf8")).settings;
    expect(settings.credits_mmpneo).toContain("Curses");
    expect(settings.credits_mmpneo).toContain("mmpneo");
    expect(settings.tooltip_curses_github_title).toContain("Curses");
    expect(settings.tooltip_curses_github_title).toContain("GitHub");
  });
});

describe("the manual", () => {
  const MANUAL_DIR = path.join(ROOT, "docs/manual");
  const mentions = fs
    .readdirSync(MANUAL_DIR)
    .filter((name) => name.endsWith(".md"))
    .flatMap((name) =>
      fs
        .readFileSync(path.join(MANUAL_DIR, name), "utf8")
        .split(/\r?\n/)
        .map((text, i) => ({ name, line: i + 1, text }))
        .filter(({ text }) => /curses/i.test(text.replace(FUNCTIONAL_NAMES, ""))),
    );

  it("names Curses only to say which old app left the service worker behind in an OBS source", () => {
    expect(mentions.map(({ name }) => name)).toEqual(["00-overview.md"]);
    expect(mentions[0]!.text).toMatch(/service worker/);
    expect(mentions[0]!.text).toContain("connecting...");
  });

  it("never tells a streamer that something is like, or follows, Curses", () => {
    const comparison = /(like|same as|matches|match|follows|following|as in|same pattern as)\s+(the\s+)?(original\s+)?curses/i;
    expect(mentions.filter(({ text }) => comparison.test(text))).toEqual([]);
  });

  it("is copied unchanged into the in-app help (public/manual)", () => {
    const read = (file: string) => fs.readFileSync(file, "utf8").replace(/\r\n/g, "\n");
    expect(read(path.join(ROOT, "public/manual/00-overview.md"))).toBe(read(path.join(MANUAL_DIR, "00-overview.md")));
  });
});
