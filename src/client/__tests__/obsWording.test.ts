import fs from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

/**
 * What a streamer reads about OBS and the /client browser source says Sigil, not Curses. Curses stays
 * where it is a credit (the About links) or a name that still works (the .cursestmp import, the legacy
 * CURSES_* environment variables in developer hints).
 */

vi.mock("i18next", () => ({ default: { t: (key: string) => key, language: "en" } }));

import { StartupErrorView } from "../ui/clientStartup";

const ROOT = path.resolve(__dirname, "../../..");
const en = JSON.parse(fs.readFileSync(path.join(ROOT, "public/i18n/en/translation.json"), "utf8")) as Record<string, unknown>;

function strings(node: unknown, prefix = "", out: Array<[string, string]> = []): Array<[string, string]> {
  if (typeof node === "string") out.push([prefix, node]);
  else if (node && typeof node === "object") for (const [key, value] of Object.entries(node)) strings(value, prefix ? `${prefix}.${key}` : key, out);
  return out;
}

const get = (key: string): string => {
  const hit = strings(en).find(([k]) => k === key);
  if (!hit) throw new Error(`missing en key ${key}`);
  return hit[1];
};

/** The names that must keep working (the Curses backup file, the legacy variables), removed before looking for a comparison to Curses. */
const FUNCTIONAL_NAMES = /(?:Curses )?\.cursestmp|CURSES_[A-Z0-9_]*\*?/g;

/** Where the word may stay: the credit to mmpneo's Curses in Settings. */
const CREDITS = ["settings.credits_mmpneo", "settings.tooltip_curses_github_title"];

describe("OBS and client wording", () => {
  it("has no Curses in the obs and client sections", () => {
    const offenders = strings(en)
      .filter(([key]) => key.startsWith("obs.") || key.startsWith("client."))
      .filter(([, value]) => /curses/i.test(value))
      .map(([key]) => key);
    expect(offenders).toEqual([]);
  });

  it("mentions Curses in no other English string except the credits and the functional names", () => {
    const offenders = strings(en)
      .filter(([, value]) => /curses/i.test(value.replace(FUNCTIONAL_NAMES, "")))
      .map(([key]) => key);
    expect(offenders.sort()).toEqual([...CREDITS].sort());
  });

  it("keeps the credit to mmpneo's Curses", () => {
    expect(get("settings.credits_mmpneo")).toContain("Curses");
    expect(get("settings.tooltip_curses_github_title")).toContain("Curses");
  });

  it("keeps the .cursestmp import and the legacy CURSES_ environment names", () => {
    expect(get("project.template_live_copy")).toContain(".cursestmp");
    expect(get("integrations.twitch.missing_client_id")).toContain("CURSES_TWITCH_CLIENT_ID");
    expect(get("integrations.kick.missing_client_id")).toContain("CURSES_KICK_CLIENT_ID");
  });

  it("still gives the OBS checklist its browser source URL", () => {
    expect(get("obs.checklist_browser_mirror")).toContain("http://localhost:PORT/client");
    expect(get("obs.browser_source_tips")).toContain("http://localhost:PORT/client");
  });

  it("tells a streamer what to do when OBS shows an old page or keeps saying connecting...", () => {
    const hint = get("obs.stale_page_hint");
    // The old page is not translated: it says this, in lowercase English, whatever the language of Sigil.
    expect(hint).toContain("connecting...");
    expect(hint).toContain("Refresh cache of current page");
    // The clean-up runs when the source next loads, so the hint must not promise a one-off manual step.
    expect(hint).not.toMatch(/only once|curses/i);
    const inspector = fs.readFileSync(path.join(ROOT, "src/core/ui/inspector/inspector_obs.tsx"), "utf8");
    expect(inspector).toContain('t("obs.stale_page_hint")');
  });

  it("start-up error panel for the browser source names Sigil, not Curses", () => {
    const html = renderToStaticMarkup(
      createElement(StartupErrorView, { error: new Error("boom"), client: { host: "localhost", port: "3030" }, retryInSeconds: 4 }),
    );
    expect(html).not.toMatch(/curses/i);
    expect(html).toContain("The OBS browser source could not connect to Sigil");
    expect(html).toContain("http://localhost:3030/client");
  });

  it("the /client loading view is Sigil-branded, not a Curses page", () => {
    const loading = fs.readFileSync(path.join(ROOT, "src/client/ui/view_loading.tsx"), "utf8");
    expect(loading).not.toMatch(/curses/i);
    expect(get("client.loading_connecting")).toBe("Connecting to Sigil…");
  });
});

