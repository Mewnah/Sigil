/**
 * Server-side render tests of the "This PC only" hints (a new install starts there, `services/state/firstRun.ts`): the hint
 * component itself (`lanAccessHints.tsx`), where it sits (the OBS inspector next to its second-PC link, the Link apps section of
 * Settings), and the lines next to the LAN access switch in Settings. The English strings of the new hints are staged in
 * `scripts/i18n-staging/B-local-default.json` until the merge, so they are read from there on top of
 * `public/i18n/en/translation.json`.
 */
import fs from "node:fs";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
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
// The OBS inspector reads the canvas size from the document; the document service itself is not needed here.
vi.mock("@/client", () => ({ useGetState: (select: (state: unknown) => unknown) => select({ canvas: { w: 1920, h: 1080 }, author: "" }) }));

import { isTauri } from "@tauri-apps/api/core";
import { Service_OBS_Schema } from "@/core/services/obs/schema";
import { ServiceNetworkState } from "@/types";
import { LanOffHint } from "../lanAccessHints";
import { LanAccessSection, LinkAppsSection } from "../inspector_settings";
import Inspector_OBS from "../inspector_obs";

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

/** English as the app has it: the strings that are merged already, plus this workstream's staged ones (a no-op once merged). */
function english(): Json {
  const staged = path.join(ROOT, "scripts", "i18n-staging", "B-local-default.json");
  return merge(en as Json, fs.existsSync(staged) ? (JSON.parse(fs.readFileSync(staged, "utf8")) as Json) : {});
}

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

/** The host-side globals the components read: the settings proxy, the pubsub web-server status, the network info. */
function mount(mode: "lan" | "localhost", token = "0123456789abcdefghjkmnpqrs") {
  const state = proxy({ lanAccess: { mode, token, legacyNoToken: false }, linkAddress: "", services: { obs: { data: Service_OBS_Schema.parse({}) } } });
  vi.stubGlobal("window", {
    ApiServer: {
      state,
      obs: { wsState: proxy({ status: ServiceNetworkState.disconnected }), getObsBrowserSourceLink: vi.fn(() => "http://localhost:3030/browser") },
      patchService: vi.fn(),
    },
    ApiShared: {
      pubsub: { webServer: proxy({ status: undefined }), serviceState: proxy({ state: ServiceNetworkState.disconnected }), regenerateLanToken: vi.fn() },
      peer: { getLanClientLink: vi.fn(() => "") },
    },
    Config: { serverNetwork: { ip: "192.168.1.50", port: "3030", lanAddresses: [] } },
  });
  return state;
}

const HINT = "To use a phone or a second PC, set Other devices to Allowed with the LAN token (Settings, LAN access).";

beforeAll(async () => {
  await i18n.use(initReactI18next).init({ lng: "en", resources: { en: { translation: english() } }, interpolation: { escapeValue: false } });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(isTauri).mockReturnValue(true);
});

describe("LanOffHint (OBS inspector, Link apps)", () => {
  it("names the switch and its value the way Settings labels them, as a next step rather than a fault", () => {
    mount("localhost");
    const out = html(<LanOffHint />);
    expect(out).toContain(HINT);
    expect(out).not.toMatch(/settings\.(lan_off_hint|field_lan_mode|lan_mode_lan|section_lan_access)/);
  });

  it("shows nothing once LAN access is on", () => {
    mount("lan");
    expect(html(<LanOffHint />)).toBe("");
  });

  it("follows the switch while the page is open", () => {
    const state = mount("lan");
    expect(html(<LanOffHint />)).toBe("");
    state.lanAccess.mode = "localhost";
    expect(html(<LanOffHint />)).toContain(HINT);
  });
});

describe("where the hint sits", () => {
  it("OBS inspector, This PC only: the hint is there and the LAN browser source link is not", () => {
    mount("localhost");
    const out = html(<Inspector_OBS />);
    expect(out).toContain("Copy browser source URL"); // the link for OBS on this PC is always offered
    expect(out).not.toContain("Copy LAN link (OBS on another PC)");
    expect(out).toContain(HINT);
  });

  it("OBS inspector, LAN access on: the LAN browser source link is there and the hint is not", () => {
    mount("lan");
    const out = html(<Inspector_OBS />);
    expect(out).toContain("Copy browser source URL");
    expect(out).toContain("Copy LAN link (OBS on another PC)");
    expect(out).not.toContain("To use a phone or a second PC");
  });

  it("Link apps, This PC only: the hint follows the Copy my address button", () => {
    mount("localhost");
    const out = html(<LinkAppsSection />);
    expect(out).toContain("Link apps");
    const button = out.indexOf("Copy my address");
    expect(button).toBeGreaterThan(-1);
    expect(out.indexOf(HINT)).toBeGreaterThan(button);
  });

  it("Link apps, LAN access on: no hint", () => {
    mount("lan");
    const out = html(<LinkAppsSection />);
    expect(out).toContain("Copy my address");
    expect(out).not.toContain("To use a phone or a second PC");
  });

  it("Link apps in the browser build (no Tauri): no hint, there is no LAN access setting to point to", () => {
    mount("localhost");
    vi.mocked(isTauri).mockReturnValue(false);
    const out = html(<LinkAppsSection />);
    expect(out).toContain("Copy my address");
    expect(out).not.toContain("To use a phone or a second PC");
  });
});

describe("LAN access section of Settings", () => {
  it("in This PC only: says what the mode means, how to allow other devices, and that Windows may ask once", () => {
    mount("localhost");
    const out = html(<LanAccessSection />);
    expect(out).toContain("Sigil only listens on this PC.");
    expect(out).toContain("To allow a phone, a second PC or Link apps, choose Allowed with the LAN token above.");
    expect(out).toContain("Windows may ask once to allow Sigil on the network when LAN access is turned on.");
    // The token and the links for other devices only exist while LAN access is on.
    expect(out).not.toContain("Copy LAN canvas link");
    expect(out).not.toMatch(/settings\.lan_mode_(localhost_turn_on|firewall_note)/);
  });

  it("with LAN access on: no turn-on line and no Windows note, but the token and the links", () => {
    mount("lan");
    const out = html(<LanAccessSection />);
    expect(out).not.toContain("To allow a phone");
    expect(out).not.toContain("Windows may ask once");
    expect(out).toContain("Copy LAN canvas link");
    expect(out).toContain("Phones, a second PC with OBS and Link apps on your network can connect");
  });
});
