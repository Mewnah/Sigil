/**
 * Server-side render tests of the Speech to Text inspector's tabs (`SttTabs.tsx`) and of the Rewrite tab
 * (`Rewrite.tsx`). Real English strings plus the staged ones (`sttTestI18n`), so a missing key shows up as the key.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => undefined) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { isActive: vi.fn(() => false), dismiss: vi.fn() }) }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { Service_Transform_Schema, Transform_Backends, type Transform_State } from "@/core/services/transform/schema";
import { ServiceNetworkState } from "@/types";
import { Rewrite } from "../Rewrite";
import { SttTabPanel, SttTabs, nextSttView, sttViewForSection } from "../SttTabs";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  i18n.use(initReactI18next); // components read the default instance through useTranslation
  await initSttTestI18n();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

/** A window with just the Rewrite service: its saved settings and its running state. */
function stubWindow(status: ServiceNetworkState, data: Partial<Transform_State> = {}) {
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { transform: proxy({ showActionButton: true, data: { ...Service_Transform_Schema.parse({}), ...data } }) } },
      transform: { serviceState: proxy({ status }), start: vi.fn(), stop: vi.fn() },
      patchService: vi.fn(),
    },
  });
}

const noop = () => {};

describe("sttViewForSection", () => {
  it("opens Rewrite for the rewrite section and Speech for anything else", () => {
    expect(sttViewForSection("rewrite")).toBe("rewrite");
    expect(sttViewForSection(undefined)).toBe("speech");
    expect(sttViewForSection("")).toBe("speech");
    expect(sttViewForSection("something-else")).toBe("speech");
  });
});

describe("SttTabs", () => {
  it("shows a Speech tab and a Rewrite tab as text, with the open one selected", () => {
    stubWindow(ServiceNetworkState.disconnected);
    const speech = html(<SttTabs view="speech" onChange={noop} />);
    expect(speech).toContain('role="tablist"');
    expect(speech).toMatch(/aria-selected="true"[^>]*>Speech</);
    expect(speech).toMatch(/aria-selected="false"[^>]*>Rewrite</);
    expect(speech).not.toMatch(/stt\.tab_/);

    const rewrite = html(<SttTabs view="rewrite" onChange={noop} />);
    expect(rewrite).toMatch(/aria-selected="false"[^>]*>Speech</);
    expect(rewrite).toMatch(/aria-selected="true"[^>]*>Rewrite</);
  });

  it("has no dot while Rewrite is off", () => {
    stubWindow(ServiceNetworkState.disconnected);
    const out = html(<SttTabs view="speech" onChange={noop} />);
    expect(out).not.toContain("bg-success");
    expect(out).not.toContain("bg-warning");
  });

  it("shows a green dot while Rewrite runs and an amber one while it starts", () => {
    stubWindow(ServiceNetworkState.connected);
    const running = html(<SttTabs view="speech" onChange={noop} />);
    expect(running).toContain("bg-success");
    expect(running).not.toContain("bg-warning");
    expect(running).toContain("Connected");

    stubWindow(ServiceNetworkState.connecting);
    const starting = html(<SttTabs view="speech" onChange={noop} />);
    expect(starting).toContain("bg-warning");
    expect(starting).not.toContain("bg-success");
  });

  it("does not show a dot for an error state", () => {
    stubWindow(ServiceNetworkState.error);
    const out = html(<SttTabs view="speech" onChange={noop} />);
    expect(out).not.toContain("bg-success");
    expect(out).not.toContain("bg-warning");
  });
});

describe("SttTabs keyboard and screen reader support", () => {
  it("only the open tab is a tab stop, and both tabs name the panel they control", () => {
    stubWindow(ServiceNetworkState.disconnected);
    const speech = html(<SttTabs view="speech" onChange={noop} />);
    expect(speech).toMatch(/id="stt-tab-speech"[^>]*tabindex="0"|tabindex="0"[^>]*id="stt-tab-speech"/);
    expect(speech).toMatch(/id="stt-tab-rewrite"[^>]*tabindex="-1"|tabindex="-1"[^>]*id="stt-tab-rewrite"/);
    expect(speech.match(/aria-controls="stt-tabpanel"/g)).toHaveLength(2);

    const rewrite = html(<SttTabs view="rewrite" onChange={noop} />);
    expect(rewrite).toMatch(/id="stt-tab-speech"[^>]*tabindex="-1"|tabindex="-1"[^>]*id="stt-tab-speech"/);
    expect(rewrite).toMatch(/id="stt-tab-rewrite"[^>]*tabindex="0"|tabindex="0"[^>]*id="stt-tab-rewrite"/);
  });

  it("the panel is labelled by the open tab", () => {
    expect(html(<SttTabPanel view="speech">x</SttTabPanel>)).toBe('<div role="tabpanel" id="stt-tabpanel" aria-labelledby="stt-tab-speech">x</div>');
    expect(html(<SttTabPanel view="rewrite">x</SttTabPanel>)).toContain('aria-labelledby="stt-tab-rewrite"');
  });

  it("the arrow keys, Home and End move between the tabs (either arrow goes to the other tab, so right-to-left needs no mirror)", () => {
    expect(nextSttView("speech", "ArrowRight")).toBe("rewrite");
    expect(nextSttView("rewrite", "ArrowRight")).toBe("speech"); // wraps
    expect(nextSttView("rewrite", "ArrowLeft")).toBe("speech");
    expect(nextSttView("speech", "ArrowLeft")).toBe("rewrite");
    expect(nextSttView("rewrite", "Home")).toBe("speech");
    expect(nextSttView("speech", "End")).toBe("rewrite");
    expect(nextSttView("speech", "Enter")).toBeNull();
    expect(nextSttView("speech", "ArrowDown")).toBeNull();
  });
});

describe("Rewrite tab", () => {
  const engineLabel = ">Engine<";

  it("leads with what it does, then provider, key, model, instructions and Start; the rest is under More options", () => {
    vi.stubEnv("DEV", false);
    stubWindow(ServiceNetworkState.disconnected);
    const out = html(<Rewrite />);
    const at = (s: string) => {
      const i = out.indexOf(s);
      expect(i, `"${s}" in the Rewrite tab`).toBeGreaterThanOrEqual(0);
      return i;
    };

    expect(out).toContain("Rewrites each finished sentence with a language model, using your instructions");
    const order = [at("Rewrites each finished sentence"), at(">Provider<"), at("API key"), at(">Instructions<"), at(">Start rewriting<"), at("More options")];
    expect(order).toEqual([...order].sort((a, b) => a - b));

    // The four toggles come after Start and before the fold; everything else is inside it.
    for (const toggle of [">Add to action bar<", ">Start automatically<", ">Stop with Twitch stream<", ">Enable context history ✱<"]) {
      expect(at(toggle), toggle).toBeGreaterThan(at(">Start rewriting<"));
      expect(at(toggle), toggle).toBeLessThan(at("More options"));
    }
    for (const label of [">Max output tokens<", ">Reasoning effort<", ">Stream the reply<", ">Timeout per line"]) {
      expect(out.indexOf(label), label).toBeGreaterThan(at("More options"));
    }
    expect(out.indexOf("Sends your previous sentences to the model")).toBeGreaterThan(at(">Start rewriting<"));
    expect(out.indexOf("Sends your previous sentences to the model")).toBeLessThan(at("More options"));
    expect(out).toMatch(/<details/);
    expect(out).not.toMatch(/<details[^>]*\bopen\b/);
  });

  it("hides the debug engine in a normal build", () => {
    vi.stubEnv("DEV", false);
    stubWindow(ServiceNetworkState.disconnected);
    const out = html(<Rewrite />);
    expect(out).not.toContain(engineLabel);
    expect(out).not.toContain("Debug (uppercase)");
  });

  it("shows the engine in a development build", () => {
    vi.stubEnv("DEV", true);
    stubWindow(ServiceNetworkState.disconnected);
    expect(html(<Rewrite />)).toContain(engineLabel);
  });

  it("lets a profile that is still on the debug engine switch back, and then has no language model fields", () => {
    vi.stubEnv("DEV", false);
    stubWindow(ServiceNetworkState.disconnected, { backend: Transform_Backends.dummy });
    const out = html(<Rewrite />);
    expect(out).toContain(engineLabel);
    expect(out).toContain(">Start rewriting<");
    expect(out).not.toContain(">Provider<");
    expect(out).not.toContain("More options");
  });

  it("offers Stop while running, in the same words", () => {
    stubWindow(ServiceNetworkState.connected);
    const out = html(<Rewrite />);
    expect(out).toContain(">Stop rewriting<");
    expect(out).not.toContain(">Start rewriting<");
  });

  it("says nothing about AI (Google AI Studio is a proper name and stays)", () => {
    stubWindow(ServiceNetworkState.disconnected);
    const labels = (provider: Transform_State["provider"]) => {
      stubWindow(ServiceNetworkState.disconnected, { provider });
      return html(<Rewrite />).replace(/Google AI Studio/g, "");
    };
    for (const provider of ["openrouter", "gemini", "openai", "custom"] as const) {
      const out = labels(provider);
      expect(out, provider).not.toMatch(/\bAI\b/);
      expect(out, provider).not.toMatch(/\bLLM\b/);
      expect(out, provider).not.toMatch(/transform\.[a-z_]+/);
    }
    // The language model's engine option is no longer an acronym.
    expect(i18n.t("transform.engine_llm")).toBe("Language model");
  });
});
