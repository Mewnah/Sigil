/**
 * The silent-microphone notice in the Speech to Text inspector ("Sigil hears nothing from <mic>"): it sits under the
 * Start button beside the engine status line, shows the device and the Windows setting to check, and is gone when
 * the service has no silence to report. Real English strings plus the staged ones.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => undefined) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { isActive: vi.fn(() => false), dismiss: vi.fn() }) }));
// The engines' own panels have their own tests; this one checks the status area around them.
vi.mock("../Azure", () => ({ Azure: () => <div data-testid="engine-panel">azure panel</div> }));
vi.mock("../LanguageOffer", () => ({ LanguageOfferPanel: () => null }));
vi.mock("@/core/services/readings/ReadingsSection", () => ({ ReadingsSection: () => null }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { Service_STT_Schema, STT_Backends } from "@/core/services/stt/schema";
import { ServiceNetworkState } from "@/types";
import { useAppUIStore } from "@/core/ui/store";
import Inspector_STT from "..";

beforeAll(async () => {
  i18n.use(initReactI18next);
  await initSttTestI18n();
});

afterEach(() => {
  vi.unstubAllGlobals();
  useAppUIStore.setState({ sidebar: { tab: undefined, show: false, expand: false, selections: [] } });
});

function html(serviceState: { statusText?: string; progress?: number | null; silentInput?: { device: string } | null }): string {
  vi.stubGlobal("window", {
    ApiServer: {
      state: proxy({
        shortcuts: { start: "", muteMic: "" },
        services: { stt: { showActionButton: true, data: { ...Service_STT_Schema.parse({}), backend: STT_Backends.azure } } },
      }),
      keyboard: { ui: proxy({ showRecorder: false, currentTarget: "", currentValue: "", recorderHint: "" }) },
      stt: {
        serviceState: proxy({
          status: ServiceNetworkState.connected,
          statusText: "",
          progress: null as number | null,
          statusLevel: "info",
          silentInput: null as { device: string } | null,
          ...serviceState,
        }),
      },
      transform: { serviceState: proxy({ status: ServiceNetworkState.disconnected }) },
    },
  });
  return renderToStaticMarkup(<Inspector_STT />).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&amp;/g, "&");
}

describe("Speech to Text inspector: silent microphone", () => {
  it("shows what the microphone delivers when it is only silence: the device, and where to look", () => {
    const out = html({ silentInput: { device: "Headset Mic (USB)" } });
    expect(out).toContain('role="status"');
    expect(out).toContain("Sigil hears nothing from Headset Mic (USB).");
    expect(out).toContain("Settings > Privacy & security > Microphone > Let desktop apps access your microphone");
    expect(out).toMatch(/<span class="text-warning">Sigil hears nothing/);
  });

  it("does not make up a name when the capture could not name the microphone", () => {
    const out = html({ silentInput: { device: "" } });
    expect(out).toContain("Sigil hears nothing from your microphone.");
  });

  it("shows nothing about it for a service that hears its microphone", () => {
    expect(html({ silentInput: null })).not.toContain("hears nothing");
  });

  it("sits beside the engine status line instead of replacing it, and a progress bar belongs to the status line", () => {
    const out = html({ statusText: "Loading the model", progress: 40, silentInput: { device: "Headset Mic" } });
    const status = out.indexOf("Loading the model");
    const silence = out.indexOf("Sigil hears nothing from Headset Mic.");
    expect(status).toBeGreaterThan(-1);
    expect(silence).toBeGreaterThan(status);
    expect(out).toContain('<progress class="progress progress-primary w-full" value="40"');
    // And the progress bar is not drawn for a notice alone.
    expect(html({ silentInput: { device: "Headset Mic" } })).not.toContain("<progress");
  });
});
