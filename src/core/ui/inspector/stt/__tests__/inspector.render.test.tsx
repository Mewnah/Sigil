/**
 * Server-side render of the Speech to Text inspector as a whole: which tab it opens on (the store's section), and
 * that the grouped engine list is what the Speech tab offers. Real English strings plus the staged ones.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

// The app shell the inspector sits in: desktop or web, toggled per test.
const runtime = vi.hoisted(() => ({ tauri: true }));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => runtime.tauri, invoke: vi.fn(async () => undefined) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { isActive: vi.fn(() => false), dismiss: vi.fn() }) }));
// The engines' own panels have their own tests; this one checks the inspector around them.
vi.mock("../Azure", () => ({ Azure: () => <div data-testid="engine-panel">azure panel</div> }));
vi.mock("../LanguageOffer", () => ({ LanguageOfferPanel: () => null }));
vi.mock("@/core/services/readings/ReadingsSection", () => ({ ReadingsSection: () => null }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { Service_STT_Schema, STT_Backends } from "@/core/services/stt/schema";
import { Services } from "@/services-registry";
import { ServiceNetworkState } from "@/types";
import { useAppUIStore } from "@/core/ui/store";
import Inspector_STT from "..";

beforeAll(async () => {
  i18n.use(initReactI18next);
  await initSttTestI18n();
});

const closed = { tab: undefined, show: false, expand: false, selections: [] as string[] };

afterEach(() => {
  runtime.tauri = true;
  vi.unstubAllGlobals();
  useAppUIStore.setState({ sidebar: { ...closed } });
  useAppUIStore.getInitialState().sidebar = { ...closed };
});

function html(backend: STT_Backends = STT_Backends.azure): string {
  vi.stubGlobal("window", {
    ApiServer: {
      state: proxy({
        shortcuts: { start: "", muteMic: "" },
        services: { stt: { showActionButton: true, data: { ...Service_STT_Schema.parse({}), backend } } },
      }),
      keyboard: { ui: proxy({ showRecorder: false, currentTarget: "", currentValue: "", recorderHint: "" }) },
      stt: { serviceState: proxy({ status: ServiceNetworkState.disconnected, statusText: "", progress: null, statusLevel: "info" }) },
      transform: { serviceState: proxy({ status: ServiceNetworkState.disconnected }) },
    },
  });
  return renderToStaticMarkup(<Inspector_STT />).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

/** The inspector opened by `changeTab`. Server rendering reads zustand's initial state, so the test sets that too. */
const open = (value?: string) => {
  const sidebar = { tab: { tab: Services.stt, value }, show: true, expand: false, selections: [] as string[] };
  useAppUIStore.setState({ sidebar });
  useAppUIStore.getInitialState().sidebar = sidebar;
};

describe("Speech to Text inspector", () => {
  it("opens on the Speech tab by default, with the engine selector and the engine's own panel", () => {
    open();
    const out = html();
    expect(out).toMatch(/aria-selected="true"[^>]*>Speech</);
    expect(out).toMatch(/aria-selected="false"[^>]*>Rewrite</);
    expect(out).toContain('role="combobox"');
    expect(out).toContain("azure panel");
    expect(out).not.toContain(">Start rewriting<");
  });

  it("opens on the Rewrite tab when the store asks for the rewrite section (the bottom-bar card)", () => {
    open("rewrite");
    const out = html();
    expect(out).toMatch(/aria-selected="false"[^>]*>Speech</);
    expect(out).toMatch(/aria-selected="true"[^>]*>Rewrite</);
    // The Speech tab's content is not rendered behind it.
    expect(out).not.toContain(">Speech to Text engine<");
    expect(out).not.toContain('role="combobox"');
    expect(out).not.toContain("azure panel");
  });

  it("opens on Speech for a section it does not know", () => {
    open("something-else");
    expect(html()).toMatch(/aria-selected="true"[^>]*>Speech</);
  });

  it("puts the open tab's content in a tab panel labelled by that tab", () => {
    open();
    expect(html()).toContain('role="tabpanel" id="stt-tabpanel" aria-labelledby="stt-tab-speech"');
    open("rewrite");
    expect(html()).toContain('role="tabpanel" id="stt-tabpanel" aria-labelledby="stt-tab-rewrite"');
  });

  it("keeps the header and the tab strip on both tabs", () => {
    for (const value of [undefined, "rewrite"]) {
      open(value);
      const out = html();
      expect(out).toContain(i18n.t("stt.title"));
      expect(out).toContain('role="tablist"');
    }
  });
});
