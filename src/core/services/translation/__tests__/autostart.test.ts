/**
 * The launch auto start runs during host service init, before ApiServer.init awaits initI18n. This
 * file deliberately leaves i18next uninitialised until the test initialises it (service.test.ts
 * initialises it up front and cannot see this).
 */
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("../../systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { toast } from "react-toastify";
import Service_Translation, { TranslationEngineMap } from "../index";
import { Service_Translation_Schema, Translation_Backends } from "../schema";
import { ITranslationEngine } from "../types";

const starts = vi.fn<() => void>();

/** Fails like the on-device engine on stable WebView2: its message comes from i18n.t(). */
class UnavailableEngine implements ITranslationEngine {
  async start() {
    starts();
    throw new Error(i18n.t("transl.ondevice_error_unavailable"));
  }
  async translate(): Promise<string> {
    throw new Error("unused");
  }
  stop() {}
}

const engines = Object.fromEntries(
  Object.values(Translation_Backends).map((b) => [b, UnavailableEngine]),
) as unknown as TranslationEngineMap;

const flush = () => new Promise<void>((r) => setTimeout(r, 0));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_Translation auto start", () => {
  it("waits for i18n, so a start that fails at launch is explained; a user stop meanwhile wins", async () => {
    expect(i18n.isInitialized).toBeFalsy();
    const data = proxy(Service_Translation_Schema.parse({}));
    data.autoStart = true;
    data.backend = Translation_Backends.onDevice;
    vi.stubGlobal("window", {
      ApiServer: { state: { services: { translation: { data } } } },
      ApiShared: { pubsub: { subscribeText: vi.fn(() => "sub"), unsubscribe: vi.fn(), publishText: vi.fn() } },
    });
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}")));

    const launched = new Service_Translation(engines);
    launched.init();
    const stoppedByUser = new Service_Translation(engines);
    stoppedByUser.init();
    stoppedByUser.stop();
    await flush();
    expect(starts).not.toHaveBeenCalled();

    await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
    await flush();

    expect(starts).toHaveBeenCalledTimes(1);
    expect(launched.serviceState.status).toBe(ServiceNetworkState.disconnected);
    // Before the fix the message was "" (i18n.t returned undefined) and nothing was shown at all.
    expect(launched.serviceState.error).toBe("transl.ondevice_error_unavailable");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transl.ondevice_error_unavailable", expect.objectContaining({ type: "error" }));
    expect(stoppedByUser.serviceState.error).toBe("");
  });
});
