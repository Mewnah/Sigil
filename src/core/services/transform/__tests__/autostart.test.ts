/**
 * The launch auto start runs during host service init, before ApiServer.init awaits initI18n. This
 * file deliberately leaves i18next uninitialised until the test initialises it.
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
import Service_Transform, { TransformEngineMap } from "../index";
import { Service_Transform_Schema, Transform_Backends } from "../schema";
import { ITransformEngine } from "../types";

const starts = vi.fn<() => void>();

class BrokenEngine implements ITransformEngine {
  async start() {
    starts();
    throw new Error("server refused");
  }
  async transform(): Promise<string> {
    throw new Error("unused");
  }
  stop() {}
}

const engines = Object.fromEntries(
  Object.values(Transform_Backends).map((b) => [b, BrokenEngine]),
) as unknown as TransformEngineMap;

const flush = () => new Promise<void>((r) => setTimeout(r, 0));

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_Transform auto start", () => {
  it("waits for i18n, so a start that fails at launch is explained", async () => {
    expect(i18n.isInitialized).toBeFalsy();
    const data = proxy(Service_Transform_Schema.parse({}));
    data.autoStart = true;
    vi.stubGlobal("window", {
      ApiServer: { state: { services: { transform: { data } } }, patchService: vi.fn() },
      ApiShared: {
        pubsub: { subscribe: vi.fn(() => "sub"), subscribeText: vi.fn(() => "sub"), unsubscribe: vi.fn(), publishText: vi.fn() },
      },
    });

    const service = new Service_Transform(engines);
    service.init();
    await flush();
    expect(starts).not.toHaveBeenCalled();

    await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
    await flush();

    expect(starts).toHaveBeenCalledTimes(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    // Before the fix i18n.t returned undefined here and the failure was silent.
    expect(service.serviceState.error).toBe("transform.error_start_failed");
    expect(vi.mocked(toast)).toHaveBeenCalledWith("transform.error_start_failed", expect.objectContaining({ type: "error" }));
  });
});
