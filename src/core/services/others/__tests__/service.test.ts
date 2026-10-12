import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => undefined), isTauri: () => false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warn: vi.fn(), warning: vi.fn(), success: vi.fn(), dismiss: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { toast } from "react-toastify";
import { pushSystemLog } from "@/core/services/systemLog";
import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import type { TranslationEngineMap } from "@/core/services/translation";
import { Service_Translation_Schema, Translation_Backends } from "@/core/services/translation/schema";
import type { ITranslationEngine, TranslationEngineHost } from "@/core/services/translation/types";
import Service_Others from "../index";
import { Service_Others_Schema } from "../schema";

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

beforeAll(async () => {
  await initSttTestI18n();
});

beforeEach(() => {
  vi.mocked(pushSystemLog).mockClear();
  vi.mocked(toast.error).mockClear();
  vi.stubGlobal("window", {
    Config: { isServer: () => true, isApp: () => false, isClient: () => false, serverNetwork: {} },
    ApiShared: { pubsub: { publishText: vi.fn() }, peer: { broadcastPubSubParts: vi.fn() } },
    ApiServer: {
      state: {
        services: {
          others: { data: Service_Others_Schema.parse({}) },
          translation: { data: Service_Translation_Schema.parse({}) },
          stt: { data: { local: { language: "auto" } } },
        },
      },
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Service_Others", () => {
  it("reports a driver failure once in the system log, plus a toast", async () => {
    // Off the desktop app the driver fails at once.
    const service = new Service_Others(undefined, { available: () => false });
    service.start();
    await settle();
    expect(service.serviceState.phase).toBe("idle");
    const error = service.serviceState.error;
    expect(error).not.toBe("");
    expect(vi.mocked(pushSystemLog).mock.calls).toEqual([["Other players", error, "warning"]]);
    expect(toast.error).toHaveBeenCalledTimes(1);
    service.dispose();
  });
});

describe("Service_Others reverse translation", () => {
  it("gives its translation engine the model store's view, so a translator whose model is missing is held (not retried every minute)", async () => {
    const hosts: TranslationEngineHost[] = [];
    class HostEngine implements ITranslationEngine {
      attach(host: TranslationEngineHost) {
        hosts.push(host);
      }
      async start() {}
      async translate(request: { text: string }) {
        return request.text;
      }
      stop() {}
    }
    const engines = Object.fromEntries(Object.values(Translation_Backends).map((b) => [b, HostEngine])) as unknown as TranslationEngineMap;
    const othersData = proxy(Service_Others_Schema.parse({}));
    const win = window as unknown as { ApiServer: { state: { services: { others: { data: unknown } } }; models?: unknown } };
    win.ApiServer.state.services.others.data = othersData;
    const service = new Service_Others(engines, { available: () => false });
    service.init();
    service.serviceState.phase = "running"; // reverse translation follows the switch while the capture runs
    othersData.translate = true;
    await settle();
    expect(hosts).toHaveLength(1);
    const modelInstalled = hosts[0].modelInstalled;
    expect(modelInstalled).toBeTypeOf("function");
    const states = (state: string) => ({ state: { states: { "mt/nllb": { state } } } });
    win.ApiServer.models = states("not_installed");
    expect(modelInstalled?.("mt/nllb")).toBe(false);
    win.ApiServer.models = states("installed");
    expect(modelInstalled?.("mt/nllb")).toBe(true);
    service.serviceState.phase = "idle";
    service.dispose();
  });
});

