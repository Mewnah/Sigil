/**
 * The Moonshine server backend was retired (the built-in engine runs Moonshine and better models inside Sigil).
 * `planRetiredBackendMigration` is the pure plan; `Service_STT` applies it at launch and at Start: a desktop profile
 * on it opens on the built-in engine (`local`) with its microphone and language, is told so, and never auto-starts
 * (a Start asks before any model download). The web build is untouched.
 */
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => ""), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), {
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
    dismiss: vi.fn(),
  }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ migrateLegacyNativeInputDevice: vi.fn(async () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("@/shared/services/pubsub", () => ({ PUBSUB_STT_EXTERNAL_COMMAND: "stt.external.command" }));
// Real engines pull in SDKs and Tauri APIs; the service is tested against a fake engine.
vi.mock("../services/azure", () => ({ STT_AzureService: class {} }));
vi.mock("../services/deepgram", () => ({ STT_DeepgramService: class {} }));
vi.mock("../services/native", () => ({ STT_NativeService: class {} }));
vi.mock("../services/whisper", () => ({ STT_WhisperService: class {} }));
vi.mock("../services/vosk", () => ({ STT_VoskService: class {} }));
vi.mock("../services/moonshine", () => ({ STT_MoonshineService: class {} }));
vi.mock("../services/openaiAudio", () => ({ STT_OpenAI_AudioService: class {} }));
vi.mock("../services/externalMic", () => ({ STT_ExternalMicService: class {} }));

import { ask } from "@tauri-apps/plugin-dialog";
import { toast } from "react-toastify";
import { pushSystemLog } from "@/core/services/systemLog";
import { useAppUIStore } from "@/core/ui/store";
import { Services } from "@/services-registry";
import Service_STT, { type SttBackendMap } from "../index";
import { planRetiredBackendMigration } from "../migrate";
import { Service_STT_Schema, STT_Backends, type STT_State } from "../schema";
import type { ISTTReceiver, ISTTService } from "../types";
import { initSttTestI18n } from "./sttTestI18n";

class FakeEngine implements ISTTService {
  static instances: FakeEngine[] = [];
  startAbortSignal: AbortSignal | null = null;
  started = 0;
  constructor(public receiver: ISTTReceiver) {
    FakeEngine.instances.push(this);
  }
  start(_params: STT_State) {
    this.started++;
  }
  stop() {}
  dispose() {}
}

const fakeMap = Object.fromEntries(Object.values(STT_Backends).map((b) => [b, FakeEngine])) as SttBackendMap;
const defaults = Service_STT_Schema.parse({});
let isApp = false;

function makeService(overrides: Partial<STT_State> = {}) {
  const data = proxy({ ...defaults, ...overrides });
  const state = { services: { stt: { data } }, onboardingComplete: true };
  vi.stubGlobal("window", {
    Config: { isApp: () => isApp },
    ApiServer: {
      state,
      patchService: (key: "stt", fn: (s: { data: STT_State }) => void) => fn(state.services[key]),
    },
    ApiShared: {
      pubsub: {
        publishText: vi.fn(),
        subscribe: vi.fn(() => "stream-sub"),
        unsubscribe: vi.fn(),
        broadcastLocalWebSocketOnly: vi.fn(),
      },
    },
  });
  const service = new Service_STT(fakeMap);
  service.updateReplacementsCache();
  return { service, data };
}

const flush = () => vi.advanceTimersByTimeAsync(0);

/** A profile on the retired Moonshine server with a chosen microphone and language. */
const moonshineProfile = (extra: Partial<STT_State> = {}): Partial<STT_State> => ({
  backend: STT_Backends.moonshine,
  desktopDefaultMigrated: true,
  moonshine: { ...defaults.moonshine, language: "ja", device: "Mic X" },
  ...extra,
});

describe("planRetiredBackendMigration", () => {
  it("moves a desktop profile on Moonshine to the built-in engine", () => {
    expect(planRetiredBackendMigration({ isDesktop: true, backend: STT_Backends.moonshine })).toEqual({ action: "local" });
  });

  it("leaves every other engine alone", () => {
    for (const backend of Object.values(STT_Backends).filter((b) => b !== STT_Backends.moonshine))
      expect(planRetiredBackendMigration({ isDesktop: true, backend })).toEqual({ action: "none" });
  });

  it("leaves the web build alone (it has no in-process engine to move to)", () => {
    expect(planRetiredBackendMigration({ isDesktop: false, backend: STT_Backends.moonshine })).toEqual({ action: "none" });
  });
});

describe("Service_STT retired Moonshine server backend", () => {
  beforeAll(async () => {
    await initSttTestI18n();
  });

  beforeEach(() => {
    vi.useFakeTimers();
    FakeEngine.instances = [];
    isApp = true;
    vi.stubGlobal("navigator", { hardwareConcurrency: 12 });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("opens a Moonshine profile on the built-in engine with its language and microphone, tells the user, and does not auto-start", async () => {
    const { service, data } = makeService(moonshineProfile({ autoStart: true }));
    await service.init();
    await flush();

    expect(data.backend).toBe(STT_Backends.local);
    expect(data.local.language).toBe("ja");
    expect(data.local.device).toBe("Mic X");
    // No download consent without a click: the launch does not start anything.
    expect(FakeEngine.instances).toHaveLength(0);

    expect(toast.info).toHaveBeenCalledTimes(1);
    const [message, options] = vi.mocked(toast.info).mock.calls[0];
    expect(message).toContain("Moonshine server mode was retired.");
    expect(message).toContain("Built-in (on your PC)");
    expect(message).toContain("The first Start asks before downloading models.");
    expect(message).not.toMatch(/^stt\./);
    expect(options).toMatchObject({ autoClose: 12_000 });
    expect(pushSystemLog).toHaveBeenCalledWith("STT", message, "info");
    // The one-time offer for Web Speech profiles is not involved.
    expect(ask).not.toHaveBeenCalled();
  });

  it("keeps a language and microphone the built-in engine already has", async () => {
    const { service, data } = makeService(
      moonshineProfile({ local: { ...defaults.local, language: "fr", device: "Other mic" } })
    );
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.local);
    expect(data.local.language).toBe("fr");
    expect(data.local.device).toBe("Other mic");
  });

  it("migrates once: the next launch starts normally and says nothing", async () => {
    const { service, data } = makeService(moonshineProfile({ autoStart: true }));
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.local);
    vi.mocked(toast.info).mockClear();

    await new Service_STT(fakeMap).init();
    await flush();
    expect(FakeEngine.instances).toHaveLength(1);
    expect(toast.info).not.toHaveBeenCalled();
  });

  it("leaves the web build on its saved backend", async () => {
    isApp = false;
    const { service, data } = makeService(moonshineProfile({ autoStart: true }));
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.moonshine);
    expect(toast.info).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("also catches a profile that reaches Moonshine after launch when Start is pressed", async () => {
    const { service, data } = makeService({ backend: STT_Backends.whisper, desktopDefaultMigrated: true });
    await service.init();
    await flush();
    data.backend = STT_Backends.moonshine;
    service.start();
    await flush();
    expect(data.backend).toBe(STT_Backends.local);
    expect(FakeEngine.instances).toHaveLength(1);
    expect(toast.info).toHaveBeenCalledTimes(1);

    // Idempotent: a second Start on the built-in engine adds no notice.
    service.start();
    await flush();
    expect(toast.info).toHaveBeenCalledTimes(1);
  });

  it("does nothing for a profile on another engine", async () => {
    const { service, data } = makeService({ backend: STT_Backends.whisper, desktopDefaultMigrated: true, autoStart: true });
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.whisper);
    expect(toast.info).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1);
  });
});

describe("Service_STT download progress while the STT inspector is open on the Rewrite tab", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeEngine.instances = [];
    isApp = false;
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    useAppUIStore.setState({ sidebar: { tab: undefined, show: false, expand: false, selections: [] } });
  });

  const progress = async (tab: { tab: Services; value?: string }) => {
    useAppUIStore.setState({ sidebar: { tab, show: true, expand: false, selections: [] } });
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    FakeEngine.instances[0].receiver.onStatus("Downloading: 40%", { progress: 40 });
  };

  it("toasts the progress, because the Rewrite tab does not show it", async () => {
    await progress({ tab: Services.stt, value: "rewrite" });
    expect(toast.info).toHaveBeenCalledTimes(1);
  });

  it("stays quiet on the Speech tab, which shows it inline", async () => {
    await progress({ tab: Services.stt });
    expect(toast.info).not.toHaveBeenCalled();
  });
});
