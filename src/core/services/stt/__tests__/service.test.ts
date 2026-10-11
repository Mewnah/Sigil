import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEventSource, TextEventType } from "@/types";

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

import { invoke } from "@tauri-apps/api/core";
import { ask } from "@tauri-apps/plugin-dialog";
import { toast } from "react-toastify";
import { useAppUIStore } from "@/core/ui/store";
import { Services } from "@/services-registry";
import Service_STT, { type SttBackendMap } from "../index";
import { Service_STT_Schema, STT_Backends, type STT_State } from "../schema";
import { SttMuteState, type ISTTReceiver, type ISTTService } from "../types";

class FakeEngine implements ISTTService {
  static instances: FakeEngine[] = [];
  static startImpl: (self: FakeEngine) => void | Promise<void> = () => {};
  startAbortSignal: AbortSignal | null = null;
  started = 0;
  stopped = 0;
  constructor(public receiver: ISTTReceiver) {
    FakeEngine.instances.push(this);
  }
  start(_params: STT_State) {
    this.started++;
    return FakeEngine.startImpl(this);
  }
  stop() {
    this.stopped++;
  }
  dispose() {}
}

const fakeMap = Object.fromEntries(Object.values(STT_Backends).map((b) => [b, FakeEngine])) as SttBackendMap;
const last = () => FakeEngine.instances[FakeEngine.instances.length - 1];

let publishText: ReturnType<typeof vi.fn>;
let isApp = false;

function makeService(overrides: Partial<STT_State> = {}, rootOverrides: Record<string, unknown> = {}) {
  const data = proxy({ ...Service_STT_Schema.parse({}), ...overrides });
  const state = { services: { stt: { data } }, onboardingComplete: true, ...rootOverrides };
  publishText = vi.fn();
  vi.stubGlobal("window", {
    Config: { isApp: () => isApp },
    ApiServer: {
      state,
      patchService: (key: "stt", fn: (s: { data: STT_State }) => void) => fn(state.services[key]),
    },
    ApiShared: {
      pubsub: {
        publishText,
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

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

beforeEach(() => {
  vi.useFakeTimers();
  FakeEngine.instances = [];
  FakeEngine.startImpl = () => {};
  isApp = false;
  vi.stubGlobal("navigator", { hardwareConcurrency: 12 });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  useAppUIStore.setState({ sidebar: { tab: undefined, show: false, expand: false, selections: [] } });
});

describe("Service_STT status channel", () => {
  it("shows engine status in serviceState and never publishes it", async () => {
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    const engine = last();
    engine.receiver.onStatus("Downloading ggml-small.en-q5_1.bin: 40%", { progress: 40 });
    engine.receiver.onStatus("Loading the Whisper model…");
    expect(service.serviceState.statusText).toBe("Loading the Whisper model…");
    expect(service.serviceState.progress).toBeNull();
    engine.receiver.onStatus("Downloading ggml-small.en-q5_1.bin: 60%", { progress: 60 });
    expect(service.serviceState.progress).toBe(60);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    await flush();
    expect(publishText).not.toHaveBeenCalled();

    engine.receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(service.serviceState.statusText).toBe("");
    expect(service.serviceState.progress).toBeNull();

    engine.receiver.onStatus("Rate limit: next request in 3 s", { level: "warning" });
    engine.receiver.onFinal("hello world");
    await flush();
    expect(publishText).toHaveBeenCalledTimes(1);
    expect(publishText).toHaveBeenCalledWith(TextEventSource.stt, { value: "hello world", type: TextEventType.final });
    expect(service.serviceState.statusLevel).toBe("warning");
  });

  it("clears status on stop and drops every report of a stopped session", async () => {
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    const engine = last();
    engine.receiver.onStart();
    engine.receiver.onStatus("working");
    service.stop();
    expect(service.serviceState.statusText).toBe("");
    engine.receiver.onFinal("late final");
    engine.receiver.onInterim("late interim");
    engine.receiver.onStatus("late status");
    await flush();
    expect(publishText).not.toHaveBeenCalled();
    expect(service.serviceState.statusText).toBe("");
    expect(engine.receiver.isMuted()).toBe(true);
  });

  it("status keeps the connecting watchdog alive during a long download, then times out when it goes quiet", async () => {
    const { service } = makeService({ backend: STT_Backends.openai_audio });
    service.start();
    await flush();
    const engine = last();
    for (let i = 0; i < 4; i++) {
      await vi.advanceTimersByTimeAsync(40_000);
      engine.receiver.onStatus(`Downloading: ${i * 25}%`, { progress: i * 25 });
    }
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    await vi.advanceTimersByTimeAsync(46_000);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(toast.warning).toHaveBeenCalled();
    expect(engine.stopped).toBe(1);
  });

  it("a transient warning clears itself; a newer status cancels the expiry", async () => {
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    const engine = last();
    engine.receiver.onStart();
    engine.receiver.onStatus("Some speech was skipped", { level: "warning", expireMs: 15_000 });
    await vi.advanceTimersByTimeAsync(14_000);
    expect(service.serviceState.statusText).toBe("Some speech was skipped");
    await vi.advanceTimersByTimeAsync(2_000);
    expect(service.serviceState.statusText).toBe("");
    expect(service.serviceState.statusLevel).toBe("info");

    engine.receiver.onStatus("Some speech was skipped", { level: "warning", expireMs: 15_000 });
    engine.receiver.onStatus("Rate limit: next request in 3 s", { level: "warning" });
    await vi.advanceTimersByTimeAsync(30_000);
    expect(service.serviceState.statusText).toBe("Rate limit: next request in 3 s");
  });

  it("shows download progress as a toast when the STT inspector is closed, and dismisses it when ready", async () => {
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    const engine = last();
    engine.receiver.onStatus("Loading…"); // no progress: no toast
    expect(toast.info).not.toHaveBeenCalled();
    engine.receiver.onStatus("Downloading ggml-small.en-q5_1.bin: 10%", { progress: 10 });
    expect(toast.info).toHaveBeenCalledTimes(1);
    expect(toast.info).toHaveBeenCalledWith(
      "Downloading ggml-small.en-q5_1.bin: 10%",
      expect.objectContaining({ toastId: "stt-model-progress", progress: 0.1, autoClose: false }),
    );
    engine.receiver.onStatus("Downloading ggml-small.en-q5_1.bin: 100%", { progress: 100 });
    expect(toast.info).toHaveBeenCalledTimes(1);
    // Never 1: a controlled toast would close itself before the model is ready.
    expect(toast.update).toHaveBeenLastCalledWith("stt-model-progress", { render: "Downloading ggml-small.en-q5_1.bin: 100%", progress: 0.99 });
    engine.receiver.onStart();
    expect(toast.dismiss).toHaveBeenCalledWith("stt-model-progress");
    expect(publishText).not.toHaveBeenCalled();
  });

  it("does not toast download progress while the STT inspector shows it", async () => {
    useAppUIStore.setState({ sidebar: { tab: { tab: Services.stt }, show: true, expand: false, selections: [] } });
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    last().receiver.onStatus("Downloading: 40%", { progress: 40 });
    expect(service.serviceState.progress).toBe(40);
    expect(toast.info).not.toHaveBeenCalled();
    service.stop();
    expect(toast.dismiss).not.toHaveBeenCalled();
  });

  it("an engine error stops the session with a toast and clears the status", async () => {
    const { service } = makeService({ backend: STT_Backends.openai_audio });
    service.start();
    await flush();
    const engine = last();
    engine.receiver.onStatus("Retrying…", { level: "warning" });
    engine.receiver.onStop("Groq: enter your API key");
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("Groq: enter your API key");
    expect(service.serviceState.statusText).toBe("");
    expect(toast).toHaveBeenCalledWith("Groq: enter your API key", expect.objectContaining({ type: "error" }));
  });
});

describe("Service_STT mute", () => {
  async function running() {
    const ctx = makeService({ backend: STT_Backends.vosk });
    ctx.service.start();
    await flush();
    last().receiver.onStart();
    return { ...ctx, engine: last() };
  }

  it("setMuted is idempotent and suppresses publishing; engines can ask", async () => {
    const { service, engine } = await running();
    service.setMuted(true);
    service.setMuted(true);
    expect(service.serviceState.muted).toBe(SttMuteState.muted);
    expect(engine.receiver.isMuted()).toBe(true);
    engine.receiver.onFinal("secret");
    await flush();
    expect(publishText).not.toHaveBeenCalled();
    service.setMuted(false);
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
    expect(engine.receiver.isMuted()).toBe(false);
    engine.receiver.onFinal("public");
    await flush();
    expect(publishText).toHaveBeenCalledWith(TextEventSource.stt, { value: "public", type: TextEventType.final });
  });

  it("muting mid-sentence closes the interim once; unmuting mid-interim waits for the sentence to end", async () => {
    const { service, engine } = await running();
    engine.receiver.onInterim("hello");
    await flush();
    service.setMuted(true);
    service.setMuted(true);
    await flush();
    const finals = publishText.mock.calls.filter(([, e]) => e.type === TextEventType.final);
    expect(finals).toHaveLength(1); // the "[...]" cancel marker, sent once

    engine.receiver.onInterim("still talking");
    await flush();
    service.setMuted(false);
    expect(service.serviceState.muted).toBe(SttMuteState.pendingUnmute);
    engine.receiver.onFinal("still talking done");
    await flush();
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
    expect(publishText.mock.calls.some(([, e]) => e.value.startsWith("still talking"))).toBe(false);
  });

  it("toggleMute from a pending unmute unmutes at once", async () => {
    const { service, engine } = await running();
    service.setMuted(true);
    engine.receiver.onInterim("abc");
    await flush();
    service.toggleMute();
    expect(service.serviceState.muted).toBe(SttMuteState.pendingUnmute);
    service.toggleMute();
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
  });

  it("a pending unmute does not make engines skip work; an abandoned or final-only sentence resolves it", async () => {
    const { service, engine } = await running();
    service.setMuted(true);
    engine.receiver.onInterim("muted words");
    await flush();
    service.setMuted(false);
    expect(service.serviceState.muted).toBe(SttMuteState.pendingUnmute);
    // Engines (e.g. HTTP utterance capture) keep sending: only a hard mute skips work.
    expect(engine.receiver.isMuted()).toBe(false);
    // The engine drops the open hypothesis without a final: nothing is left to wait for.
    engine.receiver.onInterim("");
    await flush();
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
    engine.receiver.onFinal("spoken after unmute");
    await flush();
    expect(publishText).toHaveBeenCalledWith(TextEventSource.stt, { value: "spoken after unmute", type: TextEventType.final });
  });

  it("an engine that stops mid-sentence closes the open interim like a user stop", async () => {
    const { service } = makeService({ backend: STT_Backends.deepgram });
    service.start();
    await flush();
    const deepgram = last();
    deepgram.receiver.onStart();
    deepgram.receiver.onInterim("half a sent");
    await flush();
    deepgram.receiver.onStop("[Deepgram] WebSocket error");
    await flush();
    expect(publishText).toHaveBeenLastCalledWith(TextEventSource.stt, { value: "[...]", type: TextEventType.final });
  });

  it("a muted engine that stops mid-sentence leaves no pending unmute for a finals-only engine", async () => {
    const { service, data } = makeService({ backend: STT_Backends.deepgram });
    service.start();
    await flush();
    const deepgram = last();
    deepgram.receiver.onStart();
    service.setMuted(true);
    deepgram.receiver.onInterim("muted words"); // suppressed, but a sentence is open
    await flush();
    deepgram.receiver.onStop("[Deepgram] WebSocket error");
    await flush();

    // Switch to Groq (finals only, no interims) and unmute with the shortcut: it must not stay pending forever.
    data.backend = STT_Backends.openai_audio;
    service.start();
    await flush();
    const groq = last();
    groq.receiver.onStart();
    service.toggleMute();
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
    expect(groq.receiver.isMuted()).toBe(false);
    groq.receiver.onFinal("hello again");
    await flush();
    expect(publishText).toHaveBeenLastCalledWith(TextEventSource.stt, { value: "hello again", type: TextEventType.final });
    expect(publishText.mock.calls.some(([, e]) => e.value.includes("muted words"))).toBe(false);
  });

  it("a new start resolves an unmute left pending by the previous session", async () => {
    const { service, engine } = await running();
    service.setMuted(true);
    engine.receiver.onInterim("abc");
    await flush();
    service.setMuted(false);
    expect(service.serviceState.muted).toBe(SttMuteState.pendingUnmute);
    service.stop();
    service.start();
    await flush();
    expect(service.serviceState.muted).toBe(SttMuteState.unmuted);
  });

  it("toggleRunning starts when stopped and stops when starting or running", async () => {
    const { service } = makeService({ backend: STT_Backends.vosk });
    service.toggleRunning();
    await flush();
    expect(FakeEngine.instances).toHaveLength(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    service.toggleRunning();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    await flush();
    expect(last().stopped).toBe(1);
  });
});

describe("Service_STT desktop default migration", () => {
  it("offers local to an existing profile on native once; no moves it to Whisper without auto-starting", async () => {
    isApp = true;
    const { service, data } = makeService({ autoStart: true });
    expect(data.backend).toBe(STT_Backends.native);
    await service.init();
    await flush();
    // The offer is pending until the startup offers run (after i18n); nothing starts meanwhile.
    expect(data.backend).toBe(STT_Backends.native);
    expect(FakeEngine.instances).toHaveLength(0);
    vi.mocked(ask).mockResolvedValueOnce(false);
    await service.runStartupOffer();
    await flush();
    expect(ask).toHaveBeenCalledTimes(1);
    expect(data.localOfferShown).toBe(true);
    expect(data.backend).toBe(STT_Backends.whisper);
    expect(data.whisper.model).toBe("small.en-q5_1");
    expect(data.desktopDefaultMigrated).toBe(true);
    expect(FakeEngine.instances).toHaveLength(0);
    expect(toast.info).toHaveBeenCalledTimes(1);

    // A later deliberate choice of native is kept (and never auto-started)...
    data.backend = STT_Backends.native;
    await new Service_STT(fakeMap).init();
    await flush();
    expect(data.backend).toBe(STT_Backends.native);
    expect(FakeEngine.instances).toHaveLength(0);
    // ...and auto-start works again on later launches.
    data.backend = STT_Backends.whisper;
    await new Service_STT(fakeMap).init();
    await flush();
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("keeps native in the web build", async () => {
    isApp = false;
    const { service, data } = makeService();
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.native);
    expect(data.desktopDefaultMigrated).toBe(false);
  });

  it("switches an existing profile that accepts the offer to local, carrying its language over", async () => {
    isApp = true;
    const { service, data } = makeService({ native: { ...Service_STT_Schema.parse({}).native, language: "ja-JP" } });
    await service.init();
    await flush();
    vi.mocked(ask).mockResolvedValueOnce(true);
    await service.runStartupOffer();
    await flush();
    expect(data.backend).toBe(STT_Backends.local);
    expect(data.local.language).toBe("ja");
    expect(data.desktopDefaultMigrated).toBe(true);
    expect(toast.info).not.toHaveBeenCalled();
    // Runs once per launch.
    await service.runStartupOffer();
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it("goes straight to Whisper when the offer was already shown", async () => {
    isApp = true;
    const { service, data } = makeService({ localOfferShown: true });
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.whisper);
    await service.runStartupOffer();
    expect(ask).not.toHaveBeenCalled();
  });

  it("starts a fresh install on local, silently (onboarding shows the choice)", async () => {
    isApp = true;
    const { service, data } = makeService({}, { onboardingComplete: false });
    await service.init();
    await flush();
    expect(data.backend).toBe(STT_Backends.local);
    expect(data.local.language).toBe("auto");
    expect(data.desktopDefaultMigrated).toBe(true);
    await service.runStartupOffer();
    expect(ask).not.toHaveBeenCalled();
    expect(toast.info).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(0);
  });
});

describe("Service_STT language suggestion (the automatic backend on automatic language)", () => {
  const FRENCH = "Je voudrais essayer ce nouveau micro ce soir";
  const plan = (missing: string[]) => ({
    language: "fr",
    tier: "auto",
    streaming: null,
    final: { engine: "sherpa-offline", model: "stt/parakeet-ultra" },
    fallback: null,
    languageId: null,
    models: ["stt/parakeet-ultra"],
    missing,
    vramMb: 0,
    notes: [],
  });
  let n = 0;
  const say = (engine: FakeEngine, lang = "fr", text = FRENCH) =>
    engine.receiver.onFinal(text, { segmentId: `local-x:1:${++n}`, revision: 2, lang });

  async function running(overrides: Partial<STT_State> = {}) {
    isApp = true;
    const ctx = makeService({ backend: STT_Backends.local, desktopDefaultMigrated: true, ...overrides });
    await ctx.service.init();
    ctx.service.start();
    await flush();
    last().receiver.onStart();
    return { ...ctx, engine: last() };
  }

  it("offers the detected language after three consistent finals and publishes them as usual", async () => {
    const { service, engine } = await running();
    say(engine);
    say(engine);
    expect(service.languageOffer.view.language).toBeNull();
    say(engine);
    await flush();
    expect(service.languageOffer.view).toMatchObject({ language: "fr", source: "detected" });
    expect(publishText).toHaveBeenCalledTimes(3);
    expect(publishText).toHaveBeenLastCalledWith(TextEventSource.stt, expect.objectContaining({ value: FRENCH, type: TextEventType.final, lang: "fr" }));
  });

  it("never lets the suggestion get in the way of a caption", async () => {
    const { service, engine } = await running();
    const failing = vi.spyOn(service.languageOffer, "noteFinal").mockImplementation(() => {
      throw new Error("boom");
    });
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    say(engine);
    await flush();
    expect(failing).toHaveBeenCalledTimes(1);
    expect(logged).toHaveBeenCalled();
    expect(publishText).toHaveBeenCalledWith(TextEventSource.stt, expect.objectContaining({ value: FRENCH, type: TextEventType.final }));
  });

  it("starts counting again with each Start", async () => {
    const { service, engine } = await running();
    say(engine);
    say(engine);
    service.start();
    await flush();
    const next = last();
    expect(next).not.toBe(engine);
    next.receiver.onStart();
    say(next);
    expect(service.languageOffer.view.language).toBeNull();
  });

  it("counts the finals of the current session only", async () => {
    const { service, engine } = await running();
    service.start();
    await flush();
    say(engine);
    say(engine);
    say(engine);
    expect(service.languageOffer.view.language).toBeNull();
  });

  it("does not count another backend, or a language that is already set", async () => {
    const other = await running({ backend: STT_Backends.whisper });
    for (let i = 0; i < 4; i++) say(other.engine);
    expect(other.service.languageOffer.view.language).toBeNull();

    const set = await running({ local: { ...Service_STT_Schema.parse({}).local, language: "fr" } });
    for (let i = 0; i < 4; i++) say(set.engine);
    expect(set.service.languageOffer.view.language).toBeNull();
  });

  it("withdraws the offer when the language is set in the settings", async () => {
    const { service, engine, data } = await running();
    for (let i = 0; i < 3; i++) say(engine);
    expect(service.languageOffer.view.language).toBe("fr");
    data.local.language = "de";
    await flush();
    expect(service.languageOffer.view.language).toBeNull();
  });

  it("Lock stores the language and restarts the running session when no model has to be downloaded", async () => {
    const { service, engine, data } = await running();
    for (let i = 0; i < 3; i++) say(engine);
    vi.mocked(invoke).mockResolvedValueOnce(plan([]));
    expect(await service.languageOffer.lock("fr")).toBe("restarted");
    await flush();
    expect(vi.mocked(invoke)).toHaveBeenCalledWith("plugin:speech|preview_route", { request: expect.objectContaining({ language: "fr" }) });
    expect(data.local.language).toBe("fr");
    expect(data.local.languageDecision).toMatchObject({ outcome: "locked", language: "fr" });
    expect(engine.stopped).toBe(1);
    expect(FakeEngine.instances).toHaveLength(2);
    // An automatic start: it must never open a download question for a viewer who is live.
    expect((last() as unknown as { startAutomatic?: boolean }).startAutomatic).toBe(true);
    expect(service.languageOffer.view.result).toEqual({ language: "fr", outcome: "restarted" });
  });

  it("Lock leaves a running session alone when the new route needs a download", async () => {
    const { service, engine, data } = await running();
    for (let i = 0; i < 3; i++) say(engine);
    vi.mocked(invoke).mockResolvedValueOnce(plan(["stt/parakeet-ultra"]));
    expect(await service.languageOffer.lock("fr")).toBe("download");
    await flush();
    expect(data.local.language).toBe("fr");
    expect(FakeEngine.instances).toHaveLength(1);
    expect(engine.stopped).toBe(0);
  });

  it("Keep automatic and Don't ask again are stored in the profile", async () => {
    const kept = await running();
    for (let i = 0; i < 3; i++) say(kept.engine);
    kept.service.languageOffer.keepAutomatic();
    expect(kept.data.local.languageDecision).toMatchObject({ outcome: "kept", language: "fr" });
    expect(kept.data.local.language).toBe("auto");

    const never = await running();
    for (let i = 0; i < 3; i++) say(never.engine);
    never.service.languageOffer.neverAsk();
    expect(never.data.local.languageDecision).toMatchObject({ outcome: "never", language: "fr" });
    // Persisted answers survive a new launch: nothing is offered.
    const again = await running({ local: never.data.local });
    for (let i = 0; i < 6; i++) say(again.engine);
    expect(again.service.languageOffer.view.language).toBeNull();
  });
});
