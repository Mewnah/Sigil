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
import { pushSystemLog } from "@/core/services/systemLog";
import { useAppUIStore } from "@/core/ui/store";
import { Services } from "@/services-registry";
import { sttControlAction } from "../controls";
import Service_STT, { connectingFallbackMs, type SttBackendMap } from "../index";
import { MAX_RESTARTS, RESTART_DELAYS_MS, SLOW_RESTART_MS, STABLE_RUN_MS } from "../restartPolicy";
import { STREAM_OFFLINE_CONFIRM_MS } from "../streamStop";
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
/** Handlers of the local pubsub topics the service subscribed to (`stream.on_ended`, `stream.on_started`). */
let topics: Record<string, () => void>;

function makeService(overrides: Partial<STT_State> = {}, rootOverrides: Record<string, unknown> = {}) {
  const data = proxy({ ...Service_STT_Schema.parse({}), ...overrides });
  const state = { services: { stt: { data } }, onboardingComplete: true, ...rootOverrides };
  publishText = vi.fn();
  topics = {};
  vi.stubGlobal("window", {
    Config: { isApp: () => isApp },
    ApiServer: {
      state,
      patchService: (key: "stt", fn: (s: { data: STT_State }) => void) => fn(state.services[key]),
    },
    ApiShared: {
      pubsub: {
        publishText,
        subscribe: vi.fn((topic: string, fn: () => void) => {
          topics[topic] = fn;
          return `sub-${topic}`;
        }),
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

  it("toasts download progress while the Rewrite tab is showing (it has no progress of its own) and not after the Speech tab is chosen", async () => {
    // Open on Rewrite (the status-bar card), then the in-panel tabs move the store (setSidebarTab) as the user clicks.
    useAppUIStore.getState().changeTab({ tab: Services.transform });
    const { service } = makeService({ backend: STT_Backends.whisper });
    service.start();
    await flush();
    last().receiver.onStatus("Downloading: 40%", { progress: 40 });
    expect(toast.info).toHaveBeenCalledTimes(1);

    useAppUIStore.getState().setSidebarTab({ tab: Services.stt }); // the Speech tab shows the progress inline
    last().receiver.onStatus("Downloading: 50%", { progress: 50 });
    expect(toast.dismiss).toHaveBeenCalledWith("stt-model-progress");
    expect(toast.info).toHaveBeenCalledTimes(1);
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

describe("Service_STT start at launch while the setup wizard shows", () => {
  // What the wizard's speech step leaves: the built-in engine, the migration done, Start when Sigil opens on.
  const saved = { autoStart: true, backend: STT_Backends.local, desktopDefaultMigrated: true };

  it("starts nothing behind a wizard that shows again (the user quit in the middle of it)", async () => {
    isApp = true;
    const { service } = makeService(saved, { onboardingComplete: false });
    await service.init();
    await flush();
    expect(FakeEngine.instances).toHaveLength(0);
    expect(toast).not.toHaveBeenCalled(); // no "models not downloaded" error over a wizard that cannot open Speech to Text
  });

  it("starts at launch once the wizard is done", async () => {
    isApp = true;
    const { service } = makeService(saved, { onboardingComplete: true });
    await service.init();
    await flush();
    expect(FakeEngine.instances).toHaveLength(1);
    expect(last().started).toBe(1);
  });

  it("the web build has no wizard, so it starts as before", async () => {
    isApp = false;
    const { service } = makeService({ autoStart: true, backend: STT_Backends.local }, { onboardingComplete: false });
    await service.init();
    await flush();
    expect(FakeEngine.instances).toHaveLength(1);
  });
});

describe("Service_STT a session that did not start by design", () => {
  it("shows a notice that closes by itself, not an error that stays", async () => {
    const { service } = makeService({ backend: STT_Backends.local });
    service.start({ automatic: true });
    await flush();
    last().receiver.onStop("Speech models are not downloaded.", { notice: true });
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("Speech models are not downloaded."); // the status still says why
    expect(toast.info).toHaveBeenCalledWith("Speech models are not downloaded.", { autoClose: 12_000 });
    expect(toast).not.toHaveBeenCalled(); // no error toast
  });

  it("an engine error without the notice option is still an error that stays", async () => {
    const { service } = makeService({ backend: STT_Backends.local });
    service.start({ automatic: true });
    await flush();
    last().receiver.onStop("The microphone is in use.");
    expect(toast).toHaveBeenCalledWith("The microphone is in use.", { type: "error", autoClose: false });
    expect(toast.info).not.toHaveBeenCalled();
    expect(service.serviceState.error).toBe("The microphone is in use.");
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

describe("Service_STT reconnect after a failure that may pass", () => {
  const lost = "The connection to Deepgram was lost (1006).";

  /** A running session on the engine of `backend`. */
  async function running(backend = STT_Backends.deepgram, overrides: Partial<STT_State> = {}, options: { automatic?: boolean } = {}) {
    const made = makeService({ backend, ...overrides });
    made.service.start(options);
    await flush();
    last().receiver.onStart();
    expect(made.service.serviceState.status).toBe(ServiceNetworkState.connected);
    return made;
  }

  it("reconnects on the restart ladder with a status line and no error, and is connected again", async () => {
    const { service } = await running();
    const first = last();
    first.receiver.onStop(lost, { retry: true });
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(service.serviceState.error).toBe("");
    expect(toast).not.toHaveBeenCalled();
    expect(service.serviceState.statusText).toBe("stt.status_reconnecting");
    expect(service.serviceState.statusLevel).toBe("warning");
    await flush();
    expect(first.stopped).toBe(1); // the dead session is released (the microphone with it) while the wait runs

    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0] - 1);
    expect(FakeEngine.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeEngine.instances).toHaveLength(2);
    expect((last() as unknown as { startAutomatic?: boolean }).startAutomatic).toBe(true);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(service.serviceState.statusText).toBe("");
    expect(toast).not.toHaveBeenCalled();
    // The old session is stale for good: it cannot end the new one.
    first.receiver.onStop("late");
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("keeps trying at the slow pace once the ladder is used up, never with an error toast", async () => {
    const { service } = await running();
    last().receiver.onStop(lost, { retry: true });
    // Each attempt fails to connect, the way an outage does.
    for (const delay of RESTART_DELAYS_MS) {
      await vi.advanceTimersByTimeAsync(delay);
      last().receiver.onStop(lost, { retry: true });
    }
    expect(FakeEngine.instances).toHaveLength(1 + MAX_RESTARTS);
    expect(service.serviceState.statusText).toBe("stt.status_reconnecting_slow");
    for (let i = 0; i < 3; i++) {
      await vi.advanceTimersByTimeAsync(SLOW_RESTART_MS - 1);
      expect(FakeEngine.instances).toHaveLength(1 + MAX_RESTARTS + i);
      await vi.advanceTimersByTimeAsync(1);
      expect(FakeEngine.instances).toHaveLength(2 + MAX_RESTARTS + i);
      last().receiver.onStop(lost, { retry: true });
    }
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(toast).not.toHaveBeenCalled();
    // The outage ends: the next attempt connects, and the streak is over.
    await vi.advanceTimersByTimeAsync(SLOW_RESTART_MS);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(toast).not.toHaveBeenCalled();
  });

  it("a session that ran STABLE_RUN_MS begins a new streak; one that never connected keeps counting", async () => {
    const { service } = await running();
    last().receiver.onStop(lost, { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    last().receiver.onStop(lost, { retry: true }); // failed to connect: the second wait
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[1] - 1);
    expect(FakeEngine.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeEngine.instances).toHaveLength(3);
    last().receiver.onStart();
    await vi.advanceTimersByTimeAsync(STABLE_RUN_MS);
    last().receiver.onStop(lost, { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(FakeEngine.instances).toHaveLength(4);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
  });

  it("a failure that needs the user still stops with its message, also during a reconnect", async () => {
    const { service } = await running();
    last().receiver.onStop(lost, { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    last().receiver.onStop("Deepgram rejected the key.");
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe("Deepgram rejected the key.");
    expect(toast).toHaveBeenCalledWith("Deepgram rejected the key.", { type: "error", autoClose: false });
    await vi.advanceTimersByTimeAsync(10 * SLOW_RESTART_MS);
    expect(FakeEngine.instances).toHaveLength(2);
  });

  it("a Start by hand that fails keeps failing with its message; a start at launch is retried", async () => {
    const manual = makeService({ backend: STT_Backends.deepgram });
    manual.service.start();
    await flush();
    last().receiver.onStop(lost, { retry: true });
    expect(manual.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(toast).toHaveBeenCalledWith(lost, { type: "error", autoClose: false });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(FakeEngine.instances).toHaveLength(1);

    FakeEngine.instances = [];
    vi.mocked(toast).mockClear();
    const launch = makeService({ backend: STT_Backends.deepgram });
    launch.service.start({ automatic: true });
    await flush();
    last().receiver.onStop(lost, { retry: true });
    expect(launch.service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(toast).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(FakeEngine.instances).toHaveLength(2);
    last().receiver.onStart();
    expect(launch.service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("Stop ends the wait; a Start by hand reconnects at once and begins a new streak", async () => {
    const stopped = await running();
    last().receiver.onStop(lost, { retry: true });
    stopped.service.stop();
    expect(stopped.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(stopped.service.serviceState.statusText).toBe("");
    await vi.advanceTimersByTimeAsync(10 * SLOW_RESTART_MS);
    expect(FakeEngine.instances).toHaveLength(1);

    FakeEngine.instances = [];
    const manual = await running();
    last().receiver.onStop(lost, { retry: true });
    manual.service.start();
    await flush();
    expect(FakeEngine.instances).toHaveLength(2);
    last().receiver.onStart();
    expect(manual.service.serviceState.status).toBe(ServiceNetworkState.connected);
    await vi.advanceTimersByTimeAsync(10 * SLOW_RESTART_MS);
    expect(FakeEngine.instances).toHaveLength(2);
  });

  it("gives up the wait when another engine was chosen meanwhile", async () => {
    const { service, data } = await running();
    last().receiver.onStop(lost, { retry: true });
    data.backend = STT_Backends.whisper;
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("says it is retrying for as long as it retries, so the controls offer Stop instead of a dead 'connecting'", async () => {
    const { service } = await running();
    const state = service.serviceState;
    expect(state.retrying).toBe(false);
    expect(sttControlAction(state)).toBe("stop");
    last().receiver.onStop(lost, { retry: true });
    expect(state.status).toBe(ServiceNetworkState.connecting);
    expect(state.retrying).toBe(true);
    expect(sttControlAction(state)).toBe("stop");
    // The attempt itself is still a retry (the status reads 'connecting' while it opens the microphone).
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    last().receiver.onStatus("Opening the microphone...");
    expect(state.retrying).toBe(true);
    last().receiver.onStart();
    expect(state.retrying).toBe(false);
    // Stop ends a wait that has no end of its own.
    last().receiver.onStop(lost, { retry: true });
    expect(state.retrying).toBe(true);
    service.stop();
    expect(state.status).toBe(ServiceNetworkState.disconnected);
    expect(state.retrying).toBe(false);
    expect(sttControlAction(state)).toBe("start");
    // A failure that needs the user is not a retry.
    const rejected = await running();
    last().receiver.onStop(lost, { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    last().receiver.onStop("Deepgram rejected the key.");
    expect(rejected.service.serviceState.retrying).toBe(false);
    expect(rejected.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });

  it("an engine's own retry wait (a status flagged 'retrying') also reads as retrying until it reports an empty status", async () => {
    const { service } = await running(STT_Backends.whisper);
    const state = service.serviceState;
    last().receiver.onStatus("Trying again in 5 s", { level: "warning", retrying: true });
    expect(state.retrying).toBe(true);
    last().receiver.onStatus("Loading the model...");
    expect(state.retrying).toBe(true);
    last().receiver.onStatus("");
    expect(state.retrying).toBe(false);
    // Stop clears it even when the engine never got to.
    last().receiver.onStatus("Trying again in 5 s", { level: "warning", retrying: true });
    service.stop();
    expect(state.retrying).toBe(false);
  });

  it("a refused handshake of a start at launch is retried on the fast ladder only, then stops with its message", async () => {
    // A rejected key and a network error look the same to a browser: a wrong key must not loop for ever.
    const refused = "Could not connect to Deepgram. Check the API key and your internet connection.";
    const unproven = { retry: true, unproven: true };
    const { service } = makeService({ backend: STT_Backends.deepgram });
    service.start({ automatic: true });
    await flush();
    last().receiver.onStop(refused, unproven);
    for (const delay of RESTART_DELAYS_MS) {
      expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
      expect(toast).not.toHaveBeenCalled();
      await vi.advanceTimersByTimeAsync(delay);
      last().receiver.onStop(refused, unproven);
    }
    expect(FakeEngine.instances).toHaveLength(1 + MAX_RESTARTS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(service.serviceState.error).toBe(refused);
    expect(service.serviceState.retrying).toBe(false);
    expect(toast).toHaveBeenCalledWith(refused, { type: "error", autoClose: false });
    await vi.advanceTimersByTimeAsync(10 * SLOW_RESTART_MS);
    expect(FakeEngine.instances).toHaveLength(1 + MAX_RESTARTS);
  });

  it("the same refusal after a session connected is an outage: the retries go on at the slow pace", async () => {
    const refused = "Could not connect to Deepgram. Check the API key and your internet connection.";
    const { service } = await running();
    last().receiver.onStop(lost, { retry: true });
    for (let i = 0; i < MAX_RESTARTS + 3; i++) {
      await vi.advanceTimersByTimeAsync(i < MAX_RESTARTS ? RESTART_DELAYS_MS[i] : SLOW_RESTART_MS);
      last().receiver.onStop(refused, { retry: true, unproven: true });
    }
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(service.serviceState.retrying).toBe(true);
    expect(toast).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1 + MAX_RESTARTS + 3);
  });

  it("a rate limit waits the slow pace before the first retry, with a line of its own", async () => {
    const { service } = await running(STT_Backends.openai_audio);
    last().receiver.onStop("5 utterances failed in a row; last error: rate_limited: slow down", { retry: true, slow: true });
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(service.serviceState.statusText).toBe("stt.status_reconnecting_limited");
    await vi.advanceTimersByTimeAsync(SLOW_RESTART_MS - 1);
    expect(FakeEngine.instances).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(FakeEngine.instances).toHaveLength(2);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(toast).not.toHaveBeenCalled();
  });

  it("an attempt that hangs is a failed attempt: the connecting watchdog does not end the loop", async () => {
    const { service } = await running();
    last().receiver.onStop(lost, { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(FakeEngine.instances).toHaveLength(2); // the attempt never reports anything
    await vi.advanceTimersByTimeAsync(connectingFallbackMs(STT_Backends.deepgram));
    expect(toast.warning).not.toHaveBeenCalled();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(service.serviceState.retrying).toBe(true);
    expect(service.serviceState.statusText).toBe("stt.status_reconnecting");
    // The hung session is released, and the next wait of the ladder ends in a new attempt.
    expect(FakeEngine.instances[1].stopped).toBe(1);
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[1]);
    expect(FakeEngine.instances).toHaveLength(3);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("a Start by hand that hangs is still stopped by the connecting watchdog", async () => {
    const { service } = makeService({ backend: STT_Backends.deepgram });
    service.start();
    await flush();
    await vi.advanceTimersByTimeAsync(connectingFallbackMs(STT_Backends.deepgram));
    expect(toast.warning).toHaveBeenCalledTimes(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
  });
});

describe("Service_STT stop with Twitch stream", () => {
  async function running(overrides: Partial<STT_State> = {}) {
    const made = makeService({ backend: STT_Backends.whisper, stopWithStream: true, ...overrides });
    await made.service.init();
    made.service.start();
    await flush();
    last().receiver.onStart();
    expect(made.service.serviceState.status).toBe(ServiceNetworkState.connected);
    return made;
  }

  it("stops when the stream stays offline, and starts again (automatically) when it comes back", async () => {
    const { service } = await running();
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS - 1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    await vi.advanceTimersByTimeAsync(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(FakeEngine.instances).toHaveLength(1);

    topics["stream.on_started"]();
    await flush();
    expect(FakeEngine.instances).toHaveLength(2);
    expect((last() as unknown as { startAutomatic?: boolean }).startAutomatic).toBe(true);
    last().receiver.onStart();
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
  });

  it("a gap between broadcasts does not stop it, and a stop by hand is never undone", async () => {
    const { service } = await running();
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 2);
    topics["stream.on_started"]();
    await vi.advanceTimersByTimeAsync(2 * STREAM_OFFLINE_CONFIRM_MS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(FakeEngine.instances).toHaveLength(1);

    service.stop();
    topics["stream.on_started"]();
    await flush();
    expect(service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("a Start by hand while the stop waits for its confirmation is not undone when the timer runs out", async () => {
    const { service } = await running();
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS / 3);
    service.start();
    await flush();
    last().receiver.onStart();
    expect(FakeEngine.instances).toHaveLength(2);
    await vi.advanceTimersByTimeAsync(2 * STREAM_OFFLINE_CONFIRM_MS);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connected);
    expect(FakeEngine.instances[1].stopped).toBe(0);
  });

  it("resumes on the real live state of Twitch, the only thing that says the stream is back (nothing publishes stream.on_started)", async () => {
    const made = makeService({ backend: STT_Backends.whisper, stopWithStream: true });
    // The host's Twitch service: a valtio proxy whose liveStatus flips with every poll.
    const twitch = { state: proxy({ liveStatus: ServiceNetworkState.connected }) };
    (window as unknown as { ApiServer: Record<string, unknown> }).ApiServer.twitch = twitch;
    await made.service.init();
    made.service.start();
    await flush();
    last().receiver.onStart();
    expect(made.service.serviceState.status).toBe(ServiceNetworkState.connected);

    twitch.state.liveStatus = ServiceNetworkState.disconnected;
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS);
    expect(made.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    expect(FakeEngine.instances).toHaveLength(1);

    twitch.state.liveStatus = ServiceNetworkState.connected;
    await flush();
    expect(FakeEngine.instances).toHaveLength(2);
    expect((last() as unknown as { startAutomatic?: boolean }).startAutomatic).toBe(true);
    last().receiver.onStart();
    expect(made.service.serviceState.status).toBe(ServiceNetworkState.connected);
    made.service.dispose();
  });

  it("also stops a session that is reconnecting, and leaves it alone when the option is off", async () => {
    const reconnecting = await running({ backend: STT_Backends.deepgram });
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(STREAM_OFFLINE_CONFIRM_MS - 2_000);
    // The connection drops just before the stop is due: the reconnect (connecting) is stopped like a running session.
    last().receiver.onStop("The connection to Deepgram was lost (1006).", { retry: true });
    expect(reconnecting.service.serviceState.status).toBe(ServiceNetworkState.connecting);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(reconnecting.service.serviceState.status).toBe(ServiceNetworkState.disconnected);
    await vi.advanceTimersByTimeAsync(10 * SLOW_RESTART_MS);
    expect(FakeEngine.instances).toHaveLength(2);
    expect(toast).not.toHaveBeenCalled();

    FakeEngine.instances = [];
    const off = await running({ stopWithStream: false });
    topics["stream.on_ended"]();
    await vi.advanceTimersByTimeAsync(2 * STREAM_OFFLINE_CONFIRM_MS);
    expect(off.service.serviceState.status).toBe(ServiceNetworkState.connected);
  });
});

describe("Service_STT silent microphone", () => {
  /** A connected session whose engine can report what the microphone delivers. */
  async function connected(overrides: Partial<STT_State> = {}) {
    const made = makeService({ backend: STT_Backends.local, ...overrides });
    made.service.start();
    await flush();
    last().receiver.onStart();
    await flush();
    return made;
  }

  it("shows the device beside the status line, logs the outage once, and never publishes or toasts it", async () => {
    const { service } = await connected();
    const engine = last();
    engine.receiver.onInputSilence?.(true, "Headset Mic");
    expect(service.serviceState.silentInput).toEqual({ device: "Headset Mic" });
    expect(pushSystemLog).toHaveBeenCalledTimes(1);
    expect(pushSystemLog).toHaveBeenCalledWith("STT", expect.stringContaining("no_signal"), "warning");

    // Windows moves the default to another silent device: the panel follows, the log keeps its one line.
    engine.receiver.onInputSilence?.(true, "CABLE Output");
    expect(service.serviceState.silentInput).toEqual({ device: "CABLE Output" });
    engine.receiver.onInputSilence?.(true, "CABLE Output");
    expect(pushSystemLog).toHaveBeenCalledTimes(1);

    expect(publishText).not.toHaveBeenCalled();
    expect(toast).not.toHaveBeenCalled();
    expect(toast.warning).not.toHaveBeenCalled();
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("is a state of its own: status lines come and go without touching it", async () => {
    const { service } = await connected();
    const engine = last();
    engine.receiver.onInputSilence?.(true, "Headset Mic");
    engine.receiver.onStatus("Some speech was skipped", { level: "warning", expireMs: 15_000 });
    expect(service.serviceState.statusText).toBe("Some speech was skipped");
    expect(service.serviceState.silentInput).not.toBeNull();
    await vi.advanceTimersByTimeAsync(15_000);
    expect(service.serviceState.statusText).toBe("");
    engine.receiver.onStatus("");
    expect(service.serviceState.silentInput).toEqual({ device: "Headset Mic" });
  });

  it("clears as soon as signal returns", async () => {
    const { service } = await connected();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    last().receiver.onInputSilence?.(false);
    expect(service.serviceState.silentInput).toBeNull();
    // A second silence is shown again; the log keeps the one line of this run.
    last().receiver.onInputSilence?.(true, "");
    expect(service.serviceState.silentInput).toEqual({ device: "" });
    expect(pushSystemLog).toHaveBeenCalledTimes(1);
  });

  it("writes one log line per run however often a noise gate or a muting headset comes and goes", async () => {
    const { service } = await connected();
    const engine = last();
    // NVIDIA Broadcast / Krisp style: a pause, a phrase, a pause ...
    for (let i = 0; i < 6; i++) {
      engine.receiver.onInputSilence?.(true, "Broadcast Mic");
      expect(service.serviceState.silentInput).toEqual({ device: "Broadcast Mic" });
      engine.receiver.onInputSilence?.(false);
      expect(service.serviceState.silentInput).toBeNull();
    }
    expect(pushSystemLog).toHaveBeenCalledTimes(1);
    expect(pushSystemLog).toHaveBeenCalledWith("STT", expect.stringContaining("no_signal"), "warning");
  });

  it("logs the first outage of the next run again, but not after a reconnect of the same run", async () => {
    const silentLines = () => vi.mocked(pushSystemLog).mock.calls.filter(([, line]) => String(line).includes("no_signal")).length;
    const first = await connected({ backend: STT_Backends.deepgram });
    last().receiver.onInputSilence?.(true, "Headset Mic");
    expect(silentLines()).toBe(1);
    // The service reconnects after a lost connection: still the same run.
    last().receiver.onStop("The connection to Deepgram was lost (1006).", { retry: true });
    await vi.advanceTimersByTimeAsync(RESTART_DELAYS_MS[0]);
    expect(FakeEngine.instances).toHaveLength(2);
    last().receiver.onStart();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    expect(silentLines()).toBe(1);
    // Stop and Start by hand: a new run.
    first.service.stop();
    first.service.start();
    await flush();
    last().receiver.onStart();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    expect(silentLines()).toBe(2);
  });

  it("ends with the session: a Stop, an engine that stops itself, and a new Start", async () => {
    const stopped = await connected();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    stopped.service.stop();
    expect(stopped.service.serviceState.silentInput).toBeNull();

    const ended = await connected();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    last().receiver.onStop("The microphone is gone.");
    expect(ended.service.serviceState.silentInput).toBeNull();

    const restarted = await connected();
    last().receiver.onInputSilence?.(true, "Headset Mic");
    restarted.service.start();
    await flush();
    expect(restarted.service.serviceState.silentInput).toBeNull();
  });

  it("ends while the service reconnects (the engine session is over)", async () => {
    const { service } = await connected({ backend: STT_Backends.deepgram });
    last().receiver.onInputSilence?.(true, "Headset Mic");
    last().receiver.onStop("The connection was lost.", { retry: true });
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
    expect(service.serviceState.silentInput).toBeNull();
  });

  it("ignores a report of a session that is over", async () => {
    const { service } = await connected();
    const old = last();
    service.stop();
    old.receiver.onInputSilence?.(true, "Headset Mic");
    expect(service.serviceState.silentInput).toBeNull();
    expect(pushSystemLog).not.toHaveBeenCalled();
  });
});

describe("Service_STT a page that loads again", () => {
  const saved = { autoStart: true, backend: STT_Backends.local, desktopDefaultMigrated: true };
  const calls = () => vi.mocked(invoke).mock.calls.map(([command, args]) => [command, args] as const);
  const stops = () => calls().filter(([command]) => command === "plugin:speech|stop_pipeline");

  /** The Rust side as the previous page left it: `running` pipelines, everything else idle. */
  function rustHas(running: { pipelineId: string }[], fail: Record<string, boolean> = {}) {
    vi.mocked(invoke).mockImplementation((async (command: string) => {
      if (fail[command]) throw new Error(`${command} failed`);
      return command === "plugin:speech|get_pipeline_status" ? running : "";
    }) as never);
  }

  it("stops the mic pipeline the previous page left running, before anything starts, and then starts captions again", async () => {
    isApp = true;
    rustHas([{ pipelineId: "mic" }, { pipelineId: "loopback" }]);
    let stoppedBeforeStart: boolean | null = null;
    FakeEngine.startImpl = () => {
      stoppedBeforeStart = stops().length === 1;
    };
    const { service } = makeService(saved);
    await service.init();
    await flush();

    expect(stops()).toEqual([["plugin:speech|stop_pipeline", { pipelineId: "mic" }]]);
    expect(stoppedBeforeStart).toBe(true);
    expect(FakeEngine.instances).toHaveLength(1);
    expect(last().started).toBe(1);
    expect(pushSystemLog).toHaveBeenCalledWith("STT", "stt.log_leftover_stopped", "info");
  });

  it("also stops the Whisper recording and the raw capture of the other backends, which cost nothing when idle", async () => {
    isApp = true;
    rustHas([]);
    const { service } = makeService({ ...saved, autoStart: false });
    await service.init();
    await flush();
    const commands = calls().map(([command]) => command);
    expect(commands).toEqual([
      "plugin:speech|get_pipeline_status",
      "plugin:whisper|stop_recording",
      "plugin:audio|stop_audio_capture",
      "plugin:speech|stop_pipeline",
    ]);
    // Nothing was listed, so there is no log line: the stop itself is a no-op in Rust.
    expect(pushSystemLog).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(0);
  });

  it("stops the mic pipeline also when the status list does not show it yet (a start still loading models), without a log line", async () => {
    isApp = true;
    // The old page's start_pipeline is still building the session in Rust: get_pipeline_status lists finished starts only.
    rustHas([]);
    let stoppedBeforeStart: boolean | null = null;
    FakeEngine.startImpl = () => {
      stoppedBeforeStart = stops().length === 1;
    };
    const { service } = makeService(saved);
    await service.init();
    await flush();
    expect(stops()).toEqual([["plugin:speech|stop_pipeline", { pipelineId: "mic" }]]);
    expect(stoppedBeforeStart).toBe(true);
    expect(pushSystemLog).not.toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("stops the mic pipeline when the status question never answers, and still starts", async () => {
    isApp = true;
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(invoke).mockImplementation((async (command: string) => {
      if (command === "plugin:speech|get_pipeline_status") return new Promise(() => {});
      return "";
    }) as never);
    const { service } = makeService(saved);
    await service.init();
    await flush();
    expect(stops()).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(stops()).toEqual([["plugin:speech|stop_pipeline", { pipelineId: "mic" }]]);
    expect(error).toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1);
    expect(pushSystemLog).not.toHaveBeenCalled();
  });

  it("holds a Start by hand until the old session is gone", async () => {
    isApp = true;
    let release!: () => void;
    const held = new Promise<void>((resolve) => { release = resolve; });
    vi.mocked(invoke).mockImplementation((async (command: string) => {
      if (command === "plugin:speech|get_pipeline_status") {
        await held;
        return [{ pipelineId: "mic" }];
      }
      return "";
    }) as never);
    const { service } = makeService({ ...saved, autoStart: false });
    void service.init();
    service.start();
    await flush();
    expect(FakeEngine.instances).toHaveLength(0);
    release();
    await flush();
    expect(stops()).toHaveLength(1);
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("starts anyway when Rust does not answer", async () => {
    isApp = true;
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    rustHas([], { "plugin:speech|get_pipeline_status": true, "plugin:whisper|stop_recording": true, "plugin:audio|stop_audio_capture": true });
    const { service } = makeService(saved);
    await service.init();
    await flush();
    expect(error).toHaveBeenCalled();
    expect(FakeEngine.instances).toHaveLength(1);
  });

  it("starts anyway when a Rust call never answers, after a bounded wait", async () => {
    isApp = true;
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    // A device release stuck in a driver: nothing ever comes back from Rust.
    vi.mocked(invoke).mockImplementation((() => new Promise(() => {})) as never);
    const { service } = makeService(saved);
    await service.init();
    await vi.advanceTimersByTimeAsync(7_000);
    expect(FakeEngine.instances, "still within the bound").toHaveLength(0);
    await vi.advanceTimersByTimeAsync(2_000);
    expect(error).toHaveBeenCalledWith(expect.stringContaining("did not stop in time"), expect.anything());
    expect(FakeEngine.instances).toHaveLength(1);
    expect(last().started).toBe(1);
    expect(service.serviceState.status).toBe(ServiceNetworkState.connecting);
  });

  it("a Start by hand is not held back for good by a hung stop either", async () => {
    isApp = true;
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.mocked(invoke).mockImplementation((() => new Promise(() => {})) as never);
    const { service } = makeService({ ...saved, autoStart: false });
    await service.init();
    service.start();
    await vi.advanceTimersByTimeAsync(9_000);
    expect(FakeEngine.instances).toHaveLength(1);
    expect(last().started).toBe(1);
  });

  it("does nothing in the web build", async () => {
    isApp = false;
    const { service } = makeService({ ...saved, autoStart: false });
    await service.init();
    await flush();
    expect(invoke).not.toHaveBeenCalled();
  });
});
