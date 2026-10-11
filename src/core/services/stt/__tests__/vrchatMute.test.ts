/**
 * Privacy contract of "Mute captions when I mute in VRChat" (P6-C mute sync), end to end through the real services:
 * VRChat's `MuteSelf` arrives as the Rust `osc:mute_self` event, the VRChat service's mute controller mutes
 * speech-to-text, and every consumer of `text.stt` (the VRChat chatbox, text-to-speech, the OBS caption feed, the
 * text history) is a real service on the real pub/sub bus.
 *
 * The rule (the same one the Rust speech pipeline applies at the mute edge, `speech/mute.rs`):
 * - the sentence still being spoken when the mute arrives is dropped: it is closed with the `[...]` cancel marker and
 *   neither its later hypotheses nor its final are ever published;
 * - a sentence the engine had already finished (spoken unmuted) but still holds for its two-pass replacement is
 *   published once, with its first-pass text, by the mute itself;
 * - nothing is published for audio that arrives while muted, and a sentence that was open during the mute does not
 *   come out after the unmute.
 */
import i18n from "i18next";
import PubSub from "pubsub-js";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEventSource, TextEventType, type ObsCaptionEnvelope, type RustSpeechUtterance, type TextEvent } from "@/types";

const h = vi.hoisted(() => ({
  listeners: new Map<string, (e: { payload: unknown }) => void>(),
  sendOsc: vi.fn(),
  /** `muteSelf` that `plugin:osc|listen_start` reports (VRChat's mute when the receiver starts). */
  muteSelfAtStart: { value: null as boolean | null },
}));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => false,
  invoke: vi.fn(async (cmd: string) => {
    if (cmd === "plugin:osc|listen_start")
      return {
        running: true,
        mode: "oscQuery",
        address: "127.0.0.1:53124",
        port: 53124,
        oscQueryPort: 53125,
        advertised: true,
        vrchatFound: true,
        muteSelf: h.muteSelfAtStart.value,
        packets: 0,
        lastPacketAgoMs: null,
        error: null,
        warning: null,
      };
    if (cmd === "plugin:osc|bind_status") return { ok: true };
    return "";
  }),
}));
vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async (event: string, cb: (e: { payload: unknown }) => void) => {
    h.listeners.set(event, cb);
    return () => void h.listeners.delete(event);
  }),
}));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warn: vi.fn(), warning: vi.fn(), success: vi.fn(), update: vi.fn(), dismiss: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({
  migrateLegacyNativeInputDevice: vi.fn(async () => {}),
  resolveRustInputDeviceForInvoke: (d: string | undefined) => (d && d.trim() ? d.trim() : null),
}));
vi.mock("@/core/services/audioDevices", () => ({ audioDevicesApi: { setCaptureMute: vi.fn(async () => {}) } }));
vi.mock("../../vrc/targets/utils", () => ({ sendOsc: (...args: unknown[]) => h.sendOsc(...args) }));
// Real engines pull in SDKs and Tauri APIs; the service is tested against a fake engine.
vi.mock("../services/azure", () => ({ STT_AzureService: class {} }));
vi.mock("../services/deepgram", () => ({ STT_DeepgramService: class {} }));
vi.mock("../services/native", () => ({ STT_NativeService: class {} }));
vi.mock("../services/whisper", () => ({ STT_WhisperService: class {} }));
vi.mock("../services/vosk", () => ({ STT_VoskService: class {} }));
vi.mock("../services/moonshine", () => ({ STT_MoonshineService: class {} }));
vi.mock("../services/openaiAudio", () => ({ STT_OpenAI_AudioService: class {} }));
vi.mock("../services/externalMic", () => ({ STT_ExternalMicService: class {} }));

import Service_PubSub from "@/shared/services/pubsub";
import { ObsCaptionFeedService } from "../../obs/captionFeed";
import type { OBS_State } from "../../obs/schema";
import Service_TTS, { type TTSBackendMap } from "../../tts";
import { Service_TTS_Schema, TTS_Backends, type TTS_State } from "../../tts/schema";
import type { ITTSReceiver, ITTSService } from "../../tts/types";
import { Service_VRC_Schema } from "../../vrc/schema";
import Service_STT, { type SttBackendMap } from "../index";
import { Service_STT_Schema, STT_Backends } from "../schema";
import { SegmentPublisher } from "../services/local";
import { SttMuteState, type ISTTReceiver, type ISTTService } from "../types";

const { default: Service_VRC } = await import("../../vrc");

const MARKER = "[...]";

/** The `local` engine as far as the mute rule is concerned: a real SegmentPublisher (two-pass) feeding the receiver. */
class LocalLikeEngine implements ISTTService {
  static instances: LocalLikeEngine[] = [];
  startAbortSignal: AbortSignal | null = null;
  startAutomatic?: boolean;
  readonly pub: SegmentPublisher;
  constructor(public receiver: ISTTReceiver) {
    LocalLikeEngine.instances.push(this);
    this.pub = new SegmentPublisher({
      pipelineId: "mic",
      stream: "mic",
      epoch: 5,
      twoPass: true,
      sink: {
        interim: (text, meta) => this.receiver.onInterim(text, meta),
        final: (text, meta) => this.receiver.onFinal(text, meta),
        abandon: (meta) => this.receiver.onInterim("", meta),
      },
    });
  }
  commitPending() {
    this.pub.commitHeld();
  }
  start() {}
  stop() {}
  dispose() {}
  /** The first-pass final of a sentence (held for its two-pass replacement). */
  first(segmentId: number, revision: number, text: string) {
    this.pub.final({ segmentId, revision, text, lang: "en", pass: "first", latencyMs: 0, startMs: 0, endMs: 2_000, turnEnd: null });
  }
  second(segmentId: number, revision: number, text: string) {
    this.pub.final({ segmentId, revision, text, lang: "en", pass: "second", latencyMs: 0, startMs: 0, endMs: 2_000, turnEnd: null });
  }
  partial(segmentId: number, revision: number, text: string) {
    this.pub.partial({ segmentId, revision, text, lang: "en" });
  }
}

class FakeTts implements ITTSService {
  static instances: FakeTts[] = [];
  played: string[] = [];
  constructor(public receiver: ITTSReceiver) {
    FakeTts.instances.push(this);
  }
  start(_state: TTS_State) {}
  stop() {
    this.receiver.onStop();
  }
  play(value: string) {
    this.played.push(value);
  }
  dispose() {}
}

const sttMap = Object.fromEntries(Object.values(STT_Backends).map((b) => [b, LocalLikeEngine])) as unknown as SttBackendMap;
const ttsMap = Object.fromEntries(Object.values(TTS_Backends).map((b) => [b, FakeTts])) as unknown as TTSBackendMap;

const flush = () => vi.advanceTimersByTimeAsync(0);

interface World {
  stt: Service_STT;
  engine: LocalLikeEngine;
  /** Every `text.stt` event, in publication order. */
  bus: TextEvent[];
  /** Messages the VRChat chatbox would show. */
  chatbox(): string[];
  /** The last typing-bubble value sent to VRChat (null: never sent). */
  typing(): boolean | null;
  spoken(): string[];
  obs: ObsCaptionEnvelope[];
  history(): string[];
  /** VRChat reports its microphone mute (the Rust `osc:mute_self` event). */
  vrchatMute(muted: boolean): void;
  dispose(): void;
}

async function world(opts: { vrchatMutedAtStart?: boolean | null; backend?: STT_Backends; muteSync?: boolean } = {}): Promise<World> {
  h.listeners.clear();
  h.sendOsc.mockReset();
  LocalLikeEngine.instances = [];
  FakeTts.instances = [];
  h.muteSelfAtStart.value = opts.vrchatMutedAtStart ?? null;

  const sttData = proxy({ ...Service_STT_Schema.parse({}), backend: opts.backend ?? STT_Backends.local, desktopDefaultMigrated: true, localOfferShown: true });
  const vrcData = proxy(Service_VRC_Schema.parse({}));
  vrcData.enable = true;
  vrcData.muteSync = opts.muteSync ?? true;
  vrcData.source = TextEventSource.stt;
  vrcData.textbox.indicator = true;
  const ttsData = proxy({
    ...Service_TTS_Schema.parse({}),
    source: TextEventSource.stt,
    inputField: false,
    desktopDefaultMigrated: true,
    localOfferShown: true,
  } as TTS_State);
  const obsData = proxy({
    browserCaptionsEnable: true,
    browserSource: TextEventSource.stt,
    browserInputField: false,
    browserInterim: true,
  } as unknown as OBS_State);

  const state = {
    id: "test",
    onboardingComplete: true,
    audioOutputDevice: "",
    services: { stt: { data: sttData }, vrc: { data: vrcData }, tts: { data: ttsData }, translation: { data: {} } },
  };
  const pubsub = new Service_PubSub();
  const stt = new Service_STT(sttMap);
  const apiServer: Record<string, unknown> = {
    state,
    stt,
    patchService: (key: "stt", fn: (s: { data: unknown }) => void) => fn((state.services as Record<string, { data: unknown }>)[key]),
    voiceChanger: { state: { isRunning: false } },
    sound: { enqueueVoiceClip: vi.fn(async () => {}), clearVoiceClips: vi.fn() },
  };
  vi.stubGlobal("window", {
    Config: { isApp: () => true, isServer: () => true, isClient: () => false },
    ApiServer: apiServer,
    ApiShared: { pubsub, peer: { broadcastPubSubParts: vi.fn() } },
  });
  stt.updateReplacementsCache();
  await pubsub.init();

  const bus: TextEvent[] = [];
  pubsub.subscribeText(TextEventSource.stt, (e) => void (e && bus.push(e)), true);
  const obs: ObsCaptionEnvelope[] = [];
  const feed = new ObsCaptionFeedService(() => obsData, (e) => void obs.push(e));
  feed.init();

  const vrc = new Service_VRC();
  await vrc.init();
  const tts = new Service_TTS(ttsMap);
  tts.updateReplacementsCache();
  await tts.init();
  tts.start();
  FakeTts.instances[FakeTts.instances.length - 1].receiver.onStart();

  stt.start();
  await flush();
  const engine = LocalLikeEngine.instances[LocalLikeEngine.instances.length - 1];
  engine.receiver.onStart();
  await flush();
  expect(stt.serviceState.status).toBe(ServiceNetworkState.connected);

  const oscArgs = (path: string) => h.sendOsc.mock.calls.filter(([p]) => p === path).map(([, args]) => args as Array<Record<string, unknown>>);
  return {
    stt,
    engine,
    bus,
    chatbox: () => oscArgs("/chatbox/input").map((args) => String(args[0].String)),
    typing: () => {
      const calls = oscArgs("/chatbox/typing");
      return calls.length === 0 ? null : (calls[calls.length - 1][0].Bool as boolean);
    },
    spoken: () => FakeTts.instances[FakeTts.instances.length - 1].played,
    obs,
    history: () => pubsub.textHistory.list.map((l) => l.value),
    vrchatMute: (muted) => {
      const listener = h.listeners.get("osc:mute_self");
      expect(listener, "the VRChat service listens for osc:mute_self").toBeDefined();
      listener!({ payload: { muted, origin: "osc" } });
    },
    dispose: () => {
      vrc.dispose();
      tts.dispose();
      feed.dispose();
      stt.dispose();
      PubSub.clearAllSubscriptions();
    },
  };
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

let w: World;

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal("navigator", { hardwareConcurrency: 8 });
});

afterEach(() => {
  w?.dispose();
  w = undefined as never;
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("VRChat mute sync: nothing of a muted sentence is published", () => {
  it("control: the same sentence without a mute reaches every consumer (the checks below can fail)", async () => {
    w = await world();
    w.engine.partial(1, 0, "I was just saying");
    w.engine.partial(1, 1, "I was just saying something private");
    w.engine.first(1, 2, "I was just saying something private.");
    w.engine.second(1, 3, "I was just saying something private!");
    await vi.advanceTimersByTimeAsync(30_000);
    expect(w.bus.filter((e) => e.type === TextEventType.final).map((e) => e.value)).toEqual(["I was just saying something private!"]);
    expect(w.chatbox()).toEqual(["I was just saying something private!"]);
    expect(w.spoken()).toEqual(["I was just saying something private!"]);
    expect(w.history()).toEqual(["I was just saying something private!"]);
    expect(w.obs.some((e) => e.type === "final" && e.value.includes("private"))).toBe(true);
  });

  it("is off by default, and with it off VRChat's mute changes nothing", async () => {
    // Off on purpose: a VRChat user who stays muted in the game and talks only to the stream, or who uses speech as the
    // chatbox's voice, would otherwise lose every caption on a new profile without a word of explanation.
    expect(Service_VRC_Schema.parse({}).muteSync).toBe(false);
    w = await world({ muteSync: false });
    w.vrchatMute(true);
    await flush();
    expect(w.stt.serviceState.muted).toBe(SttMuteState.unmuted);
    w.engine.receiver.onFinal("I am muted in VRChat but talking to chat.");
    await vi.advanceTimersByTimeAsync(5_000);
    expect(w.chatbox()).toEqual(["I am muted in VRChat but talking to chat."]);
  });

  it("drops the sentence in progress and its pending partial; only the cancel marker follows the mute", async () => {
    w = await world();
    w.engine.partial(1, 0, "I was just saying");
    await flush();
    expect(w.bus.map((e) => [e.type, e.value])).toEqual([[TextEventType.interim, "I was just saying"]]);
    expect(w.typing()).toBe(true);

    const mark = w.bus.length;
    w.vrchatMute(true);
    await flush();
    expect(w.stt.serviceState.muted).toBe(SttMuteState.muted);

    // The Rust pipeline drops the open utterance; the engine's last hypotheses and a final may still be in flight.
    w.engine.pub.utterance({ segmentId: 1, kind: "dropped" } as RustSpeechUtterance);
    w.engine.partial(1, 1, "I was just saying something private");
    w.engine.first(1, 2, "I was just saying something private.");
    w.engine.second(1, 3, "I was just saying something private!");
    await vi.advanceTimersByTimeAsync(30_000);

    // Every consumer of text.stt heard exactly one thing after the mute: the marker that closes the half sentence.
    expect(w.bus.slice(mark).map((e) => [e.type, e.value])).toEqual([[TextEventType.final, MARKER]]);
    expect(w.chatbox()).toEqual([]);
    expect(w.typing()).toBe(false); // the typing bubble the partial turned on is off again
    expect(w.spoken()).toEqual([]);
    expect(w.obs.map((e) => [e.type, e.value])).toEqual([["interim", "I was just saying"], ["final", MARKER]]);
    expect(w.history()).toEqual([MARKER]);
    for (const line of [...w.chatbox(), ...w.spoken(), ...w.history(), ...w.obs.map((e) => e.value)])
      expect(line).not.toContain("private");
  });

  it("publishes a sentence that was finished before the mute exactly once, and nothing after it", async () => {
    w = await world();
    // Finished and spoken unmuted; the first-pass text waits for its two-pass replacement, shown as an interim.
    w.engine.partial(1, 0, "Hello every");
    w.engine.first(1, 1, "Hello everyone, welcome to the stream.");
    await flush();
    expect(w.bus.every((e) => e.type === TextEventType.interim)).toBe(true);
    expect(w.chatbox()).toEqual([]);

    // The next sentence has started when the mute lands.
    w.engine.partial(2, 0, "and now a priv");
    await flush();
    w.vrchatMute(true);
    await flush();
    w.engine.pub.utterance({ segmentId: 2, kind: "dropped" } as RustSpeechUtterance);
    w.engine.second(1, 2, "Hello everyone, welcome to the stream!"); // the replacement of the finished sentence
    w.engine.partial(2, 1, "and now a private matter");
    w.engine.first(2, 2, "And now a private matter.");
    await vi.advanceTimersByTimeAsync(30_000);

    const finals = w.bus.filter((e) => e.type === TextEventType.final).map((e) => e.value);
    expect(finals).toEqual(["Hello everyone, welcome to the stream."]);
    expect(w.chatbox()).toEqual(["Hello everyone, welcome to the stream."]);
    expect(w.spoken()).toEqual(["Hello everyone, welcome to the stream."]);
    expect(w.history()).toEqual(["Hello everyone, welcome to the stream."]);
    expect(w.bus.some((e) => e.value.includes("private"))).toBe(false);
  });

  it("publishes nothing for speech that arrives while VRChat is muted, and the sentence open during the mute stays dropped after the unmute", async () => {
    w = await world();
    w.engine.receiver.onInterim("one moment");
    await flush();
    w.vrchatMute(true);
    await flush();
    const mark = w.bus.length;

    // An engine without segment ids that keeps delivering while muted (nothing gates its audio).
    w.engine.receiver.onInterim("private talk");
    w.engine.receiver.onInterim("private talk with the roommate");
    await flush();
    w.vrchatMute(false);
    await flush();
    expect(w.stt.serviceState.muted).toBe(SttMuteState.pendingUnmute); // waits for that sentence to end
    w.engine.receiver.onFinal("private talk with the roommate.");
    await flush();
    expect(w.stt.serviceState.muted).toBe(SttMuteState.unmuted);
    expect(w.bus.slice(mark)).toEqual([]);

    // The next sentence after the unmute is published normally, everywhere.
    w.engine.receiver.onFinal("Okay, back.");
    await vi.advanceTimersByTimeAsync(5_000);
    expect(w.bus.slice(mark).map((e) => e.value)).toEqual(["Okay, back."]);
    expect(w.chatbox()).toEqual(["Okay, back."]);
    expect(w.spoken()).toEqual(["Okay, back."]);
    expect(w.history().filter((v) => v.includes("private"))).toEqual([]);
    expect(w.obs.some((e) => e.value.includes("private"))).toBe(false);
  });

  it("a VRChat that is already muted when the receiver starts keeps speech-to-text silent from the first word", async () => {
    w = await world({ vrchatMutedAtStart: true });
    expect(w.stt.serviceState.muted).toBe(SttMuteState.muted);
    w.engine.partial(1, 0, "hello");
    w.engine.first(1, 1, "Hello there.");
    w.engine.receiver.onInterim("direct hypothesis");
    w.engine.receiver.onFinal("Direct sentence.");
    await vi.advanceTimersByTimeAsync(30_000);
    expect(w.bus).toEqual([]);
    expect(w.chatbox()).toEqual([]);
    expect(w.spoken()).toEqual([]);
    expect(w.obs).toEqual([]);
    expect(w.history()).toEqual([]);
  });

  it("the user's own mute is kept when VRChat unmutes, and VRChat's mute never outlives it", async () => {
    w = await world();
    w.stt.setMuted(true); // the user's hotkey
    w.vrchatMute(true);
    w.vrchatMute(false);
    expect(w.stt.serviceState.muted).toBe(SttMuteState.muted);
    w.engine.receiver.onFinal("still muted by me");
    await flush();
    expect(w.bus).toEqual([]);
  });
});
