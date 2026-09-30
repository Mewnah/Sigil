import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

const h = vi.hoisted(() => {
  /** Recognizers created by the fake Azure SDK. */
  const recognizers: FakeRecognizer[] = [];
  class FakeRecognizer {
    sessionStarted?: () => void;
    sessionStopped?: () => void;
    canceled?: (r: unknown, e: { errorCode: number; errorDetails: string }) => void;
    recognizing?: (r: unknown, e: { result: { text: string; reason: number; language?: string } }) => void;
    recognized?: (r: unknown, e: { result: { text: string; reason: number; language?: string } }) => void;
    started = false;
    stopped = false;
    closed = false;
    startError?: (e: string) => void;
    constructor(
      public speechConfig: { key: string; region: string; props: Record<string, string>; dictation: boolean; speechRecognitionLanguage: string },
      public langConfig: { languages: string[]; mode?: string },
      public audioConfig: { device: string },
    ) {
      recognizers.push(this);
    }
    startContinuousRecognitionAsync(_ok?: () => void, err?: (e: string) => void) {
      this.started = true;
      this.startError = err;
    }
    stopContinuousRecognitionAsync(ok?: () => void) {
      this.stopped = true;
      ok?.();
    }
    close() {
      this.closed = true;
    }
  }
  return { recognizers, FakeRecognizer, sdkLoads: 0 };
});

vi.mock("microsoft-cognitiveservices-speech-sdk", () => {
  h.sdkLoads++;
  const codes = ["NoError", "AuthenticationFailure", "BadRequestParameters", "TooManyRequests", "Forbidden", "ConnectionFailure", "ServiceTimeout", "ServiceError", "RuntimeError"];
  const CancellationErrorCode: Record<string | number, string | number> = {};
  codes.forEach((name, i) => {
    CancellationErrorCode[name] = i;
    CancellationErrorCode[i] = name;
  });
  return {
    CancellationErrorCode,
    ResultReason: { NoMatch: 0, RecognizingSpeech: 2, RecognizedSpeech: 3 },
    LanguageIdMode: { Continuous: "continuous" },
    PropertyId: {
      SpeechServiceResponse_ProfanityOption: "profanity",
      SpeechServiceConnection_InitialSilenceTimeoutMs: "initialSilence",
      SpeechServiceConnection_EndSilenceTimeoutMs: "endSilence",
      Speech_SegmentationSilenceTimeoutMs: "segmentationSilence",
    },
    SpeechConfig: {
      fromSubscription: (key: string, region: string) => ({
        key,
        region,
        props: {} as Record<string, string>,
        dictation: false,
        speechRecognitionLanguage: "",
        setProperty(id: string, value: string) {
          this.props[id] = value;
        },
        enableDictation() {
          this.dictation = true;
        },
      }),
    },
    AutoDetectSourceLanguageConfig: { fromLanguages: (languages: string[]) => ({ languages }) },
    AutoDetectSourceLanguageResult: { fromResult: (r: { language?: string }) => ({ language: r.language }) },
    AudioConfig: {
      fromDefaultMicrophoneInput: () => ({ device: "default" }),
      fromMicrophoneInput: (device: string) => ({ device }),
    },
    SpeechRecognizer: {
      FromConfig: (s: never, l: never, a: never) => new h.FakeRecognizer(s, l, a),
    },
  };
});
let desktop = true;
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => desktop }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ resolveRustInputDeviceForInvoke: () => null }));
vi.mock("@/shared/services/pubsub", () => ({ PUBSUB_STT_EXTERNAL_COMMAND: "stt.external.command" }));

import { STT_Backends, Service_STT_Schema, type STT_State } from "../schema";
import { azureCancelMessage, STT_AzureService } from "../services/azure";
import {
  DEEPGRAM_KEEPALIVE_MS,
  DEEPGRAM_MUTE_POLL_MS,
  DEEPGRAM_TIMESLICE_MS,
  deepgramCloseMessage,
  deepgramMicrophoneError,
  STT_DeepgramService,
} from "../services/deepgram";
import { EXTERNAL_MIC_WAITING_LINE_MS, STT_ExternalMicService } from "../services/externalMic";
import { nativeErrorMessage, STT_NativeService } from "../services/native";
import type { ISTTReceiver } from "../types";
import { initSttTestI18n, translated } from "./sttTestI18n";

function receiver(muted: () => boolean = () => false) {
  return {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onInterim: vi.fn(),
    onFinal: vi.fn(),
    onStatus: vi.fn(),
    isMuted: vi.fn(muted),
  } satisfies ISTTReceiver;
}

const settings = (patch: (s: STT_State) => void = () => {}): STT_State => {
  const s = Service_STT_Schema.parse({});
  patch(s);
  return s;
};

beforeAll(async () => {
  await initSttTestI18n();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------------------------------------------

describe("Azure", () => {
  beforeEach(() => {
    h.recognizers.length = 0;
  });

  const azure = (s: STT_State) => {
    s.azure.key = "azure-key";
    s.azure.location = "westeurope";
    s.azure.language = "en-US";
  };

  it("loads the Speech SDK only when a session starts (never with the app's startup graph)", async () => {
    expect(h.sdkLoads).toBe(0);
    const r = receiver();
    await new STT_AzureService(r).start(settings(azure));
    expect(h.sdkLoads).toBe(1);
  });

  it("refuses to start without key, region and language (no SDK work)", async () => {
    const r = receiver();
    await new STT_AzureService(r).start(settings());
    expect(r.onStop).toHaveBeenCalledWith(expect.stringContaining("Azure"));
    expect(h.recognizers).toHaveLength(0);
  });

  it("configures recognition and maps session events, with the detected language", async () => {
    const r = receiver();
    await new STT_AzureService(r).start(
      settings((s) => {
        azure(s);
        s.azure.secondary_language = "ja-JP";
        s.azure.use_secondary_language = true;
        s.azure.segmentationSilenceMs = "400";
      }),
    );
    expect(r.onStatus.mock.calls[0][0]).toBe("Connecting to Azure…");
    const rec = h.recognizers[0];
    expect(rec.speechConfig).toMatchObject({ key: "azure-key", region: "westeurope", speechRecognitionLanguage: "en-US", dictation: true });
    expect(rec.speechConfig.props).toMatchObject({ profanity: "masked", endSilence: "20000", segmentationSilence: "400" });
    expect(rec.langConfig).toEqual({ languages: ["en-US", "ja-JP"], mode: "continuous" });
    expect(rec.audioConfig).toEqual({ device: "default" });
    expect(rec.started).toBe(true);

    rec.sessionStarted!();
    expect(r.onStart).toHaveBeenCalledTimes(1);
    rec.recognizing!(null, { result: { text: "konni", reason: 2, language: "ja-JP" } });
    expect(r.onInterim).toHaveBeenLastCalledWith("konni", { lang: "ja" });
    // NoMatch after a partial: the partial was not speech.
    rec.recognized!(null, { result: { text: "", reason: 0 } });
    expect(r.onInterim).toHaveBeenLastCalledWith("");
    rec.recognized!(null, { result: { text: "", reason: 0 } });
    expect(r.onInterim).toHaveBeenCalledTimes(2);
    rec.recognized!(null, { result: { text: "Hello there.", reason: 3, language: "en-US" } });
    expect(r.onFinal).toHaveBeenCalledWith("Hello there.", { lang: "en" });
    expect(r.onFinal).toHaveBeenCalledTimes(1);
  });

  it("maps cancellations to readable stop messages", async () => {
    const r = receiver();
    await new STT_AzureService(r).start(settings(azure));
    h.recognizers[0].canceled!(null, { errorCode: 1, errorDetails: "WebSocket upgrade failed: Authentication error (401)" });
    expect(r.onStop).toHaveBeenCalledWith("Azure rejected the key or the region. Check both in the Azure settings.");
    const r2 = receiver();
    await new STT_AzureService(r2).start(settings(azure));
    h.recognizers[1].canceled!(null, { errorCode: 0, errorDetails: "" });
    expect(r2.onStop).toHaveBeenCalledWith(undefined);
  });

  it("azureCancelMessage", () => {
    expect(azureCancelMessage("NoError", "")).toBeNull();
    expect(azureCancelMessage(undefined, "")).toBeNull();
    for (const code of ["AuthenticationFailure", "Forbidden", "ConnectionFailure", "ServiceTimeout", "TooManyRequests"])
      expect(translated(azureCancelMessage(code, "details")), code).toBe(true);
    expect(azureCancelMessage("BadRequestParameters", "unsupported language")).toContain("unsupported language");
    expect(azureCancelMessage("ServiceError", "")).toContain("ServiceError");
  });

  it("a stop while the SDK loads creates nothing; a stop after start closes the recognizer", async () => {
    const r = receiver();
    const svc = new STT_AzureService(r);
    const starting = svc.start(settings(azure));
    svc.stop();
    await starting;
    expect(h.recognizers).toHaveLength(0);

    const svc2 = new STT_AzureService(r);
    await svc2.start(settings(azure));
    const rec = h.recognizers[0];
    svc2.stop();
    expect(rec.stopped && rec.closed).toBe(true);
    // Events after the stop are ignored.
    rec.recognized!(null, { result: { text: "late", reason: 3 } });
    expect(r.onFinal).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------------------------------------------

class FakeTrack {
  stopped = false;
  stop() {
    this.stopped = true;
  }
}

class FakeRecorder {
  static instances: FakeRecorder[] = [];
  state: "inactive" | "recording" | "paused" = "inactive";
  timeslice = 0;
  listeners: ((e: { data: unknown }) => void)[] = [];
  constructor(public stream: unknown, public options: unknown) {
    FakeRecorder.instances.push(this);
  }
  addEventListener(_: string, cb: (e: { data: unknown }) => void) {
    this.listeners.push(cb);
  }
  start(ms: number) {
    this.timeslice = ms;
    this.state = "recording";
  }
  pause() {
    this.state = "paused";
  }
  resume() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
  }
}

class FakeSocket {
  static instances: FakeSocket[] = [];
  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSING = 2;
  readonly CLOSED = 3;
  readyState = 0;
  sent: unknown[] = [];
  onopen?: () => void;
  onmessage?: (e: { data: string }) => void;
  onclose?: (e: { code: number; reason: string; wasClean: boolean }) => void;
  onerror?: () => void;
  constructor(public url: string, public protocols: string[]) {
    FakeSocket.instances.push(this);
  }
  send(data: unknown) {
    this.sent.push(data);
  }
  close() {
    this.remoteClose(1000, "", true);
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  message(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }
  remoteClose(code: number, reason: string, wasClean: boolean) {
    this.readyState = 3;
    this.onclose?.({ code, reason, wasClean });
  }
}

describe("Deepgram", () => {
  let track: FakeTrack;
  let getUserMedia: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    FakeRecorder.instances = [];
    FakeSocket.instances = [];
    track = new FakeTrack();
    getUserMedia = vi.fn(async () => ({ getTracks: () => [track] }));
    vi.stubGlobal("MediaRecorder", FakeRecorder);
    vi.stubGlobal("WebSocket", FakeSocket);
    vi.stubGlobal("navigator", { mediaDevices: { enumerateDevices: async () => [], getUserMedia } });
    vi.stubGlobal("window", { ApiServer: { state: { audioInputDevice: "" } } });
  });

  const deepgram = (s: STT_State) => {
    s.deepgram.key = "dg-key";
    s.deepgram.language = "en-GB";
  };

  async function connected(r: ISTTReceiver) {
    const svc = new STT_DeepgramService(r);
    svc.start(settings(deepgram));
    await vi.waitFor(() => expect(FakeSocket.instances).toHaveLength(1));
    const socket = FakeSocket.instances[0];
    socket.open();
    return { svc, socket, recorder: FakeRecorder.instances[0] };
  }

  it("reports the start on the status line and streams once the socket opens", async () => {
    const r = receiver();
    const { svc, socket, recorder } = await connected(r);
    expect(r.onStatus.mock.calls.map((c) => c[0])).toEqual(["Opening the microphone…", "Connecting to Deepgram…"]);
    expect(socket.url).toContain("model=nova-3");
    expect(socket.protocols).toEqual(["token", "dg-key"]);
    expect(r.onStart).toHaveBeenCalledTimes(1);
    expect(recorder.state).toBe("recording");
    expect(recorder.timeslice).toBe(250);
    svc.stop();
    expect(track.stopped).toBe(true);
  });

  it("maps results: interims, finals, and a final without words abandons the open interim", async () => {
    const r = receiver();
    const { svc, socket } = await connected(r);
    const result = (transcript: string, extra: Record<string, unknown> = {}) =>
      socket.message({ type: "Results", channel: { alternatives: [{ transcript }] }, ...extra });
    result("hel");
    expect(r.onInterim).toHaveBeenLastCalledWith("hel", { lang: "en" });
    result("", { is_final: true });
    expect(r.onInterim).toHaveBeenLastCalledWith("");
    result("", { is_final: true, speech_final: true }); // nothing open: nothing to abandon
    expect(r.onInterim).toHaveBeenCalledTimes(2);
    socket.message({ type: "Metadata", request_id: "x" });
    result("hello there", { is_final: true });
    expect(r.onFinal).toHaveBeenCalledWith("hello there", { lang: "en" });
    svc.stop();
  });

  it("a refused connection and a dropped connection end with readable messages", async () => {
    const r = receiver();
    const svc = new STT_DeepgramService(r);
    svc.start(settings(deepgram));
    await vi.waitFor(() => expect(FakeSocket.instances).toHaveLength(1));
    FakeSocket.instances[0].remoteClose(1006, "", false);
    expect(r.onStop).toHaveBeenCalledWith("Could not connect to Deepgram. Check the API key and your internet connection.");
    expect(track.stopped).toBe(true);

    FakeSocket.instances = [];
    FakeRecorder.instances = [];
    const r2 = receiver();
    const { socket } = await connected(r2);
    socket.remoteClose(1011, "NET-0001", true);
    expect(r2.onStop).toHaveBeenCalledWith("The connection to Deepgram was lost (1011: NET-0001).");
  });

  it("pauses the recorder while muted and keeps the socket alive; resumes on unmute", async () => {
    vi.useFakeTimers();
    let muted = false;
    const r = receiver(() => muted);
    const { svc, socket, recorder } = await connected(r);
    await vi.advanceTimersByTimeAsync(DEEPGRAM_KEEPALIVE_MS * 2);
    expect(socket.sent).toHaveLength(0); // recording: audio keeps the stream alive
    muted = true;
    await vi.advanceTimersByTimeAsync(DEEPGRAM_MUTE_POLL_MS);
    expect(recorder.state).toBe("paused");
    await vi.advanceTimersByTimeAsync(DEEPGRAM_KEEPALIVE_MS * 3);
    const keepAlives = socket.sent.filter((d) => d === JSON.stringify({ type: "KeepAlive" }));
    expect(keepAlives.length).toBeGreaterThanOrEqual(2);
    expect(socket.sent.every((d) => typeof d === "string")).toBe(true); // no audio while muted
    muted = false;
    await vi.advanceTimersByTimeAsync(DEEPGRAM_MUTE_POLL_MS);
    expect(recorder.state).toBe("recording");
    svc.stop();
  });

  it("a session that opens while muted records nothing until it is unmuted", async () => {
    vi.useFakeTimers();
    let muted = true;
    const r = receiver(() => muted);
    const { svc, socket, recorder } = await connected(r);
    // Checked synchronously when the socket opens: not even the first timeslice of muted audio is recorded.
    expect(recorder.state).toBe("inactive");
    expect(r.onStart).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(DEEPGRAM_KEEPALIVE_MS * 3);
    expect(recorder.state).toBe("inactive");
    const keepAlives = socket.sent.filter((d) => d === JSON.stringify({ type: "KeepAlive" }));
    expect(keepAlives.length).toBeGreaterThanOrEqual(2); // the socket stays open meanwhile
    expect(socket.sent.every((d) => typeof d === "string")).toBe(true);
    muted = false;
    await vi.advanceTimersByTimeAsync(DEEPGRAM_MUTE_POLL_MS);
    expect(recorder.state).toBe("recording");
    expect(recorder.timeslice).toBe(DEEPGRAM_TIMESLICE_MS);
    recorder.listeners.forEach((cb) => cb({ data: "webm-chunk" }));
    expect(socket.sent).toContain("webm-chunk");
    svc.stop();
    expect(track.stopped).toBe(true);
  });

  it("a recorder that ended on its own is not restarted and not kept alive", async () => {
    vi.useFakeTimers();
    const r = receiver();
    const { svc, socket, recorder } = await connected(r);
    recorder.state = "inactive"; // e.g. the microphone was unplugged
    await vi.advanceTimersByTimeAsync(DEEPGRAM_KEEPALIVE_MS * 3);
    expect(recorder.state).toBe("inactive");
    expect(socket.sent).toHaveLength(0); // Deepgram's no-audio timeout then ends the session with a message
    svc.stop();
  });

  it("microphone failures and a stop while the microphone opens", async () => {
    getUserMedia.mockRejectedValueOnce(new DOMException("Permission denied", "NotAllowedError"));
    const r = receiver();
    new STT_DeepgramService(r).start(settings(deepgram));
    await vi.waitFor(() => expect(r.onStop).toHaveBeenCalled());
    expect(r.onStop.mock.calls[0][0]).toContain("Microphone permission denied");

    // stop() before the microphone was requested: it is never opened.
    getUserMedia.mockClear();
    const r2 = receiver();
    const early = new STT_DeepgramService(r2);
    early.start(settings(deepgram));
    early.stop();
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(getUserMedia).not.toHaveBeenCalled();

    // stop() while getUserMedia is pending: the microphone is released as soon as it opens, and no socket is opened.
    let grant!: (stream: unknown) => void;
    getUserMedia.mockImplementationOnce(() => new Promise((resolve) => (grant = resolve)));
    const svc = new STT_DeepgramService(r2);
    svc.start(settings(deepgram));
    await vi.waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1));
    svc.stop();
    grant({ getTracks: () => [track] });
    await vi.waitFor(() => expect(track.stopped).toBe(true));
    expect(FakeSocket.instances).toHaveLength(0);
    expect(r2.onStop).not.toHaveBeenCalled();
  });

  it("pure message helpers", () => {
    expect(deepgramCloseMessage(true, 1000, "", true)).toBeNull();
    expect(deepgramCloseMessage(false, 1000, "", true)).toContain("Could not connect");
    expect(deepgramCloseMessage(true, 1006, "", false)).toBe("The connection to Deepgram was lost (1006).");
    expect(deepgramMicrophoneError(new DOMException("Requested device not found", "NotFoundError"))).toBe(
      "The selected microphone was not found. Pick another input device.",
    );
    expect(deepgramMicrophoneError(new Error("boom"))).toBe("Microphone: boom");
  });
});

// ---------------------------------------------------------------------------------------------------------------

describe("Web Speech (native)", () => {
  it("maps error codes; no-speech is not an error", () => {
    expect(nativeErrorMessage("no-speech", "en-US", true)).toBeNull();
    expect(nativeErrorMessage("network", "en-US", true)).toContain("desktop app");
    expect(nativeErrorMessage("network", "en-US", false)).toContain("connection");
    expect(nativeErrorMessage("language-not-supported", "tlh", false)).toContain("tlh");
    for (const code of ["not-allowed", "service-not-allowed", "audio-capture"]) expect(translated(nativeErrorMessage(code, "en", false))).toBe(true);
    expect(nativeErrorMessage("bad-grammar", "en", false)).toContain("bad-grammar");
  });

  it("reports the start, publishes results with the language, and stops with a readable error", () => {
    const instances: FakeRecognition[] = [];
    class FakeRecognition {
      lang = "";
      continuous = false;
      interimResults = false;
      onstart: (() => void) | null = null;
      onresult: ((e: unknown) => void) | null = null;
      onend: (() => void) | null = null;
      errorListener?: (e: unknown) => void;
      started = 0;
      stopped = 0;
      constructor() {
        instances.push(this);
      }
      addEventListener(_: string, cb: (e: unknown) => void) {
        this.errorListener = cb;
      }
      start() {
        this.started++;
      }
      stop() {
        this.stopped++;
      }
    }
    const listeners = new Map<string, unknown>();
    vi.stubGlobal("window", {
      webkitSpeechRecognition: FakeRecognition,
      setTimeout: (fn: () => void, ms: number) => setTimeout(fn, ms),
      addEventListener: (ev: string, fn: unknown) => listeners.set(ev, fn),
      removeEventListener: (ev: string) => listeners.delete(ev),
    });
    desktop = true;
    const r = receiver();
    const svc = new STT_NativeService(r);
    svc.start(settings((s) => (s.native.language = "de-DE")));
    const rec = instances[0];
    expect(rec.lang).toBe("de-DE");
    expect(r.onStatus).toHaveBeenCalledWith("Starting Web Speech…");
    expect(listeners.has("beforeunload")).toBe(true);
    rec.onstart!();
    expect(r.onStart).toHaveBeenCalledTimes(1);
    rec.onresult!({ resultIndex: 0, results: [Object.assign([{ transcript: "hallo" }], { isFinal: true })] });
    expect(r.onFinal).toHaveBeenCalledWith("hallo", { lang: "de" });
    rec.errorListener!({ error: "no-speech" });
    expect(r.onStop).not.toHaveBeenCalled();
    rec.errorListener!({ error: "network" });
    expect(r.onStop).toHaveBeenCalledWith(expect.stringContaining("desktop app"));
    expect(rec.stopped).toBe(1);
    expect(listeners.has("beforeunload")).toBe(false);
    // A later error of the stopped instance does not stop twice.
    rec.errorListener!({ error: "aborted" });
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------------------------------------------

describe("Chrome / Edge microphone page", () => {
  it("says what it waits for, and sends start / stop to the page", () => {
    const broadcast = vi.fn();
    vi.stubGlobal("window", { ApiShared: { pubsub: { broadcastLocalWebSocketOnly: broadcast } } });
    const r = receiver();
    const svc = new STT_ExternalMicService(r);
    svc.start(settings((s) => (s.backend = STT_Backends.edge)));
    expect(r.onStatus.mock.calls[0][0]).toContain("Edge");
    expect(translated(r.onStatus.mock.calls[0][0])).toBe(true);
    // Nothing clears the line when the page connects or gives up: it expires with the connect timeout.
    expect(r.onStatus.mock.calls[0][1]).toEqual({ expireMs: EXTERNAL_MIC_WAITING_LINE_MS });
    expect(broadcast).toHaveBeenLastCalledWith({ topic: "stt.external.command", data: { cmd: "start" } });
    svc.stop();
    expect(broadcast).toHaveBeenLastCalledWith({ topic: "stt.external.command", data: { cmd: "stop" } });
    expect(r.onStatus).toHaveBeenLastCalledWith("");
    expect(r.onStop).toHaveBeenCalledTimes(1);
  });
});
