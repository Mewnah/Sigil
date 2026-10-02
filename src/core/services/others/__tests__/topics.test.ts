import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { proxy } from "valtio";
import {
  OthersEventSource,
  TextEventSource,
  TextEventType,
  type RustSpeechFinal,
  type RustSpeechPartial,
  type RustSpeechPipelineConfig,
  type RustSpeechRoutePlan,
  type TextEvent,
} from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => undefined), isTauri: () => false }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warn: vi.fn(), warning: vi.fn(), success: vi.fn(), dismiss: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import PubSub from "pubsub-js";
import Service_PubSub from "@/shared/services/pubsub";
import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { createTranslationPipeline, type TranslationEngineMap } from "@/core/services/translation/pipeline";
import { Service_Translation_Schema, Translation_Backends, type Translation_State } from "@/core/services/translation/schema";
import type { ITranslationEngine, TranslationRequest } from "@/core/services/translation/types";
import { OthersSpeechDriver, OTHERS_PIPELINE_ID, type OthersDriverDeps } from "../driver";
import { OthersPublisher } from "../publish";
import { Service_Others_Schema } from "../schema";
import { othersTranslationPair } from "../translation";

/**
 * Acceptance (P6-A): other players' text travels only on `others.*`. pubsub-js bubbles `text.*` up to `text`, the topic
 * the chatbox, TTS, history and every "any text" consumer listen to; `others.*` is outside that hierarchy, so none of
 * them ever hears another player. Uses the real pubsub service and pubsub-js.
 */

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

class EchoEngine implements ITranslationEngine {
  static requests: TranslationRequest[] = [];
  async start() {}
  async translate(request: TranslationRequest) {
    EchoEngine.requests.push(request);
    return `[${request.to}] ${request.text}`;
  }
  stop() {}
}

const PLAN: RustSpeechRoutePlan = {
  language: "auto",
  tier: "cpu",
  streaming: { engine: "sherpa-online", model: "stt/nemotron-3.5-streaming-560ms-int8" },
  final: null,
  fallback: null,
  languageId: null,
  models: [],
  missing: [],
  vramMb: 0,
  notes: [],
};

let pubsub: Service_PubSub;
let heard: Record<string, { topic: string; event: TextEvent }[]>;
let tokens: string[];

function listen(topic: string) {
  heard[topic] = [];
  // `allowEmpty`: even an empty event on a topic must never arrive.
  tokens.push(pubsub.subscribeText(topic as TextEventSource, (event, eventName) => void heard[topic].push({ topic: String(eventName), event: event! }), true));
}

beforeAll(async () => {
  await initSttTestI18n();
});

beforeEach(() => {
  vi.stubGlobal("window", {
    Config: { isServer: () => true, isApp: () => false, isClient: () => false, serverNetwork: {} },
    ApiShared: { peer: { broadcastPubSubParts: vi.fn() } },
    ApiServer: { state: {} },
  });
  pubsub = new Service_PubSub();
  heard = {};
  tokens = [];
  for (const topic of [
    TextEventSource.any,
    TextEventSource.stt,
    TextEventSource.translation,
    OthersEventSource.any,
    OthersEventSource.stt,
    OthersEventSource.translation,
  ])
    listen(topic);
  EchoEngine.requests = [];
});

afterEach(() => {
  tokens.forEach((t) => PubSub.unsubscribe(t));
  vi.unstubAllGlobals();
});

describe("other players' topics", () => {
  it("never reach 'text' subscribers (chatbox, TTS, history), only 'others'", async () => {
    const translationSettings: Translation_State = proxy(Service_Translation_Schema.parse({}));
    translationSettings.backend = Translation_Backends.llm;
    translationSettings.sourceLanguage = "en"; // the user's own language
    translationSettings.targetLanguage = "ja"; // what the user's captions are translated into
    translationSettings.partials.throttleMs = 0;
    const others = Service_Others_Schema.parse({ translate: true });
    const engines = Object.fromEntries(Object.values(Translation_Backends).map((b) => [b, EchoEngine])) as unknown as TranslationEngineMap;
    const translation = createTranslationPipeline({
      name: "others",
      settings: () => translationSettings,
      engines,
      publish: (event) => pubsub.publishText(OthersEventSource.translation, event),
      pair: () =>
        othersTranslationPair({
          othersLanguage: others.language,
          translateTo: others.translateTo,
          mainSource: translationSettings.sourceLanguage,
          sttLanguage: "auto",
          appLanguage: "en",
        }),
      toasts: false,
    });
    const publisher = new OthersPublisher({
      publish: (topic, event) => pubsub.publishText(topic, event),
      translate: (stream, event) => translation.push(stream, event),
      interimEnabled: () => others.interim,
    });

    // The driver, fed by fake `speech:*` events of the loopback pipeline.
    const handlers = new Map<string, (p: unknown) => void>();
    const deps: Partial<OthersDriverDeps> = {
      available: () => true,
      listen: async <T,>(event: string, handler: (payload: T) => void) => {
        handlers.set(event, handler as (p: unknown) => void);
        return () => handlers.delete(event);
      },
      invoke: async <T,>(cmd: string, args?: Record<string, unknown>): Promise<T> => {
        if (cmd === "plugin:speech|preview_route") return PLAN as T;
        if (cmd === "plugin:speech|get_pipeline_status") return [] as T;
        if (cmd === "plugin:audio|list_loopback_processes")
          return { supported: true, processes: [], plan: { requested: "process:VRChat.exe", source: "process:VRChat.exe", fallback: null, watch: "VRChat.exe", running: true } } as T;
        if (cmd === "plugin:speech|start_pipeline")
          return { pipelineId: (args!.config as RustSpeechPipelineConfig).pipelineId, epoch: 1, plan: PLAN } as T;
        return undefined as T;
      },
    };
    const driver = new OthersSpeechDriver(
      { interim: (t, m) => publisher.interim(t, m), final: (t, m) => publisher.final(t, m) },
      () => {},
      deps,
    );
    translation.start();
    await driver.start(others);
    await settle();

    const partial: RustSpeechPartial = { pipelineId: OTHERS_PIPELINE_ID, epoch: 1, segmentId: 1, revision: 1, text: "こんにちは", lang: "ja" };
    handlers.get("speech:partial")!(partial);
    const final: RustSpeechFinal = {
      ...partial,
      revision: 2,
      text: "こんにちは、元気？",
      tags: [],
      pass: "first",
      engine: "sherpa-online",
      model: "m",
      startMs: 0,
      endMs: 900,
      forced: false,
      latencyMs: 50,
      turnEnd: null,
    };
    handlers.get("speech:final")!(final);
    await settle();

    // Other players' captions and their translation arrived on others.* ...
    expect(heard[OthersEventSource.stt].map((h) => [h.event.type, h.event.value])).toEqual([
      [TextEventType.interim, "こんにちは"],
      [TextEventType.final, "こんにちは、元気？"],
    ]);
    const translated = heard[OthersEventSource.translation].filter((h) => h.event.type === TextEventType.final);
    expect(translated.map((h) => h.event.value)).toEqual(["[en] こんにちは、元気？"]);
    expect(translated[0].event.lang).toBe("en");
    // ... translated the other way round: from their language (detected per line) into the user's.
    const finalRequest = EchoEngine.requests.find((r) => r.kind === "final")!;
    expect([finalRequest.from, finalRequest.to]).toEqual(["ja", "en"]);
    // "others" (the bubbling parent) hears both.
    expect(new Set(heard[OthersEventSource.any].map((h) => h.topic))).toEqual(new Set([OthersEventSource.stt, OthersEventSource.translation]));

    // ... and never on text.*: the chatbox, TTS, history and "any text" consumers heard nothing.
    expect(heard[TextEventSource.any]).toEqual([]);
    expect(heard[TextEventSource.stt]).toEqual([]);
    expect(heard[TextEventSource.translation]).toEqual([]);

    driver.dispose();
    translation.stop();
  });

  it("the user's own captions still reach 'text' while other players run", () => {
    const publisher = new OthersPublisher({
      publish: (topic, event) => pubsub.publishText(topic, event),
      translate: () => {},
      interimEnabled: () => true,
    });
    publisher.final("their words", {});
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "my words" });
    expect(heard[TextEventSource.any].map((h) => [h.topic, h.event.value])).toEqual([[TextEventSource.stt, "my words"]]);
    expect(heard[TextEventSource.stt].map((h) => h.event.value)).toEqual(["my words"]);
    expect(heard[OthersEventSource.stt].map((h) => h.event.value)).toEqual(["their words"]);
    expect(heard[OthersEventSource.any].map((h) => h.event.value)).toEqual(["their words"]);
  });
});
