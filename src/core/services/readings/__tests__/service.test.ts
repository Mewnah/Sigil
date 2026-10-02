import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { proxy } from "valtio";
import {
  OthersEventSource,
  ReadingsEventSource,
  TextEventSource,
  TextEventType,
  type ReadingsEvent,
  type RustReadingsRequest,
  type RustReadingsResult,
  type RustReadingsStatus,
} from "@/types";

const invoke = vi.hoisted(() => vi.fn());
vi.mock("@tauri-apps/api/core", () => ({ invoke, isTauri: () => true }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warn: vi.fn(), warning: vi.fn(), success: vi.fn(), dismiss: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));

import PubSub from "pubsub-js";
import Service_PubSub from "@/shared/services/pubsub";
import Service_Readings, { JAPANESE_DICTIONARY_ID } from "..";
import { Readings_Schema } from "../schema";

/** Service wiring (P6-L): topics, settings changes, the dictionary check and unloading. Real pubsub and pubsub-js. */

const settle = async () => {
  for (let i = 0; i < 10; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

let pubsub: Service_PubSub;
let published: ReadingsEvent[];
let tokens: string[];
let status: RustReadingsStatus;
let state: { readings: ReturnType<typeof Readings_Schema.parse> };
let models: { states: Record<string, { state: string }> };

function annotateCalls(): RustReadingsRequest[] {
  return invoke.mock.calls
    .filter(([cmd]) => cmd === "plugin:text-readings|text_readings_annotate")
    .map(([, args]) => (args as { request: RustReadingsRequest }).request);
}

function calls(cmd: string): number {
  return invoke.mock.calls.filter(([c]) => c === `plugin:text-readings|${cmd}`).length;
}

beforeEach(() => {
  invoke.mockReset();
  status = { japanese: true, chinese: true, japaneseModel: JAPANESE_DICTIONARY_ID, error: null };
  invoke.mockImplementation(async (cmd: string, args?: { request?: RustReadingsRequest }) => {
    if (cmd === "plugin:text-readings|text_readings_status") return status;
    if (cmd === "plugin:text-readings|text_readings_unload") return null;
    if (cmd === "plugin:text-readings|text_readings_annotate") {
      const r = args!.request!;
      const result: RustReadingsResult =
        r.lang === "ja"
          ? { lang: "ja", hiragana: `h:${r.text}`, romaji: r.romaji ? `r:${r.text}` : null, pinyin: null, segments: [] }
          : { lang: "zh", hiragana: null, romaji: null, pinyin: `p:${r.text}`, segments: [{ text: r.text, reading: "x" }] };
      return result;
    }
    return undefined;
  });
  state = proxy({ readings: Readings_Schema.parse({}) });
  models = proxy({ states: {} as Record<string, { state: string }> });
  vi.stubGlobal("window", {
    Config: { isServer: () => true, isApp: () => false, isClient: () => false, serverNetwork: {} },
    ApiShared: { peer: { broadcastPubSubParts: vi.fn() } },
    ApiServer: { state, models: { state: models } },
  });
  pubsub = new Service_PubSub();
  (window as unknown as { ApiShared: { pubsub: Service_PubSub } }).ApiShared.pubsub = pubsub;
  published = [];
  tokens = [PubSub.subscribe(ReadingsEventSource.stt, (_: string, data: ReadingsEvent) => void published.push(data))];
});

afterEach(() => {
  tokens.forEach((t) => PubSub.unsubscribe(t));
  PubSub.clearAllSubscriptions();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Service_Readings", () => {
  it("does nothing while off (the default)", async () => {
    const service = new Service_Readings();
    await service.init();
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "今日は", lang: "ja", segmentId: "mic:1:1" });
    await settle();
    expect(invoke).not.toHaveBeenCalled();
    expect(published).toHaveLength(0);
  });

  it("annotates the selected sources once enabled and publishes on readings.stt", async () => {
    const service = new Service_Readings();
    await service.init();
    state.readings.enable = true;
    state.readings.sources = [TextEventSource.stt, OthersEventSource.stt];
    await settle();
    expect(calls("text_readings_status")).toBeGreaterThan(0);
    expect(service.state.japaneseInstalled).toBe(true);
    // The warm-up loads the dictionary before the first line.
    expect(annotateCalls().some((r) => r.text === "あ")).toBe(true);
    invoke.mockClear();

    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "今日は", lang: "ja", segmentId: "mic:1:1", revision: 2 });
    pubsub.publishText(OthersEventSource.stt, { type: TextEventType.final, value: "你好", lang: "zh", segmentId: "loopback:1:1" });
    // Not selected: translations.
    pubsub.publishText(TextEventSource.translation, { type: TextEventType.final, value: "今日は", lang: "ja", segmentId: "t:1:1" });
    await settle();
    expect(annotateCalls().map((r) => r.text)).toEqual(["今日は", "你好"]);
    expect(published).toEqual([
      { utteranceId: "mic:1:1", revision: 2, source: "text.stt", lang: "ja", value: "今日は", hiragana: "h:今日は", romaji: "r:今日は" },
      { utteranceId: "loopback:1:1", source: "others.stt", lang: "zh", value: "你好", pinyin: "p:你好", segments: [{ text: "你好", reading: "x" }] },
    ]);
    expect(service.state.published).toBe(2);
    expect(service.state.last?.utteranceId).toBe("loopback:1:1");
  });

  it("keys lines heard on a parent topic by their own topic", async () => {
    const service = new Service_Readings();
    await service.init();
    state.readings.enable = true;
    // The parent and one of its children: each line is still read once.
    state.readings.sources = [OthersEventSource.any, OthersEventSource.stt];
    await settle();
    invoke.mockClear();
    pubsub.publishText(OthersEventSource.translation, { type: TextEventType.final, value: "東京です", lang: "ja", segmentId: "o:1:1" });
    pubsub.publishText(OthersEventSource.stt, { type: TextEventType.final, value: "大阪", lang: "ja", segmentId: "o:1:2" });
    await settle();
    expect(published.map((e) => e.source)).toEqual(["others.translation", "others.stt"]);
    expect(annotateCalls()).toHaveLength(2);
  });

  it("skips Japanese while the dictionary is missing and re-checks when it is installed", async () => {
    status = { ...status, japanese: false };
    const service = new Service_Readings();
    await service.init();
    state.readings.enable = true;
    await settle();
    expect(service.state.japaneseInstalled).toBe(false);
    invoke.mockClear();
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "東京", lang: "ja", segmentId: "m:1:1" });
    await settle();
    expect(annotateCalls()).toHaveLength(0);

    status = { ...status, japanese: true };
    models.states[JAPANESE_DICTIONARY_ID] = { state: "installed" };
    await settle();
    expect(service.state.japaneseInstalled).toBe(true);
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "東京", lang: "ja", segmentId: "m:1:2" });
    await settle();
    expect(published.map((e) => e.utteranceId)).toEqual(["m:1:2"]);
  });

  it("unsubscribes when switched off and frees the dictionaries a minute later", async () => {
    const service = new Service_Readings();
    await service.init();
    state.readings.enable = true;
    await settle();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    state.readings.enable = false;
    await vi.advanceTimersByTimeAsync(10);
    invoke.mockClear();
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "東京", lang: "ja", segmentId: "m:2:1" });
    await vi.advanceTimersByTimeAsync(10);
    expect(annotateCalls()).toHaveLength(0);
    expect(calls("text_readings_unload")).toBe(0);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls("text_readings_unload")).toBe(1);

    // Switching back on within the minute keeps them loaded.
    state.readings.enable = true;
    await vi.advanceTimersByTimeAsync(10);
    state.readings.enable = false;
    await vi.advanceTimersByTimeAsync(10);
    state.readings.enable = true;
    await vi.advanceTimersByTimeAsync(120_000);
    expect(calls("text_readings_unload")).toBe(1);
  });

  it("treats every source unchecked like off: no warm-up, and switching off afterwards still frees the dictionaries", async () => {
    const service = new Service_Readings();
    await service.init();
    state.readings.enable = true;
    await settle();
    expect(annotateCalls().some((r) => r.text === "あ")).toBe(true);
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    invoke.mockClear();

    state.readings.sources = [];
    await vi.advanceTimersByTimeAsync(10);
    // Nothing will be read: no warm-up, no line annotated, but the dictionary status stays current.
    expect(annotateCalls()).toHaveLength(0);
    expect(calls("text_readings_status")).toBe(1);
    pubsub.publishText(TextEventSource.stt, { type: TextEventType.final, value: "東京", lang: "ja", segmentId: "m:3:1" });
    await vi.advanceTimersByTimeAsync(10);
    expect(annotateCalls()).toHaveLength(0);

    // "Enabled with no source" and "off" are different keys: switching off is not swallowed.
    state.readings.enable = false;
    await vi.advanceTimersByTimeAsync(10);
    expect(calls("text_readings_unload")).toBe(0);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls("text_readings_unload")).toBe(1);
    expect(annotateCalls()).toHaveLength(0);
  });

  it("enabled with no source at startup loads nothing and frees nothing", async () => {
    state.readings.enable = true;
    state.readings.sources = [];
    const service = new Service_Readings();
    await service.init();
    await settle();
    expect(annotateCalls()).toHaveLength(0);
    expect(service.state.japaneseInstalled).toBe(true);

    // Adding a source starts reading (and warms the dictionary up); switching off frees it a minute later.
    state.readings.sources = [TextEventSource.stt];
    await settle();
    expect(annotateCalls().map((r) => r.text)).toEqual(["あ"]);
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    state.readings.enable = false;
    await vi.advanceTimersByTimeAsync(60_010);
    expect(calls("text_readings_unload")).toBe(1);
  });

  it("does not schedule an unload at startup while off", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    const service = new Service_Readings();
    await service.init();
    await vi.advanceTimersByTimeAsync(120_000);
    expect(invoke).not.toHaveBeenCalled();
  });
});
