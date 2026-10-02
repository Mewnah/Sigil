import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { proxy } from "valtio";

const sendOsc = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => ({ ok: true })) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/core/services/audioDevices", () => ({ audioDevicesApi: { setCaptureMute: vi.fn(async () => {}) } }));
vi.mock("../targets/utils", () => ({ sendOsc: (...args: unknown[]) => sendOsc(...args) }));

import { TextEventSource, TextEventType, type TextEvent } from "@/types";
import { Service_VRC_Schema } from "../schema";

const { default: Service_VRC } = await import("../index");

type Subscriber = { topic: string; fn: (data?: TextEvent, topic?: string) => void };

/** pubsub-js as `subscribeText` sees it: exact topic first, then each parent level, always with the original topic. */
function miniPubSub() {
  const subs = new Map<string, Subscriber>();
  let next = 0;
  return {
    subscribeText(topic: string, fn: Subscriber["fn"]) {
      const key = `sub${next++}`;
      subs.set(key, { topic, fn });
      return key;
    },
    unsubscribe(key?: string) {
      if (key) subs.delete(key);
    },
    publish(topic: string, data: TextEvent) {
      const levels = [topic];
      for (let i = topic.lastIndexOf("."); i !== -1; i = topic.lastIndexOf(".", i - 1)) levels.push(topic.slice(0, i));
      for (const level of levels) {
        for (const s of [...subs.values()]) if (s.topic === level) s.fn(data, topic);
      }
    },
  };
}

const final = (value: string, extra: Partial<TextEvent> = {}): TextEvent => ({
  type: TextEventType.final,
  value,
  emotes: {},
  ...extra,
});

const chatbox = () => sendOsc.mock.calls.filter(([path]) => path === "/chatbox/input").map(([, args]) => args[0].String);

let pubsub: ReturnType<typeof miniPubSub>;
let state: ReturnType<typeof Service_VRC_Schema.parse>;

describe("Service_VRC chatbox routing", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sendOsc.mockReset();
    pubsub = miniPubSub();
    state = proxy(Service_VRC_Schema.parse({}));
    state.enable = true;
    state.textbox.indicator = false;
    state.textbox.commitOnTurnEnd = true;
    vi.stubGlobal("window", {
      Config: { isApp: () => false, isServer: () => true, isClient: () => false },
      ApiShared: { pubsub },
      ApiServer: { state: { services: { vrc: { data: state }, translation: { data: {} } } } },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it.each([TextEventSource.any, TextEventSource.textfield])(
    "typed text through the %s source is its own message, never joined into a held spoken turn",
    async (source) => {
      state.source = source;
      const service = new Service_VRC();
      await service.init();
      if (source === TextEventSource.any) {
        pubsub.publish(TextEventSource.stt, final("I think that", { turnEnd: false }));
        expect(chatbox()).toEqual([]);
      }
      pubsub.publish(TextEventSource.textfield, final("brb"));
      vi.advanceTimersByTime(2_000);
      expect(chatbox()).toEqual(source === TextEventSource.any ? ["I think that", "brb"] : ["brb"]);
      service.dispose();
    },
  );

  it("spoken finals through the 'any text' source are still held for the end of the turn", async () => {
    state.source = TextEventSource.any;
    const service = new Service_VRC();
    await service.init();
    pubsub.publish(TextEventSource.stt, final("I think that", { turnEnd: false }));
    pubsub.publish(TextEventSource.stt, final("we should go.", { turnEnd: true }));
    expect(chatbox()).toEqual(["I think that we should go."]);
    service.dispose();
  });
});
