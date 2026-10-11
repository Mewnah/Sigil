import PubSub from "pubsub-js";
import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TextEventSource, TextEventType, type TextEvent } from "@/types";

const posted = vi.hoisted(() => ({ chunks: [] as string[][] }));

vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("../discord/webhookExecute", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../discord/webhookExecute")>()),
  postDiscordContentChunks: vi.fn(async (_url: string, chunks: string[]) => {
    posted.chunks.push(chunks);
    return { ok: true } as Response;
  }),
}));

import Service_Discord from "../discord";
import { Service_Discord_Schema } from "../discord/schema";
import { postableFinal } from "../postText";

const ev = (type: TextEventType, value: string): TextEvent => ({ type, value, emotes: {} });

describe("postableFinal", () => {
  it("returns the text of a final", () => {
    expect(postableFinal(ev(TextEventType.final, "hello"))).toBe("hello");
  });

  it("is null for STT's cancel marker (a sentence cut off by a mute or a stop), however it is padded", () => {
    expect(postableFinal(ev(TextEventType.final, "[...]"))).toBeNull();
    expect(postableFinal(ev(TextEventType.final, "  [...] "))).toBeNull();
  });

  it("is null for interims, empty text and nothing; a sentence that merely contains the marker is posted", () => {
    expect(postableFinal(ev(TextEventType.interim, "hello"))).toBeNull();
    expect(postableFinal(ev(TextEventType.final, ""))).toBeNull();
    expect(postableFinal(undefined)).toBeNull();
    expect(postableFinal(ev(TextEventType.final, "wait [...] what"))).toBe("wait [...] what");
  });
});

describe("chat posting never carries the cancel marker", () => {
  beforeEach(() => {
    posted.chunks = [];
    const data = proxy(Service_Discord_Schema.parse({}));
    data.postEnable = true;
    data.postSource = TextEventSource.stt;
    data.channelHook = "https://discord.com/api/webhooks/123456789/abcdefghijklmnopqrstuvwxyz";
    vi.stubGlobal("window", {
      ApiServer: { state: { services: { discord: { data } } } },
      ApiShared: {
        pubsub: {
          subscribeText: (topic: string, fn: (e?: TextEvent) => void) => PubSub.subscribe(topic, (_name, e: TextEvent) => fn(e)),
          unsubscribe: (key?: string) => void (key && PubSub.unsubscribe(key)),
        },
      },
    });
  });

  afterEach(() => {
    PubSub.clearAllSubscriptions();
    vi.unstubAllGlobals();
  });

  it("a Discord webhook gets the sentence, not the '[...]' that closes a muted one", async () => {
    const service = new Service_Discord();
    await service.init();
    PubSub.publishSync(TextEventSource.stt, ev(TextEventType.interim, "I was just say"));
    PubSub.publishSync(TextEventSource.stt, ev(TextEventType.final, "[...]"));
    PubSub.publishSync(TextEventSource.stt, ev(TextEventType.final, "Back again."));
    await vi.waitFor(() => expect(posted.chunks).toEqual([["Back again."]]));
    service.dispose();
  });
});
