import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Service_Peer on a /client page: `startClient` answers with the host's first `peers:init_data`,
 * a refused LAN token is the only thing that rejects it, and a copy of `peers:init_data` after the
 * first (the host sends one on every channel it opens, i.e. after every reconnect) is applied so a
 * mirror that missed a scene switch while it was away shows the right scene again.
 */

vi.mock("react-toastify", () => ({ toast: { success: vi.fn() } }));
vi.mock("i18next", () => ({ default: { t: (key: string) => key, language: "en", changeLanguage: vi.fn(async () => {}) } }));
vi.mock("../provider", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../provider")>();
  class FakeProvider extends EventTarget {
    static last?: FakeProvider;
    /** What `connectClient` does; tests replace it before `startClient`. */
    static connect: (params: unknown) => Promise<void> = async () => {};
    connectClient = vi.fn((params: unknown) => FakeProvider.connect(params));
    dispose = vi.fn();
    constructor(readonly document: unknown) {
      super();
      FakeProvider.last = this;
    }
  }
  return { ...actual, PeerjsProvider: FakeProvider };
});

import i18n from "i18next";
import Service_Peer, { MirrorRefusedError, MirrorStartError } from "..";
import { PEER_CLIENT_MIRROR_EVENT, PEER_INVALID_KEY, PeerjsProvider } from "../provider";

const Fake = PeerjsProvider as unknown as {
  last: EventTarget & { connectClient: ReturnType<typeof vi.fn> };
  connect: (params: unknown) => Promise<void>;
};

let scenes: { state: { activeScene: string } };
let config: { clientNetwork: { host: string; port: string; token: string }; clientInitialState?: unknown };

beforeEach(() => {
  scenes = { state: { activeScene: "main" } };
  config = { clientNetwork: { host: "localhost", port: "3030", token: "tok" } };
  (globalThis as any).window = {
    ApiClient: { document: { file: {} }, scenes },
    Config: config,
    ApiShared: { pubsub: { publishLocally: vi.fn() } },
  };
  (i18n as any).language = "en";
  Fake.connect = async () => {};
});

const hostSends = (topic: string, data: unknown) =>
  Fake.last.dispatchEvent(new CustomEvent("on_event_received", { detail: { topic, data } }));

const mirrorEvent = (detail: unknown) => Fake.last.dispatchEvent(new CustomEvent(PEER_CLIENT_MIRROR_EVENT, { detail }));

describe("Service_Peer.startClient", () => {
  it("answers with the first peers:init_data, however late the channel opened", async () => {
    let opened!: () => void;
    Fake.connect = () => new Promise<void>((resolve) => (opened = resolve));
    const service = new Service_Peer();
    const started = service.startClient();
    let result: unknown;
    started.then((value) => (result = value));
    // The first attempts failed and were retried inside the provider: still waiting, no rejection.
    await Promise.resolve();
    expect(result).toBeUndefined();

    // The host sends the config the moment the channel opens, even ahead of connectClient's resolution.
    hostSends("peers:init_data", { scene: "intro", uiLanguage: "de" });
    opened();
    await expect(started).resolves.toEqual({ scene: "intro", uiLanguage: "de" });
    // The first copy only answers the start-up (Service_Scenes.init applies it): not applied here.
    expect(scenes.state.activeScene).toBe("main");
    expect(i18n.changeLanguage).not.toHaveBeenCalled();
    expect(Fake.last.connectClient).toHaveBeenCalledWith({ id: "server", host: "localhost", port: "3030", token: "tok" });
  });

  it("applies a later peers:init_data after a reconnect: scene, language and the config others read", async () => {
    const service = new Service_Peer();
    const started = service.startClient();
    hostSends("peers:init_data", { scene: "main", uiLanguage: "en" });
    await started;

    // The mirror lost the host, the user switched scene (and language) meanwhile, the mirror is back.
    hostSends("peers:init_data", { scene: "intro", uiLanguage: "es" });
    expect(scenes.state.activeScene).toBe("intro");
    expect(i18n.changeLanguage).toHaveBeenCalledWith("es");
    expect(config.clientInitialState).toEqual({ scene: "intro", uiLanguage: "es" });

    // Same language: nothing to change. A scene the document does not hold yet is still taken (the
    // host sends the config before the document; the elements draw once it arrives).
    (i18n as any).language = "es";
    hostSends("peers:init_data", { scene: "brand-new", uiLanguage: "es" });
    expect(i18n.changeLanguage).toHaveBeenCalledTimes(1);
    expect(scenes.state.activeScene).toBe("brand-new");
  });

  it("ignores a later peers:init_data that is not a config", async () => {
    const service = new Service_Peer();
    const started = service.startClient();
    hostSends("peers:init_data", { scene: "main" });
    await started;
    for (const junk of [null, 7, "scene", { scene: 3, uiLanguage: {} }, { scene: "" }]) hostSends("peers:init_data", junk);
    expect(scenes.state.activeScene).toBe("main");
    expect(i18n.changeLanguage).not.toHaveBeenCalled();
  });

  it("rejects with MirrorRefusedError, and never with the token, when the host refuses the LAN token", async () => {
    Fake.connect = async () => {
      // PeerJS's own message quotes the key, i.e. the LAN token.
      throw Object.assign(new Error('API KEY "s3cr3t-lan-token" is invalid'), { type: PEER_INVALID_KEY });
    };
    const service = new Service_Peer();
    const error = await service.startClient().catch((e) => e);
    expect(error).toBeInstanceOf(MirrorRefusedError);
    expect(error.message).toBe("client.peer_invalid_key");
    expect(error.message + error.stack).not.toContain("s3cr3t");
    expect(service.clientMirrorState.phase).toBe("failed");
    expect(service.clientMirrorState.lastError).toBe("client.peer_invalid_key");
  });

  it("wraps any other start-up error in MirrorStartError: no retry can fix it, and the page says so", async () => {
    const boom = new Error("cannot construct a Peer");
    Fake.connect = async () => {
      throw boom;
    };
    const service = new Service_Peer();
    const error = await service.startClient().catch((e) => e);
    expect(error).toBeInstanceOf(MirrorStartError);
    expect(error).not.toBeInstanceOf(MirrorRefusedError);
    expect(error.message).toBe("cannot construct a Peer");
    expect(error.original).toBe(boom);
    expect(service.clientMirrorState).toMatchObject({ phase: "failed", lastError: "cannot construct a Peer" });

    // No WebRTC at all: PeerJS's own error, same treatment.
    Fake.connect = async () => {
      throw Object.assign(new Error("Your browser does not support some WebRTC features"), { type: "browser-incompatible" });
    };
    expect(await new Service_Peer().startClient().catch((e) => e)).toBeInstanceOf(MirrorStartError);
  });

  it("never shows the LAN token in a start-up error (the browser quotes the whole URL)", async () => {
    Fake.connect = async () => {
      throw new SyntaxError("Failed to construct 'WebSocket': The URL 'ws://bad host:3030/peer/peerjs?key=tok&id=x' is invalid.");
    };
    const service = new Service_Peer();
    const error = await service.startClient().catch((e) => e);
    expect(error).toBeInstanceOf(MirrorStartError);
    expect(error.message).toContain("ws://bad host:3030/peer/peerjs?key=…&id=x");
    expect(error.message + error.stack + service.clientMirrorState.lastError).not.toContain("key=tok");
  });

  it("follows the mirror's connection phases for the banner", async () => {
    const service = new Service_Peer();
    const started = service.startClient();
    hostSends("peers:init_data", { scene: "main" });
    await started;
    const phase = () => [service.clientMirrorState.phase, service.clientMirrorState.lastError];

    mirrorEvent({ phase: "synced" });
    expect(phase()).toEqual(["synced", ""]);
    mirrorEvent({ phase: "reconnecting" });
    expect(phase()).toEqual(["reconnecting", ""]);
    mirrorEvent({ phase: "error", message: "network" });
    expect(phase()).toEqual(["reconnecting", "network"]);
    mirrorEvent({ phase: "failed", message: PEER_INVALID_KEY });
    expect(phase()).toEqual(["failed", "client.peer_invalid_key"]);
    mirrorEvent({ phase: "synced" });
    expect(phase()).toEqual(["synced", ""]);
  });

  it("counts the failed attempts for the loading view, and starts over once in sync", async () => {
    const service = new Service_Peer();
    const started = service.startClient();
    expect(service.clientMirrorState.failures).toBe(0);
    // Sigil is not listening: every retry inside the page fails the same way.
    for (let i = 1; i <= 4; i++) {
      mirrorEvent({ phase: "error", message: "network" });
      expect(service.clientMirrorState.failures).toBe(i);
    }
    expect(service.clientMirrorState.lastError).toBe("network");
    // A channel that closed is not a failed attempt.
    mirrorEvent({ phase: "reconnecting" });
    expect(service.clientMirrorState.failures).toBe(4);
    hostSends("peers:init_data", { scene: "main" });
    await started;
    mirrorEvent({ phase: "synced" });
    expect(service.clientMirrorState.failures).toBe(0);
  });
});
