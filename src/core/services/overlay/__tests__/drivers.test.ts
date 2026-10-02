import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TextEventSource, TextEventType, type CaptionEventSource, type TextEvent } from "@/types";
import { Service_VRC_Schema } from "../../vrc/schema";
import { CaptionWindow_Schema, type CaptionWindowState } from "../schema";
import type { CaptionBus } from "../feed";
import type { VrOverlaySettings } from "../config";

/** The overlay service's drivers (P6-B) against mocked Tauri APIs: what they ask Rust for, in which order, and when. */

const tauri = vi.hoisted(() => ({
  calls: [] as { cmd: string; args?: Record<string, unknown> }[],
  emitted: [] as { label: string; event: string; payload: { seq: number; lines: { text: string }[]; unlocked: boolean } }[],
  respond: (_cmd: string, _args?: Record<string, unknown>): unknown => undefined,
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: async (cmd: string, args?: Record<string, unknown>) => {
    tauri.calls.push({ cmd, args });
    return tauri.respond(cmd, args);
  },
  isTauri: () => true,
}));
vi.mock("@tauri-apps/api/event", () => ({
  emitTo: async (label: string, event: string, payload: (typeof tauri.emitted)[number]["payload"]) => {
    tauri.emitted.push({ label, event, payload });
  },
  listen: async () => () => { },
}));
vi.mock("i18next", () => ({ default: { t: (key: string) => key, on: () => { } } }));
vi.mock("../../systemLog", () => ({ pushSystemLog: () => { } }));

const { CaptionWindowDriver, VrOverlayDriver, FRAME_COALESCE_MS, FRAME_HEARTBEAT_MS, VR_TEXT_INTERVAL_MS } = await import("../drivers");
type OverlayUiState = import("../drivers").OverlayUiState;

/** A pubsub that bubbles `a.b` to `a` like pubsub-js. */
class FakeBus implements CaptionBus {
  subs = new Map<string, { source: string; fn: (e: TextEvent | undefined, topic: string | undefined) => void }>();
  #next = 0;
  subscribe(source: CaptionEventSource, fn: (e: TextEvent | undefined, topic: string | undefined) => void): string {
    const token = `t${this.#next++}`;
    this.subs.set(token, { source, fn });
    return token;
  }
  unsubscribe(token: string): void {
    this.subs.delete(token);
  }
  publish(topic: string, event: TextEvent): void {
    for (const { source, fn } of [...this.subs.values()]) {
      if (topic === source || topic.startsWith(`${source}.`)) fn(event, topic);
    }
  }
}

const final = (value: string, segmentId: string): TextEvent => ({ type: TextEventType.final, value, emotes: {}, segmentId });
const interim = (value: string, segmentId: string, revision: number): TextEvent => ({
  type: TextEventType.interim,
  value,
  emotes: {},
  segmentId,
  revision,
});

function ui(): OverlayUiState {
  return {
    window: { open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null, closedExternally: false },
    vr: { state: "off", runtimeInstalled: false, error: null, reason: null },
  };
}

const commands = () => tauri.calls.map((c) => c.cmd.replace(/^plugin:[a-z-]+\|/, ""));

beforeEach(() => {
  vi.useFakeTimers();
  tauri.calls.length = 0;
  tauri.emitted.length = 0;
  tauri.respond = () => undefined;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("CaptionWindowDriver", () => {
  const windowSettings = (patch: Partial<CaptionWindowState> = {}): CaptionWindowState => ({ ...CaptionWindow_Schema.parse(undefined), ...patch });
  const openStatus = (options = { alwaysOnTop: true, clickThrough: true }) => ({ open: true, options, error: null });

  function setup() {
    const bus = new FakeBus();
    const state = ui();
    let externalCloses = 0;
    const driver = new CaptionWindowDriver(state, bus, () => externalCloses++);
    tauri.respond = (cmd, args) =>
      cmd.endsWith("caption_window_close") ? undefined : openStatus((args?.options as { alwaysOnTop: boolean; clickThrough: boolean }) ?? undefined);
    return { bus, state, driver, externalCloses: () => externalCloses };
  }

  it("does nothing while off (the default)", async () => {
    const { driver, bus } = setup();
    driver.sync(windowSettings());
    bus.publish(TextEventSource.stt, final("hello", "a"));
    await driver.idle();
    vi.advanceTimersByTime(FRAME_HEARTBEAT_MS * 3);
    expect(tauri.calls).toEqual([]);
    expect(tauri.emitted).toEqual([]);
  });

  it("opens when enabled, sends the current frame once open, and coalesces caption bursts", async () => {
    const { driver, bus, state } = setup();
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    expect(tauri.calls).toEqual([{ cmd: "plugin:caption-window|caption_window_open", args: { options: { alwaysOnTop: true, clickThrough: true } } }]);
    expect(state.window.open).toBe(true);
    // The frame is sent as soon as the window is open (no lines yet).
    expect(tauri.emitted.map((e) => [e.label, e.event, e.payload.lines.length])).toEqual([["captions", "caption_window:frame", 0]]);

    for (let i = 1; i <= 5; i++) bus.publish(TextEventSource.stt, interim("hello ".repeat(i), "s1", i));
    vi.advanceTimersByTime(FRAME_COALESCE_MS);
    expect(tauri.emitted).toHaveLength(2);
    expect(tauri.emitted[1].payload.lines.map((l) => l.text)).toEqual(["hello ".repeat(5)]);
  });

  it("re-sends the latest frame every second while open, with the same seq", async () => {
    const { driver, bus } = setup();
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    bus.publish(TextEventSource.stt, final("hello", "s1"));
    vi.advanceTimersByTime(FRAME_COALESCE_MS);
    const seq = tauri.emitted.at(-1)!.payload.seq;
    const before = tauri.emitted.length;
    vi.advanceTimersByTime(FRAME_HEARTBEAT_MS * 2);
    expect(tauri.emitted.length).toBe(before + 2);
    expect(tauri.emitted.slice(before).every((e) => e.payload.seq === seq)).toBe(true);
  });

  it("a quick off-on closes before it opens again (Rust serializes them too)", async () => {
    const { driver } = setup();
    driver.sync(windowSettings({ enable: true }));
    driver.sync(windowSettings({ enable: false }));
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    expect(commands()).toEqual(["caption_window_open", "caption_window_close", "caption_window_open"]);
  });

  it("pushes always-on-top / click-through changes only, and marks the frame unlocked", async () => {
    const { driver } = setup();
    driver.sync(windowSettings({ enable: true }));
    driver.sync(windowSettings({ enable: true, fontSize: "40" }));
    driver.sync(windowSettings({ enable: true, fontSize: "40", clickThrough: false }));
    await driver.idle();
    expect(tauri.calls.map((c) => [c.cmd.split("|")[1], c.args])).toEqual([
      ["caption_window_open", { options: { alwaysOnTop: true, clickThrough: true } }],
      ["caption_window_set_options", { options: { alwaysOnTop: true, clickThrough: false } }],
    ]);
    vi.advanceTimersByTime(FRAME_COALESCE_MS);
    expect(tauri.emitted.at(-1)!.payload.unlocked).toBe(true);
  });

  it("a window closed by Alt+F4 or the OS turns the setting off; the toggle's own close does not", async () => {
    const { driver, externalCloses } = setup();
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    driver.onStatus({ open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null, closedExternally: true });
    expect(externalCloses()).toBe(1);

    // The toggle closes it: Rust reports closedExternally false, and nothing is wanted any more anyway.
    driver.sync(windowSettings({ enable: true, fontSize: "30" }));
    driver.sync(windowSettings({ enable: false }));
    await driver.idle();
    driver.onStatus({ open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null, closedExternally: false });
    driver.onStatus({ open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null, closedExternally: true });
    expect(externalCloses()).toBe(1);
  });

  it("stops sending once closed", async () => {
    const { driver, bus } = setup();
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    driver.onStatus({ open: false, options: { alwaysOnTop: true, clickThrough: true }, error: null });
    const sent = tauri.emitted.length;
    bus.publish(TextEventSource.stt, final("late", "s9"));
    vi.advanceTimersByTime(FRAME_HEARTBEAT_MS * 3);
    expect(tauri.emitted.length).toBe(sent);
  });

  it("an open that fails is shown in the UI state", async () => {
    const { driver, state } = setup();
    tauri.respond = () => {
      throw new Error("no window for you");
    };
    driver.sync(windowSettings({ enable: true }));
    await driver.idle();
    expect(state.window.open).toBe(false);
    expect(state.window.error).toBe("no window for you");
  });
});

describe("VrOverlayDriver", () => {
  const vrSettings = (patch: Partial<VrOverlaySettings> = {}): VrOverlaySettings => ({ ...Service_VRC_Schema.parse(undefined).overlay, ...patch });

  function setup() {
    const bus = new FakeBus();
    const state = ui();
    const driver = new VrOverlayDriver(state, bus);
    tauri.respond = (cmd) =>
      cmd.endsWith("vr_overlay_stop") || cmd.endsWith("vr_overlay_set_text")
        ? undefined
        : { state: "waiting", runtimeInstalled: true, error: null, reason: "steamvrNotRunning" };
    return { bus, state, driver };
  }

  const textCalls = () => tauri.calls.filter((c) => c.cmd.endsWith("vr_overlay_set_text"));

  it("starts, pushes config changes only, and stops", async () => {
    const { driver, state } = setup();
    driver.sync(vrSettings());
    await driver.idle();
    expect(tauri.calls).toEqual([]);

    driver.sync(vrSettings({ enable: true }));
    await driver.idle();
    expect(commands()).toEqual(["vr_overlay_start"]);
    expect(tauri.calls[0].args).toMatchObject({ config: { attach: "head", widthM: 0.6, fontSizePx: 36 } });
    expect(state.vr).toMatchObject({ state: "waiting", reason: "steamvrNotRunning" });

    // Sources and the clear timer are TS-side only: no config push.
    driver.sync(vrSettings({ enable: true, clearAfterMs: "2000" }));
    driver.sync(vrSettings({ enable: true, clearAfterMs: "2000", widthM: "0.8" }));
    driver.sync(vrSettings({ enable: false, clearAfterMs: "2000", widthM: "0.8" }));
    await driver.idle();
    expect(commands()).toEqual(["vr_overlay_start", "vr_overlay_set_config", "vr_overlay_stop"]);
    expect(tauri.calls[1].args).toMatchObject({ config: { widthM: 0.8 } });
  });

  it("sends the caption at most every 100 ms, only when it changed, and the latest lines", async () => {
    const { driver, bus } = setup();
    driver.sync(vrSettings({ enable: true }));
    await driver.idle();
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS);
    tauri.calls.length = 0;

    for (let i = 1; i <= 8; i++) bus.publish(TextEventSource.stt, interim(`word${i}`, "s1", i));
    expect(textCalls()).toHaveLength(0);
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS);
    expect(textCalls()).toHaveLength(1);
    expect(textCalls()[0].args).toEqual({ caption: { lines: [{ text: "word8", kind: "interim", source: "text.stt", lang: null }] } });

    // The same lines again (an unchanged revision is not even a change): nothing new is sent.
    bus.publish(TextEventSource.stt, interim("word8", "s1", 8));
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS * 3);
    expect(textCalls()).toHaveLength(1);

    bus.publish(TextEventSource.stt, final("word8 done", "s1"));
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS);
    expect(textCalls()).toHaveLength(2);
    expect(textCalls()[1].args).toEqual({ caption: { lines: [{ text: "word8 done", kind: "final", source: "text.stt", lang: null }] } });
  });

  it("sends nothing while off", async () => {
    const { driver, bus } = setup();
    driver.sync(vrSettings());
    bus.publish(TextEventSource.stt, final("hello", "s1"));
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS * 5);
    await driver.idle();
    expect(tauri.calls).toEqual([]);
  });

  it("re-sends what is on screen after a restart", async () => {
    const { driver, bus } = setup();
    driver.sync(vrSettings({ enable: true, clearAfterMs: "0" }));
    await driver.idle();
    bus.publish(TextEventSource.stt, final("kept", "s1"));
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS);
    expect(textCalls()).toHaveLength(1);
    // Off and on again: the feed starts empty, so the new overlay gets the (empty) current caption, then new lines.
    driver.sync(vrSettings({ enable: false, clearAfterMs: "0" }));
    driver.sync(vrSettings({ enable: true, clearAfterMs: "0" }));
    await driver.idle();
    bus.publish(TextEventSource.stt, final("again", "s2"));
    vi.advanceTimersByTime(VR_TEXT_INTERVAL_MS);
    expect(textCalls().at(-1)!.args).toEqual({ caption: { lines: [{ text: "again", kind: "final", source: "text.stt", lang: null }] } });
  });

  it("a start that fails shows the error state", async () => {
    const { driver, state } = setup();
    tauri.respond = () => {
      throw new Error("cannot start");
    };
    driver.sync(vrSettings({ enable: true }));
    await driver.idle();
    expect(state.vr).toMatchObject({ state: "error", error: "cannot start" });
  });

  it("refresh asks for the status without starting anything", async () => {
    const { driver, state } = setup();
    tauri.respond = () => ({ state: "off", runtimeInstalled: true, error: null });
    driver.refresh();
    await driver.idle();
    expect(commands()).toEqual(["vr_overlay_status"]);
    expect(state.vr).toEqual({ state: "off", runtimeInstalled: true, error: null, reason: null });
  });
});
