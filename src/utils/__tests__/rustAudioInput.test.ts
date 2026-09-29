import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

import {
  buildRustSourceOptions,
  isLoopbackSourceId,
  isRustSourceAvailable,
  LOOPBACK_DEFAULT_SOURCE,
  loopbackOutputName,
  loopbackSourceId,
  resolveRustInputDeviceForInvoke,
  type RustDeviceView,
} from "../rustAudioInput";

const labels = {
  systemDefault: "System default",
  loopbackDefault: "Default output (loopback)",
  loopback: (name: string) => `${name} (loopback)`,
  unavailable: (name: string) => `${name} (not connected)`,
};

const devices = {
  inputs: [{ id: "Mic", name: "Mic" }, { id: "USB Mic", name: "USB Mic" }],
  outputs: [{ id: "Speakers", name: "Speakers" }],
};

describe("loopback source ids", () => {
  it("round-trips output names and recognizes the default", () => {
    expect(loopbackSourceId(" Speakers (Realtek) ")).toBe("loopback:Speakers (Realtek)");
    expect(isLoopbackSourceId("loopback:Speakers")).toBe(true);
    expect(isLoopbackSourceId("LOOPBACK:default")).toBe(true);
    expect(isLoopbackSourceId("Mic")).toBe(false);
    expect(isLoopbackSourceId(null)).toBe(false);
    expect(loopbackOutputName("loopback:Speakers")).toBe("Speakers");
    expect(loopbackOutputName(LOOPBACK_DEFAULT_SOURCE)).toBe("");
    expect(loopbackOutputName("loopback:")).toBe("");
    expect(loopbackOutputName("Mic")).toBeNull();
  });
});

describe("isRustSourceAvailable", () => {
  const view: RustDeviceView = { inputs: ["Mic"], outputs: ["Speakers"], defaultInput: "Mic", defaultOutput: "Speakers" };
  it("checks the device a source needs", () => {
    expect(isRustSourceAvailable(null, view)).toBe(true);
    expect(isRustSourceAvailable("default", view)).toBe(true);
    expect(isRustSourceAvailable("Mic", view)).toBe(true);
    expect(isRustSourceAvailable("USB Mic", view)).toBe(false);
    expect(isRustSourceAvailable("loopback:Speakers", view)).toBe(true);
    expect(isRustSourceAvailable("loopback:Headset", view)).toBe(false);
    expect(isRustSourceAvailable("loopback:default", view)).toBe(true);
    const empty: RustDeviceView = { inputs: [], outputs: [], defaultInput: null, defaultOutput: null };
    expect(isRustSourceAvailable(null, empty)).toBe(false);
    expect(isRustSourceAvailable("loopback:default", empty)).toBe(false);
  });
});

describe("buildRustSourceOptions", () => {
  it("lists the default and the microphones, loopback only on request", () => {
    const plain = buildRustSourceOptions(devices, "", false, labels);
    expect(plain.inputs.map((o) => o.value)).toEqual(["", "Mic", "USB Mic"]);
    expect(plain.loopback).toEqual([]);
    const withLoopback = buildRustSourceOptions(devices, "", true, labels);
    expect(withLoopback.loopback).toEqual([
      { label: "Default output (loopback)", value: "loopback:default" },
      { label: "Speakers (loopback)", value: "loopback:Speakers" },
    ]);
  });

  it("keeps a saved device that is not connected, marked", () => {
    const { inputs } = buildRustSourceOptions(devices, "Old Headset", false, labels);
    expect(inputs.at(-1)).toEqual({ label: "Old Headset (not connected)", value: "Old Headset" });
    const lb = buildRustSourceOptions(devices, "loopback:Headphones", true, labels);
    expect(lb.loopback.at(-1)).toEqual({ label: "Headphones (loopback) (not connected)", value: "loopback:Headphones" });
    // A loopback value in a picker without loopback stays visible (and is not called disconnected).
    const hidden = buildRustSourceOptions(devices, "loopback:Speakers", false, labels);
    expect(hidden.inputs.at(-1)).toEqual({ label: "Speakers (loopback)", value: "loopback:Speakers" });
    // Present devices and default sentinels add nothing.
    expect(buildRustSourceOptions(devices, "USB Mic", false, labels).inputs).toHaveLength(3);
    expect(buildRustSourceOptions(devices, "default", false, labels).inputs).toHaveLength(3);
  });
});

describe("resolveRustInputDeviceForInvoke", () => {
  afterEach(() => {
    delete (globalThis as { window?: unknown }).window;
  });

  function withGlobal(native: string) {
    (globalThis as { window?: unknown }).window = { ApiServer: { state: { audioInputDeviceNative: native } } };
  }

  it("uses the per-service source, then the app-wide microphone", () => {
    withGlobal("USB Mic");
    expect(resolveRustInputDeviceForInvoke("Mic")).toBe("Mic");
    expect(resolveRustInputDeviceForInvoke("loopback:Speakers")).toBe("loopback:Speakers");
    expect(resolveRustInputDeviceForInvoke("default")).toBeNull();
    expect(resolveRustInputDeviceForInvoke("")).toBe("USB Mic");
  });

  it("never uses a loopback source as the app-wide microphone", () => {
    withGlobal("loopback:Speakers");
    expect(resolveRustInputDeviceForInvoke(undefined)).toBeNull();
  });
});
