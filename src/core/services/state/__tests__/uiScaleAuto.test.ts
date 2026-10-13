import { beforeEach, describe, expect, it, vi } from "vitest";
import { BackendSchema } from "../../../schema";
import { applyUiScaleAutoDecision } from "../uiScaleAuto";

/** In-memory app-data folder behind `@tauri-apps/plugin-fs`, for the Service_State wiring tests below. */
const disk = vi.hoisted(() => ({ files: new Map<string, string>() }));
vi.mock("@tauri-apps/plugin-fs", () => ({
  exists: async (path: string) => disk.files.has(path),
  readFile: async (path: string) => {
    const text = disk.files.get(path);
    if (text === undefined) throw new Error(`ENOENT ${path}`);
    return new TextEncoder().encode(text);
  },
  writeFile: async (path: string, data: Uint8Array) => void disk.files.set(path, new TextDecoder().decode(data)),
  copyFile: async (from: string, to: string) => void disk.files.set(to, disk.files.get(from) ?? ""),
  rename: async (from: string, to: string) => {
    disk.files.set(to, disk.files.get(from) ?? "");
    disk.files.delete(from);
  },
  mkdir: async () => { },
}));

import Service_State from "../index";

const SETTINGS = "user/settings";

describe("Interface size Auto: the settings", () => {
  it("is on by default, and undecided", () => {
    const state = BackendSchema.parse({});
    expect(state.uiScaleAuto).toBe(true);
    expect(state.uiScaleAutoDecided).toBe(false);
    expect(state.uiScale).toBe(1); // the manual size stays M
  });

  it("falls back to the defaults for values that are not booleans", () => {
    const state = BackendSchema.parse({ uiScaleAuto: "yes", uiScaleAutoDecided: 7 });
    expect(state.uiScaleAuto).toBe(true);
    expect(state.uiScaleAutoDecided).toBe(false);
  });
});

describe("Interface size Auto: the one-time decision", () => {
  it("moves a new install to Auto", () => {
    const state = BackendSchema.parse({});
    expect(applyUiScaleAutoDecision(state)).toBe(true);
    expect(state.uiScaleAuto).toBe(true);
    expect(state.uiScaleAutoDecided).toBe(true);
  });

  it("moves a profile at the default M (1) to Auto", () => {
    const state = BackendSchema.parse({ uiScale: 1, uiLanguage: "de" });
    expect(applyUiScaleAutoDecision(state)).toBe(true);
    expect(state.uiScaleAuto).toBe(true);
    expect(state.uiScale).toBe(1);
    expect(state.uiLanguage).toBe("de"); // nothing else is touched
  });

  it("moves a profile written before the setting existed (no uiScale key) to Auto", () => {
    const state = BackendSchema.parse({ id: "me" });
    applyUiScaleAutoDecision(state);
    expect(state.uiScaleAuto).toBe(true);
  });

  it("keeps the manual size of a profile that picked S, L or XL (Auto off)", () => {
    for (const picked of [0.8, 1.2, 1.4]) {
      const state = BackendSchema.parse({ uiScale: picked });
      expect(state.uiScaleAuto, "the schema default is Auto until the decision").toBe(true);
      expect(applyUiScaleAutoDecision(state), String(picked)).toBe(true);
      expect(state.uiScaleAuto, String(picked)).toBe(false);
      expect(state.uiScale, String(picked)).toBe(picked);
      expect(state.uiScaleAutoDecided, String(picked)).toBe(true);
    }
  });

  it("reads 1.0000001 as M", () => {
    const state = BackendSchema.parse({ uiScale: 1.0000001 });
    applyUiScaleAutoDecision(state);
    expect(state.uiScaleAuto).toBe(true);
  });

  it("is taken once: a decided profile is left exactly as it is", () => {
    const manual = BackendSchema.parse({ uiScale: 1.2, uiScaleAutoDecided: true, uiScaleAuto: true });
    expect(applyUiScaleAutoDecision(manual)).toBe(false);
    expect(manual.uiScaleAuto).toBe(true); // picked Auto after the decision: kept even with uiScale 1.2 stored

    const auto = BackendSchema.parse({ uiScale: 1, uiScaleAutoDecided: true, uiScaleAuto: false });
    expect(applyUiScaleAutoDecision(auto)).toBe(false);
    expect(auto.uiScaleAuto).toBe(false); // picked M after the decision: not moved back to Auto
  });

  it("is not repeated on a second launch", () => {
    const state = BackendSchema.parse({ uiScale: 1.2 });
    applyUiScaleAutoDecision(state);
    expect(state.uiScaleAuto).toBe(false);
    const saved = BackendSchema.parse(JSON.parse(JSON.stringify(state)));
    expect(applyUiScaleAutoDecision(saved)).toBe(false);
    expect(saved.uiScaleAuto).toBe(false);
    expect(saved.uiScale).toBe(1.2);
  });
});

describe("Interface size Auto: Service_State.init takes the decision when the settings load", () => {
  beforeEach(() => {
    disk.files.clear();
  });

  async function launch() {
    const service = new Service_State();
    await service.init();
    return service.state;
  }

  it("a new install (no settings file) starts on Auto, and the file is written with the decision", async () => {
    const state = await launch();
    expect(state.uiScaleAuto).toBe(true);
    expect(state.uiScaleAutoDecided).toBe(true);
    await vi.waitFor(() => expect(JSON.parse(disk.files.get(SETTINGS) ?? "{}").uiScaleAutoDecided).toBe(true));
  });

  it("an existing profile on L keeps L, and the decision is saved even though nothing else changed", async () => {
    // firstRunDecided is already true: only the Interface size decision is new, and it alone must trigger the write.
    disk.files.set(SETTINGS, JSON.stringify({ id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1.2 }));
    const state = await launch();
    expect(state.uiScaleAuto).toBe(false);
    expect(state.uiScale).toBe(1.2);
    await vi.waitFor(() => {
      const saved = JSON.parse(disk.files.get(SETTINGS) ?? "{}");
      expect(saved.uiScaleAutoDecided).toBe(true);
      expect(saved.uiScaleAuto).toBe(false);
      expect(saved.uiScale).toBe(1.2);
    });
  });

  it("an existing profile on the default M moves to Auto", async () => {
    disk.files.set(SETTINGS, JSON.stringify({ id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1 }));
    const state = await launch();
    expect(state.uiScaleAuto).toBe(true);
    await vi.waitFor(() => expect(JSON.parse(disk.files.get(SETTINGS) ?? "{}").uiScaleAutoDecided).toBe(true));
  });

  it("does not decide again on the next launch", async () => {
    disk.files.set(SETTINGS, JSON.stringify({ id: "me", firstRunDecided: true, onboardingComplete: true, uiScale: 1.4 }));
    const first = await launch();
    expect(first.uiScaleAuto).toBe(false);
    await vi.waitFor(() => expect(JSON.parse(disk.files.get(SETTINGS) ?? "{}").uiScaleAutoDecided).toBe(true));
    // The user then picks Auto in Settings and quits; the next launch must keep it although uiScale is still 1.4.
    const saved = JSON.parse(disk.files.get(SETTINGS) ?? "{}");
    disk.files.set(SETTINGS, JSON.stringify({ ...saved, uiScaleAuto: true }));
    const second = await launch();
    expect(second.uiScaleAuto).toBe(true);
    expect(second.uiScale).toBe(1.4);
  });
});
