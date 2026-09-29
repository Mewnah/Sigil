import { describe, expect, it } from "vitest";
import { estimatePhysicalCores, pickDesktopWhisperModel, planDesktopDefaultMigration } from "../migrate";
import { Service_STT_Schema, STT_Backends } from "../schema";
import { findWhisperModel } from "../whisperModels";

const base = { isDesktop: true, alreadyMigrated: false, backend: STT_Backends.native, logicalCores: 12 };

describe("desktop default migration (native → Whisper)", () => {
  it("does nothing in the web build or once done", () => {
    expect(planDesktopDefaultMigration({ ...base, isDesktop: false })).toEqual({ action: "none" });
    expect(planDesktopDefaultMigration({ ...base, alreadyMigrated: true })).toEqual({ action: "none" });
  });

  it("only records the migration for users on a working backend", () => {
    for (const backend of [STT_Backends.whisper, STT_Backends.azure, STT_Backends.chrome, STT_Backends.openai_audio])
      expect(planDesktopDefaultMigration({ ...base, backend })).toEqual({ action: "mark" });
  });

  it("moves a fresh desktop profile (schema defaults) to Whisper small on 6+ cores", () => {
    const defaults = Service_STT_Schema.parse({});
    expect(defaults.backend).toBe(STT_Backends.native);
    expect(defaults.desktopDefaultMigrated).toBe(false);
    const plan = planDesktopDefaultMigration({
      ...base,
      backend: defaults.backend,
      nativeLanguage: defaults.native.language,
      whisperModel: defaults.whisper.model,
      whisperLanguage: defaults.whisper.language,
    });
    expect(plan).toEqual({ action: "migrate", backend: STT_Backends.whisper, whisperModel: "small.en-q5_1", whisperLanguage: "en" });
  });

  it("maps the Web Speech language and picks a multilingual model for it", () => {
    expect(planDesktopDefaultMigration({ ...base, nativeLanguage: "ja-JP", logicalCores: 4 })).toEqual({
      action: "migrate",
      backend: STT_Backends.whisper,
      whisperModel: "base-q5_1",
      whisperLanguage: "ja",
    });
    expect(planDesktopDefaultMigration({ ...base, nativeLanguage: "yue-Hant-HK" })).toMatchObject({ whisperLanguage: "zh" });
    expect(planDesktopDefaultMigration({ ...base, nativeLanguage: "sw-KE" })).toMatchObject({
      whisperModel: "small-q5_1",
      whisperLanguage: "auto",
    });
  });

  it("treats the legacy 'browser' backend like native", () => {
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.browser }).action).toBe("migrate");
  });

  it("keeps Whisper settings the user already chose", () => {
    expect(planDesktopDefaultMigration({ ...base, whisperModel: "medium-q5_0", whisperLanguage: "de" })).toEqual({
      action: "migrate",
      backend: STT_Backends.whisper,
    });
  });

  it("estimates physical cores conservatively", () => {
    expect(estimatePhysicalCores(12)).toBe(6);
    expect(estimatePhysicalCores(16)).toBe(8);
    expect(estimatePhysicalCores(8)).toBe(4);
    expect(estimatePhysicalCores(6)).toBe(6);
    expect(estimatePhysicalCores(4)).toBe(4);
    expect(estimatePhysicalCores(undefined)).toBe(4);
    expect(estimatePhysicalCores(0)).toBe(4);
  });

  it("only picks models the catalog offers", () => {
    for (const cores of [2, 4, 6, 8])
      for (const lang of ["en", "ja", "auto"]) expect(findWhisperModel(pickDesktopWhisperModel(cores, lang))).toBeDefined();
  });
});
