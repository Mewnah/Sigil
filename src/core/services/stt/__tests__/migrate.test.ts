import { describe, expect, it } from "vitest";
import {
  estimatePhysicalCores,
  localOfferKindOf,
  pickDesktopWhisperModel,
  planDesktopDefaultMigration,
  planWhisperMigration,
} from "../migrate";
import { Service_STT_Schema, STT_Backends } from "../schema";
import { findWhisperModel } from "../whisperModels";

const base = { isDesktop: true, alreadyMigrated: false, backend: STT_Backends.native, logicalCores: 12 };

describe("desktop default migration (P2: native → local for new installs, an offer for existing profiles)", () => {
  it("starts a new install on the local backend", () => {
    expect(planDesktopDefaultMigration({ ...base, freshInstall: true })).toEqual({ action: "local", backend: STT_Backends.local });
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.browser, freshInstall: true }).action).toBe("local");
    // A new install that already picked a working backend (onboarding) is only recorded.
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.vosk, freshInstall: true })).toEqual({ action: "mark" });
  });

  it("offers local to an existing profile once, with Whisper as the answer to no", () => {
    expect(planDesktopDefaultMigration({ ...base, nativeLanguage: "ja-JP", logicalCores: 4 })).toEqual({
      action: "offer",
      kind: "native",
      fallback: { action: "migrate", backend: STT_Backends.whisper, whisperModel: "base-q5_1", whisperLanguage: "ja" },
    });
    // Offer already shown (for example the app closed with the dialog open): straight to Whisper.
    expect(planDesktopDefaultMigration({ ...base, localOfferShown: true }).action).toBe("migrate");
    expect(planDesktopDefaultMigration({ ...base, isDesktop: false, freshInstall: true })).toEqual({ action: "none" });
  });
});

describe("the one-time offer of the built-in engine, per engine the profile is on", () => {
  it("offers it to an existing profile on native or the legacy browser backend, with Whisper as the answer to no", () => {
    for (const backend of [STT_Backends.native, STT_Backends.browser]) {
      const plan = planDesktopDefaultMigration({ ...base, backend });
      expect(plan).toMatchObject({ action: "offer", kind: "native" });
      expect(plan).toHaveProperty("fallback.backend", STT_Backends.whisper);
    }
  });

  it("offers it to an existing profile on Chrome or Edge, and declining changes nothing", () => {
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.chrome })).toEqual({ action: "offer", kind: "chrome", fallback: null });
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.edge })).toEqual({ action: "offer", kind: "edge", fallback: null });
  });

  it("still offers it when the earlier migration already ran for the profile (it only recorded a working engine)", () => {
    for (const backend of [STT_Backends.chrome, STT_Backends.edge])
      expect(planDesktopDefaultMigration({ ...base, backend, alreadyMigrated: true })).toMatchObject({ action: "offer", fallback: null });
  });

  it("never asks twice: an offer already answered (also by Keep) is not repeated", () => {
    // native: straight to Whisper, as before
    expect(planDesktopDefaultMigration({ ...base, localOfferShown: true }).action).toBe("migrate");
    // Chrome and Edge: nothing to do, and only the first call records the migration
    for (const backend of [STT_Backends.chrome, STT_Backends.edge]) {
      expect(planDesktopDefaultMigration({ ...base, backend, localOfferShown: true })).toEqual({ action: "mark" });
      expect(planDesktopDefaultMigration({ ...base, backend, localOfferShown: true, alreadyMigrated: true })).toEqual({ action: "none" });
    }
  });

  it("does not ask a new install, which has just chosen its engine, nor the web build", () => {
    for (const backend of [STT_Backends.chrome, STT_Backends.edge]) {
      expect(planDesktopDefaultMigration({ ...base, backend, freshInstall: true })).toEqual({ action: "mark" });
      expect(planDesktopDefaultMigration({ ...base, backend, isDesktop: false })).toEqual({ action: "none" });
    }
  });

  it("leaves every other engine out of the offer", () => {
    for (const backend of [STT_Backends.whisper, STT_Backends.vosk, STT_Backends.azure, STT_Backends.deepgram, STT_Backends.openai_audio, STT_Backends.local, STT_Backends.geminiLive])
      expect(planDesktopDefaultMigration({ ...base, backend }).action).not.toBe("offer");
  });

  it("names the prompt kind of a backend", () => {
    expect(localOfferKindOf(STT_Backends.chrome)).toBe("chrome");
    expect(localOfferKindOf(STT_Backends.edge)).toBe("edge");
    expect(localOfferKindOf(STT_Backends.native)).toBe("native");
    expect(localOfferKindOf(STT_Backends.browser)).toBe("native");
    expect(localOfferKindOf(STT_Backends.whisper)).toBeNull();
    expect(localOfferKindOf(STT_Backends.local)).toBeNull();
  });
});

describe("desktop default migration (native → Whisper)", () => {
  it("does nothing in the web build or once done", () => {
    expect(planDesktopDefaultMigration({ ...base, isDesktop: false })).toEqual({ action: "none" });
    expect(planDesktopDefaultMigration({ ...base, alreadyMigrated: true })).toEqual({ action: "none" });
  });

  it("only records the migration for users on a working backend", () => {
    for (const backend of [STT_Backends.whisper, STT_Backends.azure, STT_Backends.openai_audio, STT_Backends.vosk])
      expect(planDesktopDefaultMigration({ ...base, backend })).toEqual({ action: "mark" });
  });

  it("moves a desktop profile on schema defaults to Whisper small on 6+ cores", () => {
    const defaults = Service_STT_Schema.parse({});
    expect(defaults.backend).toBe(STT_Backends.native);
    expect(defaults.desktopDefaultMigrated).toBe(false);
    const plan = planWhisperMigration({
      ...base,
      backend: defaults.backend,
      nativeLanguage: defaults.native.language,
      whisperModel: defaults.whisper.model,
      whisperLanguage: defaults.whisper.language,
    });
    expect(plan).toEqual({ action: "migrate", backend: STT_Backends.whisper, whisperModel: "small.en-q5_1", whisperLanguage: "en" });
  });

  it("maps the Web Speech language and picks a multilingual model for it", () => {
    expect(planWhisperMigration({ ...base, nativeLanguage: "ja-JP", logicalCores: 4 })).toEqual({
      action: "migrate",
      backend: STT_Backends.whisper,
      whisperModel: "base-q5_1",
      whisperLanguage: "ja",
    });
    expect(planWhisperMigration({ ...base, nativeLanguage: "yue-Hant-HK" })).toMatchObject({ whisperLanguage: "zh" });
    expect(planWhisperMigration({ ...base, nativeLanguage: "sw-KE" })).toMatchObject({
      whisperModel: "small-q5_1",
      whisperLanguage: "auto",
    });
  });

  it("treats the legacy 'browser' backend like native", () => {
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.browser }).action).toBe("offer");
    expect(planDesktopDefaultMigration({ ...base, backend: STT_Backends.browser, localOfferShown: true }).action).toBe("migrate");
  });

  it("keeps Whisper settings the user already chose", () => {
    expect(planWhisperMigration({ ...base, whisperModel: "medium-q5_0", whisperLanguage: "de" })).toEqual({
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
