import { describe, expect, it } from "vitest";
import { planTtsDesktopDefault } from "../defaults";
import { Service_TTS_Schema, TTS_Backends } from "../schema";

const base = { isDesktop: true, alreadyMigrated: false, freshInstall: false, backend: TTS_Backends.native, localOfferShown: false };

describe("TTS desktop default (P2: Kokoro for new installs, a one-time offer for existing profiles)", () => {
  it("does nothing in the web build or once done", () => {
    expect(planTtsDesktopDefault({ ...base, isDesktop: false, freshInstall: true })).toEqual({ action: "none" });
    expect(planTtsDesktopDefault({ ...base, alreadyMigrated: true })).toEqual({ action: "none" });
  });

  it("starts a new install (schema defaults) on the local Kokoro voice", () => {
    const defaults = Service_TTS_Schema.parse({});
    expect(defaults.backend).toBe(TTS_Backends.native);
    expect(defaults.desktopDefaultMigrated).toBe(false);
    expect(defaults.localOfferShown).toBe(false);
    expect(planTtsDesktopDefault({ ...base, freshInstall: true, backend: defaults.backend })).toEqual({ action: "local" });
  });

  it("offers Kokoro once to existing profiles on Web Speech or SAPI", () => {
    expect(planTtsDesktopDefault(base)).toEqual({ action: "offer" });
    expect(planTtsDesktopDefault({ ...base, backend: TTS_Backends.windows })).toEqual({ action: "offer" });
    expect(planTtsDesktopDefault({ ...base, localOfferShown: true })).toEqual({ action: "mark" });
  });

  it("leaves deliberate choices alone", () => {
    for (const backend of [TTS_Backends.voicevox, TTS_Backends.azure, TTS_Backends.localNeural, TTS_Backends.openaiSpeech])
      expect(planTtsDesktopDefault({ ...base, backend })).toEqual({ action: "mark" });
    expect(planTtsDesktopDefault({ ...base, freshInstall: true, backend: TTS_Backends.voicevox })).toEqual({ action: "mark" });
  });

  it("keeps saved profiles valid (new keys default to false)", () => {
    const saved = Service_TTS_Schema.parse({ backend: "windows", windows: {} });
    expect(saved.backend).toBe(TTS_Backends.windows);
    expect(saved.desktopDefaultMigrated).toBe(false);
  });
});
