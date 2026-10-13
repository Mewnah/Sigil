import { beforeAll, describe, expect, it } from "vitest";
import { noSignalMessage } from "../statusMessages";
import { initSttTestI18n, translated } from "./sttTestI18n";

beforeAll(async () => {
  await initSttTestI18n();
});

describe("noSignalMessage (the silent-microphone line of the STT panel, the footer card and the system log)", () => {
  it("names the microphone and says where to look", () => {
    const text = noSignalMessage("Headset Mic (USB)");
    expect(translated(text)).toBe(true);
    expect(text).toContain("Sigil hears nothing from Headset Mic (USB).");
    expect(text).toContain("Settings > Privacy & security > Microphone > Let desktop apps access your microphone");
    expect(text).toContain("mute switch on your headset");
  });

  it("does not make up a name for a microphone it could not name", () => {
    for (const device of ["", "   "]) {
      const text = noSignalMessage(device);
      expect(translated(text)).toBe(true);
      expect(text).toContain("Sigil hears nothing from your microphone.");
      expect(text).not.toMatch(/\{\{|undefined|null/);
    }
  });

  it("is plain text a status line can show: no markup from the device name is added", () => {
    // A device name with punctuation is shown as it is (the interpolation does not escape).
    expect(noSignalMessage("Mic & <Co> (R)")).toContain("from Mic & <Co> (R).");
  });

  it("follows the caller's translation function (a component passes its own)", () => {
    const seen: [string, Record<string, unknown> | undefined][] = [];
    const text = noSignalMessage("Headset Mic", (key, options) => {
      seen.push([key, options]);
      return `${key}|${JSON.stringify(options)}`;
    });
    expect(text).toBe('stt.status_no_signal|{"device":"Headset Mic"}');
    expect(noSignalMessage("", (key) => key)).toBe("stt.status_no_signal_unnamed");
    expect(seen).toEqual([["stt.status_no_signal", { device: "Headset Mic" }]]);
  });
});
