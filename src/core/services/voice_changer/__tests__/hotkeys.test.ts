import { describe, expect, it } from "vitest";

import contract from "../../../../../src-tauri/src/services/voice_changer/params_contract.json";
import { profileDefaultEp, RVC_PROFILES } from "../contract";
import {
    assignHotkeyVoice,
    isVcShortcut,
    normalizeHotkeyVoices,
    VC_SHORTCUT_KEYS,
    VC_VOICE_HOTKEY_COUNT,
    vcHotkeysSupported,
    vcShortcutAction,
    voiceHotkeyKey,
} from "../hotkeys";
import { RVC_PROFILE_SETTINGS, rvcProfilePatch } from "../presets";

describe("voice-changer hotkeys", () => {
    it("names every shortcut and its action", () => {
        expect(vcShortcutAction("vcToggle")).toEqual({ kind: "toggle" });
        expect(vcShortcutAction("vcNextVoice")).toEqual({ kind: "step", step: 1 });
        expect(vcShortcutAction("vcPrevVoice")).toEqual({ kind: "step", step: -1 });
        for (let i = 0; i < VC_VOICE_HOTKEY_COUNT; i++) {
            const key = voiceHotkeyKey(i);
            expect(VC_SHORTCUT_KEYS).toContain(key);
            expect(vcShortcutAction(key)).toEqual({ kind: "voice", index: i });
        }
        expect(isVcShortcut("vcVoice2")).toBe(true);
        expect(isVcShortcut("muteMic")).toBe(false);
        expect(isVcShortcut(undefined)).toBe(false);
    });

    it("is supported only once the app's shortcut store has every key", () => {
        expect(vcHotkeysSupported({ bgInput: "", start: "" })).toBe(false);
        expect(vcHotkeysSupported(Object.fromEntries(VC_SHORTCUT_KEYS.map((k) => [k, ""])))).toBe(true);
        expect(vcHotkeysSupported(null)).toBe(false);
    });

    it("keeps a fixed-length list of voice ids, one hotkey per voice", () => {
        expect(normalizeHotkeyVoices(undefined)).toEqual(["", "", "", ""]);
        expect(normalizeHotkeyVoices([" a ", 3, "b"])).toEqual(["a", "", "b", ""]);
        let v = assignHotkeyVoice([], 0, "alto");
        expect(v).toEqual(["alto", "", "", ""]);
        v = assignHotkeyVoice(v, 2, "alto");
        expect(v).toEqual(["", "", "alto", ""]);
        v = assignHotkeyVoice(v, 1, "bass");
        v = assignHotkeyVoice(v, 2, "");
        expect(v).toEqual(["", "bass", "", ""]);
        expect(assignHotkeyVoice(v, 9, "x")).toEqual(["", "bass", "", ""]);
    });
});

describe("RVC presets", () => {
    it("every preset sets the EP the contract names as its default", () => {
        const defaultEp = (contract.enums.rvc_preset as { default_ep: Record<string, string> }).default_ep;
        for (const profile of RVC_PROFILES) {
            expect(RVC_PROFILE_SETTINGS[profile].rvcExecutionProvider, profile).toBe(defaultEp[profile]);
            expect(RVC_PROFILE_SETTINGS[profile].rvcExecutionProvider, profile).toBe(profileDefaultEp(profile));
            expect(rvcProfilePatch(profile).rvcProfile).toBe(profile);
        }
    });

    it("Streamer stays on the CPU with a CPU pitch extractor; Studio-TRT keeps Quality's shapes", () => {
        expect(RVC_PROFILE_SETTINGS.streamer).toMatchObject({ rvcExecutionProvider: "cpu", pitchExtractor: "fcpe", indexRate: 0 });
        const { rvcExecutionProvider: _q, ...quality } = RVC_PROFILE_SETTINGS.quality;
        const { rvcExecutionProvider: trtEp, ...studio } = RVC_PROFILE_SETTINGS.studio_trt;
        expect(studio).toEqual(quality);
        expect(trtEp).toBe("tensorrt_rtx");
    });

    it("Quality and Studio-TRT decode 20 ms of lookahead; the latency presets none", () => {
        expect(RVC_PROFILE_SETTINGS.quality.lookaheadMs).toBe(20);
        expect(RVC_PROFILE_SETTINGS.studio_trt.lookaheadMs).toBe(20);
        expect(RVC_PROFILE_SETTINGS.low_latency.lookaheadMs).toBe(0);
        expect(RVC_PROFILE_SETTINGS.streamer.lookaheadMs).toBe(0);
    });
});
