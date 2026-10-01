import { describe, expect, it } from "vitest";

import { BLOCKING_ISSUES, CLIP_ISSUES, type VoiceMeta } from "../contract";
import {
    METER_FLOOR_DB,
    canSaveTake,
    findVoice,
    formatClock,
    formatSeconds,
    isBlockingIssue,
    isMissingVoice,
    levelHint,
    meterFraction,
    recordDeviceArg,
    sortedIssues,
    speechProgress,
    voiceLabel,
} from "../format";

const quality = (issues: string[], ok = true) => ({
    durationMs: 5000,
    trimmedMs: 1000,
    snrDb: 30,
    peakDbfs: -6,
    clippedRatio: 0,
    lufsBefore: -30,
    lufsAfter: -20,
    ok,
    issues: issues as never,
});

const voice = (id: string, name: string): VoiceMeta => ({
    id,
    name,
    createdUnixMs: 0,
    durationMs: 6400,
    sampleRate: 48000,
    lufs: -20,
    snrDb: 40,
    peakDbfs: -3,
    clipped: false,
    transcript: null,
    language: null,
    source: "recorded",
    sha256: "",
    cachedBackends: [],
});

describe("voice library format helpers", () => {
    it("only no_speech and too_short block a take, and they sort first", () => {
        expect([...BLOCKING_ISSUES].sort()).toEqual(["no_speech", "too_short"]);
        for (const i of CLIP_ISSUES) expect(isBlockingIssue(i)).toBe(i === "no_speech" || i === "too_short");
        const sorted = sortedIssues(quality(["noisy", "too_short", "clipped", "future_issue", "noisy"]));
        expect(sorted).toEqual([
            { issue: "too_short", severity: "error" },
            { issue: "noisy", severity: "warning" },
            { issue: "clipped", severity: "warning" },
        ]);
        expect(sortedIssues(null)).toEqual([]);
    });

    it("maps levels to the meter and to advice", () => {
        expect(meterFraction(0)).toBe(1);
        expect(meterFraction(METER_FLOOR_DB)).toBe(0);
        expect(meterFraction(-120)).toBe(0);
        expect(meterFraction(6)).toBe(1);
        expect(meterFraction(-30)).toBeCloseTo(0.5);
        expect(meterFraction(Number.NaN)).toBe(0);
        expect(meterFraction(undefined)).toBe(0);
        expect(levelHint(null)).toBe("silent");
        expect(levelHint({ levelDb: -100, peakDb: -100, clipping: false })).toBe("silent");
        expect(levelHint({ levelDb: -50, peakDb: -40, clipping: false })).toBe("quiet");
        expect(levelHint({ levelDb: -25, peakDb: -10, clipping: false })).toBe("good");
        expect(levelHint({ levelDb: -25, peakDb: -0.5, clipping: false })).toBe("hot");
        expect(levelHint({ levelDb: -40, peakDb: -20, clipping: true })).toBe("hot");
    });

    it("tracks speech against the 3-10 s a reference needs", () => {
        expect(speechProgress(0)).toEqual({ fraction: 0, state: "more" });
        expect(speechProgress(2999).state).toBe("more");
        expect(speechProgress(3000)).toEqual({ fraction: 0.3, state: "enough" });
        expect(speechProgress(10_000)).toEqual({ fraction: 1, state: "full" });
        expect(speechProgress(25_000).fraction).toBe(1);
        expect(speechProgress(-5).fraction).toBe(0);
        expect(speechProgress(undefined).state).toBe("more");
    });

    it("formats clocks and seconds", () => {
        expect(formatClock(0)).toBe("0:00");
        expect(formatClock(4_900)).toBe("0:04");
        expect(formatClock(72_000)).toBe("1:12");
        expect(formatClock(Number.NaN)).toBe("0:00");
        expect(formatSeconds(6_449)).toBe("6.4");
        expect(formatSeconds(10_000)).toBe("10.0");
        expect(formatSeconds(null)).toBe("0.0");
    });

    it("saves only usable takes with a name", () => {
        expect(canSaveTake({ quality: quality([]) }, "Alice")).toBe(true);
        expect(canSaveTake({ quality: quality([]) }, "   ")).toBe(false);
        expect(canSaveTake({ quality: quality(["too_short"], false) }, "Alice")).toBe(false);
        expect(canSaveTake(null, "Alice")).toBe(false);
    });

    it("labels, finds and flags voices for the pickers", () => {
        const voices = [voice("alice-1a2b3c", "Alice"), voice("voice-9f8e7d", "  ")];
        expect(voiceLabel(voices[0])).toBe("Alice");
        expect(voiceLabel(voices[1])).toBe("voice-9f8e7d");
        expect(findVoice(voices, "alice-1a2b3c")?.name).toBe("Alice");
        expect(findVoice(voices, "")).toBeUndefined();
        expect(isMissingVoice(voices, "gone-000000", true)).toBe(true);
        expect(isMissingVoice(voices, "gone-000000", false)).toBe(false);
        expect(isMissingVoice(voices, "", true)).toBe(false);
        expect(isMissingVoice(voices, "alice-1a2b3c", true)).toBe(false);
    });

    it("spells out the system default device (\"\" means the voice changer's microphone to Rust)", () => {
        expect(recordDeviceArg("")).toBe("default");
        expect(recordDeviceArg("  ")).toBe("default");
        expect(recordDeviceArg(undefined)).toBe("default");
        expect(recordDeviceArg(" USB Mic ")).toBe("USB Mic");
    });
});
