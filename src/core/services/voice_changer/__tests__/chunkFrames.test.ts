import { describe, expect, it } from "vitest";

import { crossfadeEngineBlockMs, crossfadeMaxMs, crossfadeMaxMsForEngine, deviceBlockNote, engineBlockMs } from "../chunkFrames";
import { runsOnTemplateEngine, type VoiceKind } from "../families";

describe("engine block and crossfade cap (P3B-H)", () => {
    it("rounds the setting up to the plan's whole 20 ms and keeps 0 as derived", () => {
        expect(engineBlockMs(0)).toBe(0);
        expect(engineBlockMs(Number.NaN)).toBe(0);
        expect(engineBlockMs(10)).toBe(20);
        expect(engineBlockMs(20)).toBe(20);
        expect(engineBlockMs(30)).toBe(40);
        expect(engineBlockMs(40)).toBe(40);
        expect(engineBlockMs(50)).toBe(60);
        expect(engineBlockMs(500)).toBe(200);
    });

    it("a template voice's crossfade is capped by its engine block, not by a 10 ms device block", () => {
        // 480 frames at 48 kHz: one 10 ms chunk.
        expect(crossfadeMaxMs(480, null)).toBe(10);
        expect(crossfadeMaxMsForEngine(480, null, 0)).toBe(10);
        expect(crossfadeMaxMsForEngine(480, null, engineBlockMs(40))).toBe(40);
        // The running engine's chunk metric wins over the 48 kHz assumption for the chunk part.
        expect(crossfadeMaxMsForEngine(512, 10.0, 20)).toBe(20);
        // A short engine block never lowers the chunk cap.
        expect(crossfadeMaxMsForEngine(2048, null, 20)).toBe(42);
    });
});

describe("which voices the engine block clamps the crossfade of (P3B-H)", () => {
    const kinds: VoiceKind[] = ["none", "template", "legacy", "family"];

    it("only a template voice (or none listed) follows the block_ms engine block", () => {
        expect(kinds.filter(runsOnTemplateEngine)).toEqual(["none", "template"]);
        expect(crossfadeEngineBlockMs("template", true, 40)).toBe(40);
        expect(crossfadeEngineBlockMs("none", true, 30)).toBe(40);
        // The block follows the device chunk on automatic: no engine block to cap by.
        expect(crossfadeEngineBlockMs("template", true, 0)).toBe(0);
        // A bring-your-own voice follows the chunk, and a family clamps to its own fixed block.
        expect(crossfadeEngineBlockMs("legacy", true, 100)).toBe(0);
        expect(crossfadeEngineBlockMs("family", true, 100)).toBe(0);
        // Other modes have no RVC engine block.
        expect(crossfadeEngineBlockMs("template", false, 100)).toBe(0);
    });

    it("a family voice keeps the chunk cap on its crossfade slider", () => {
        // 512 frames at 48 kHz (10.67 ms) with blockMs 100: the family's overlap is clamped to its own block,
        // so the slider stops at the chunk, not at the 100 ms the template engine would run.
        const byFamily = crossfadeMaxMsForEngine(512, null, crossfadeEngineBlockMs("family", true, 100));
        const byTemplate = crossfadeMaxMsForEngine(512, null, crossfadeEngineBlockMs("template", true, 100));
        expect(byFamily).toBe(10);
        expect(byTemplate).toBe(100);
    });
});

describe("the device-block note (P3B-H)", () => {
    it("says nothing unless the divisor mode is on", () => {
        expect(deviceBlockNote(false, true, 10)).toBeNull();
        expect(deviceBlockNote(false, false, null)).toBeNull();
    });

    it("tells other voices that only template voices use it", () => {
        expect(deviceBlockNote(true, false, null)).toBe("template_only");
        expect(deviceBlockNote(true, false, 10.67)).toBe("template_only");
    });

    it("shows the running block, and warns when the run kept the chunk size", () => {
        expect(deviceBlockNote(true, true, null)).toBeNull();
        expect(deviceBlockNote(true, true, 0)).toBeNull();
        // 480 frames at 48 kHz and 441 at 44.1 kHz are both exactly 10 ms.
        expect(deviceBlockNote(true, true, 10)).toBe("live");
        expect(deviceBlockNote(true, true, (441 / 44_100) * 1000)).toBe("live");
        // The 512-frame default stayed (10.67 ms): the divisor did not apply.
        expect(deviceBlockNote(true, true, (512 / 48_000) * 1000)).toBe("not_applied");
        expect(deviceBlockNote(true, true, 21.33)).toBe("not_applied");
    });
});
