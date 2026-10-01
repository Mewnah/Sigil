/**
 * The TS IPC contract (`ipc.ts`) against the Rust side: every command is in the plugin's ACL command list
 * (`src-tauri/plugin-commands/voice-changer.txt`, what build.rs turns into permissions), the P3 commands and
 * events are the frozen contract's (`params_contract.json`), and the normalizers keep only contract values.
 */
import { describe, expect, it } from "vitest";

import commandList from "../../../../../src-tauri/plugin-commands/voice-changer.txt?raw";
import contract from "../../../../../src-tauri/src/services/voice_changer/params_contract.json";
import { BASELINE_PITCH_EXTRACTORS, VC_LEVERS } from "../contract";
import {
    BASELINE_CAPABILITIES,
    familyOfFormat,
    IMPORT_ERROR_CODES,
    importErrorCode,
    normalizeCapabilities,
    normalizeEffectiveLevers,
    normalizeFamilyCapabilities,
    VC_COMMANDS,
    VC_EVENTS,
    VC_FAMILIES,
    VC_FAMILY_CONTROLS,
    VC_FAMILY_CREATION,
    VC_FAMILY_FORMATS,
} from "../ipc";

const PREFIX = "plugin:voice-changer|";
const listed = commandList
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l !== "" && !l.startsWith("#"));

describe("voice-changer IPC contract", () => {
    it("every command is a voice-changer command in the ACL list, and every listed command is named", () => {
        const names = Object.values(VC_COMMANDS).map((c) => {
            expect(c.startsWith(PREFIX), c).toBe(true);
            return c.slice(PREFIX.length);
        });
        for (const n of names) expect(listed, n).toContain(n);
        expect([...names].sort()).toEqual([...listed].sort());
    });

    it("the P3 commands and events are the frozen contract's", () => {
        const p3 = Object.keys(contract.commands_p3).filter((k) => !k.startsWith("$"));
        const names = Object.values(VC_COMMANDS).map((c) => c.slice(PREFIX.length));
        for (const c of p3) expect(names, c).toContain(c);
        const events = Object.values(VC_EVENTS) as string[];
        for (const e of Object.keys(contract.events_p3)) expect(events, e).toContain(e);
        expect(Object.keys(BASELINE_CAPABILITIES).sort()).toEqual([...contract.capabilities.fields].sort());
        expect([...IMPORT_ERROR_CODES]).toEqual(contract.import_error_codes);
    });

    it("capabilities keep only contract values and always the baseline", () => {
        expect(normalizeCapabilities(null)).toEqual(BASELINE_CAPABILITIES);
        const c = normalizeCapabilities({
            pitch_extractors: ["swiftf0", "SWIFTF0", "harvest", 3],
            rvc_execution_providers: ["tensorrt_rtx", "vulkan"],
            rvc_presets: ["studio_trt", "turbo"],
            precisions: ["fp16", "int4"],
            denoiser: ["gtcrn"],
            denoisers: ["gtcrn", "rnnoise"],
            levers: ["cuda_graphs", "warp_drive"],
            slot_formats: ["sigil-rvc-1"],
            importer: true,
            autotune: "yes",
            live_record: false,
            max_hot_slots: 4.7,
        });
        expect(c.pitch_extractors).toEqual([...BASELINE_PITCH_EXTRACTORS, "swiftf0"]);
        expect(c.rvc_execution_providers).toEqual(["cpu", "cuda", "dml", "tensorrt_rtx"]);
        expect(c.rvc_presets).toEqual(["streamer", "low_latency", "quality", "studio_trt"]);
        expect(c.precisions).toEqual(["auto", "fp32", "fp16"]);
        expect(c.denoisers).toEqual(["off", "gtcrn"]);
        expect(c.levers).toEqual(["cuda_graphs"]);
        expect(c.slot_formats).toEqual(["rvc", "sigil-rvc-1"]);
        expect([c.importer, c.autotune, c.live_record, c.max_hot_slots]).toEqual([true, false, false, 4]);
    });

    it("levers in use normalize to booleans for every lever", () => {
        const l = normalizeEffectiveLevers({ io_binding: true, accelerator: "warp", notes: "x" })!;
        for (const lever of VC_LEVERS) expect(typeof l[lever], lever).toBe("boolean");
        expect(l.io_binding).toBe(true);
        expect(l.accelerator).toBe("none");
        expect(l.precision).toBe("auto");
        expect(l.notes).toEqual([]);
        expect(normalizeEffectiveLevers(undefined)).toBeNull();
    });

    it("import errors expose their stable code", () => {
        expect(importErrorCode("pickle_rejected: global os.system is not allowed")).toBe("pickle_rejected");
        expect(importErrorCode(new Error("time_limit: the import took longer than 120 s"))).toBe("time_limit");
        expect(importErrorCode("something else broke")).toBeNull();
        expect(importErrorCode(42)).toBeNull();
    });
});

describe("voice families (P5-S0)", () => {
    it("the P5 commands and the families are the frozen contract's", () => {
        const names = Object.values(VC_COMMANDS).map((c) => c.slice(PREFIX.length));
        for (const c of Object.keys(contract.commands_p5).filter((k) => !k.startsWith("$"))) expect(names, c).toContain(c);
        expect([...VC_FAMILIES]).toEqual(contract.families.ids);
        expect(VC_FAMILY_FORMATS).toEqual(contract.families.formats);
        expect([...VC_FAMILY_CREATION]).toEqual(contract.families.creation);
        expect([...VC_FAMILY_CONTROLS]).toEqual(contract.families.controls);
        expect(contract.capabilities.fields).toContain("families");
    });

    it("family capabilities keep known families once, in order, with typed fields", () => {
        const fams = normalizeFamilyCapabilities([
            { family: "sidecar_v2", available: true, creation: "sidecar", execution_providers: ["cuda", "vulkan"], controls: ["pitch", "warp"] },
            { family: "sigil_zs", available: "yes", zero_shot: true, creation: "teleport" },
            { family: "sigil_zs", available: true },
            { family: "nope" },
            null,
        ]);
        expect(fams.map((f) => f.family)).toEqual(["sigil_zs", "sidecar_v2"]);
        expect(fams[0]).toMatchObject({ format: "sigil-zs-1", available: false, zero_shot: true, creation: "reference" });
        expect(fams[1]).toMatchObject({ available: true, execution_providers: ["cuda"], controls: ["pitch"] });
        expect(normalizeFamilyCapabilities("x")).toEqual([]);
        expect(normalizeCapabilities({ families: [{ family: "meanvc", available: true }] }).families.map((f) => f.family)).toEqual(["meanvc"]);
        expect(familyOfFormat("ddsp-svc-6.3")).toBe("ddsp_svc");
        expect(familyOfFormat(" SOVITS ")).toBe("sovits");
        expect(familyOfFormat("sigil-rvc-1")).toBeNull();
        expect(familyOfFormat(undefined)).toBeNull();
    });
});
