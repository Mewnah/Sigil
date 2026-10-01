/**
 * The voice-library IPC contract (`contract.ts`) against the Rust plugin: every command is in the plugin's ACL
 * command list (`src-tauri/plugin-commands/voice-library.txt`, what build.rs turns into permissions) and every
 * listed command is named; voice ids follow Rust's `valid_voice_id`.
 */
import { describe, expect, it } from "vitest";

import commandList from "../../../../../src-tauri/plugin-commands/voice-library.txt?raw";
import { validVoiceId, VL_COMMANDS } from "../contract";

const PREFIX = "plugin:voice-library|";
const listed = commandList
    .split(/\r?\n/)
    .map((l) => l.split("#")[0].trim())
    .filter((l) => l !== "");

describe("voice-library IPC contract", () => {
    it("every command is a voice-library command in the ACL list, and every listed command is named", () => {
        const names = Object.values(VL_COMMANDS).map((c) => {
            expect(c.startsWith(PREFIX), c).toBe(true);
            return c.slice(PREFIX.length);
        });
        expect([...names].sort()).toEqual([...listed].sort());
    });

    it("voice ids are safe folder names", () => {
        for (const ok of ["alice", "voice-2", "a"]) expect(validVoiceId(ok), ok).toBe(true);
        for (const bad of ["", "-a", "a-", "A", "a/b", "..", "a b", "x".repeat(65), 3, null]) expect(validVoiceId(bad), String(bad)).toBe(false);
    });
});
