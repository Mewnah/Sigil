/**
 * Server-side render tests of the shared reference-voice recorder and picker (P5-F,
 * `src/core/ui/inspector/components/ReferenceVoiceRecorder.tsx`) against the real voice-library state with a
 * stubbed Tauri: what each state shows (idle, recording, a take that cannot be used, a take ready to save, another
 * panel recording, re-recording a voice) and what the picker lists. Both consumers (the voice changer's zero-shot
 * voices and TTS cloning) read the same `voiceLibraryUi.voices`, so a saved voice appears in both.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const env = vi.hoisted(() => ({ tauri: true }));

vi.mock("react-i18next", () => ({
    useTranslation: () => ({
        t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key}${JSON.stringify(opts)}` : key),
        i18n: { language: "en" },
    }),
}));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => env.tauri, invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn(), info: vi.fn() }) }));
vi.mock("@/core/ui/inspector/components/input", () => ({
    InputSelect: ({ label, value, options }: { label: string; value: string; options: { label: string; value: string }[] }) => (
        <label>
            {label}
            <select defaultValue={value}>
                {options.map((o) => (
                    <option key={o.value} value={o.value}>
                        {o.label}
                    </option>
                ))}
            </select>
        </label>
    ),
    InputRustAudioInput: ({ label, value }: { label: string; value: string }) => <div data-device={value}>{label}</div>,
}));

import { __resetVoiceLibraryHostForTests, voiceLibraryUi, type ClipIssue, type VoiceMeta, type VoiceTake } from "@/core/services/voice_library";
import { ClipPreview, ReferenceVoicePicker, ReferenceVoiceRecorder, type ClipPreviewEnv } from "@/core/ui/inspector/components/ReferenceVoiceRecorder";

/** Rendered markup with the attribute escapes undone (the mocked `t` prints its options as JSON). */
function render(el: ReactElement): string {
    return renderToStaticMarkup(el).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

const voice = (id: string, name: string, over: Partial<VoiceMeta> = {}): VoiceMeta => ({
    id,
    name,
    createdUnixMs: 1,
    durationMs: 6400,
    sampleRate: 48000,
    lufs: -20,
    snrDb: 38,
    peakDbfs: -2,
    clipped: false,
    transcript: null,
    language: null,
    source: "recorded",
    sha256: "ab",
    cachedBackends: [],
    ...over,
});

const take = (issues: ClipIssue[], ok: boolean, over: Partial<VoiceTake> = {}): VoiceTake => ({
    takeId: "take-0000000000aa",
    quality: { durationMs: 6200, trimmedMs: 2100, snrDb: 31.6, peakDbfs: -7, clippedRatio: 0, lufsBefore: -29, lufsAfter: -20, ok, issues },
    source: "recorded",
    suggestedName: null,
    sampleRate: 48000,
    ...over,
});

/** React warnings (missing keys, invalid props, controlled-input misuse) fail the test. */
let warnings: string[] = [];

beforeEach(() => {
    env.tauri = true;
    __resetVoiceLibraryHostForTests();
    vi.stubGlobal("window", { ApiServer: { voiceChanger: { state: { voiceChanger: { inputDevice: "USB Mic" } } } } });
    warnings = [];
    vi.spyOn(console, "error").mockImplementation((msg: unknown, ...rest: unknown[]) => {
        warnings.push([msg, ...rest].map(String).join(" "));
    });
});

afterEach(() => {
    vi.unstubAllGlobals();
    expect(warnings).toEqual([]);
});

describe("<ReferenceVoiceRecorder>", () => {
    it("renders nothing outside the desktop app", () => {
        env.tauri = false;
        expect(render(<ReferenceVoiceRecorder />)).toBe("");
        expect(render(<ReferenceVoicePicker value="" onChange={() => {}} />)).toBe("");
    });

    it("idle: explains the take, offers the voice changer's microphone, recording and importing", () => {
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).toContain("voice_library.title");
        expect(html).toContain("voice_library.description");
        expect(html).toContain(">voice_library.tips<");
        expect(html).toContain("voice_library.prompt_sentence");
        expect(html).toContain('data-device="USB Mic"');
        expect(html).toMatch(/<button[^>]*>.*voice_library\.record<\/button>/);
        expect(html).toContain("voice_library.import");
        expect(html).not.toContain("voice_library.busy_elsewhere");

        const compact = render(<ReferenceVoiceRecorder compact onClose={() => {}} />);
        expect(compact).toContain("voice_library.new_voice");
        expect(compact).toContain("voice_library.tips_compact");
        expect(compact).toContain('aria-label="voice_library.close"');
        expect(compact).not.toContain("data-device");
        expect(compact).not.toContain("voice_library.description");
    });

    it("recording: live meters, the speech heard against 3-10 s, stop and cancel", () => {
        voiceLibraryUi.recording = true;
        voiceLibraryUi.progress = { elapsedMs: 4200, levelDb: -24, peakDb: -9, clipping: false, voicedMs: 3500 };
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).toContain("voice_library.recording");
        expect(html).toContain('voice_library.elapsed{"elapsed":"0:04","max":"0:12"}');
        expect(html).toContain("voice_library.level_good");
        expect(html).toMatch(/role="meter" aria-label="voice_library.level"[^>]*aria-valuenow="60"/);
        expect(html).toContain('voice_library.speech_heard{"seconds":"3.5"}');
        expect(html).toContain('voice_library.speech_enough{"min":3,"max":10}');
        expect(html).toContain("voice_library.stop<");
        expect(html).toContain("voice_library.cancel");
        expect(html).not.toContain("voice_library.import");

        voiceLibraryUi.progress = { elapsedMs: 900, levelDb: -12, peakDb: -0.2, clipping: true, voicedMs: 600 };
        const hot = render(<ReferenceVoiceRecorder />);
        expect(hot).toContain("voice_library.level_hot");
        expect(hot).toContain("voice_library.speech_more");
    });

    it("a take that cannot be used: its blocking issue first, no save", () => {
        voiceLibraryUi.take = take(["noisy", "too_short"], false);
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).toContain("voice_library.take_title_recorded");
        expect(html).toContain('voice_library.take_summary{"seconds":"6.2","trimmed":"2.1"}');
        expect(html).toContain('voice_library.snr{"snr":32}');
        expect(html.indexOf("voice_library.issue_too_short")).toBeGreaterThan(-1);
        expect(html.indexOf("voice_library.issue_too_short")).toBeLessThan(html.indexOf("voice_library.issue_noisy"));
        expect(html).toContain("voice_library.issue_blocking_hint");
        expect(html).toContain("voice_library.play<");
        expect(html).toContain("voice_library.record_again");
        expect(html).toContain("voice_library.discard");
        expect(html).not.toContain("voice_library.save<");
        expect(html).not.toContain("voice_library.name<");
    });

    it("a usable take: name, language and transcript choice; saving needs a name", () => {
        voiceLibraryUi.take = take(["too_long"], true, { source: "imported", suggestedName: "Studio take" });
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).toContain("voice_library.take_title_imported");
        expect(html).toContain("voice_library.issue_too_long");
        expect(html).not.toContain("voice_library.issue_blocking_hint");
        expect(html).toContain("voice_library.name<");
        expect(html).toMatch(/<select><option value="auto" selected="">voice_library.language_auto<\/option>/);
        expect(html).toMatch(/<input type="checkbox"[^>]*checked=""/);
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*>voice_library\.save<\/button>/);
    });

    it("another panel's recording keeps this one idle and blocked", () => {
        voiceLibraryUi.owner = "another-panel";
        voiceLibraryUi.recording = true;
        voiceLibraryUi.progress = { elapsedMs: 1000, levelDb: -30, peakDb: -12, clipping: false, voicedMs: 800 };
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).not.toContain("voice_library.recording");
        expect(html).toContain("voice_library.busy_elsewhere");
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*voice_library\.record<\/button>/);
    });

    it("a take waiting in another panel blocks recording here (it would be discarded)", () => {
        voiceLibraryUi.owner = "another-panel";
        voiceLibraryUi.take = take([], true);
        const html = render(<ReferenceVoiceRecorder />);
        expect(html).not.toContain("voice_library.take_title_recorded");
        expect(html).toContain("voice_library.take_elsewhere");
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*voice_library\.record<\/button>/);
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*>.*voice_library\.import<\/button>/);
        // Once that panel let go of the recorder (it closed), any panel shows the take.
        voiceLibraryUi.owner = null;
        expect(render(<ReferenceVoiceRecorder />)).toContain("voice_library.take_title_recorded");
    });

    it("while a save or replace runs, or a new recording starts, the take cannot be saved again", () => {
        voiceLibraryUi.voices = [voice("alice-1a2b3c", "Alice")];
        voiceLibraryUi.loaded = true;
        voiceLibraryUi.take = take([], true);
        voiceLibraryUi.busy = "saving";
        const replacing = render(<ReferenceVoiceRecorder compact replaceVoiceId="alice-1a2b3c" />);
        expect(replacing).toContain("voice_library.saving");
        expect(replacing).toMatch(/<button[^>]*disabled=""[^>]*>voice_library\.replace<\/button>/);
        voiceLibraryUi.busy = "starting";
        const starting = render(<ReferenceVoiceRecorder compact replaceVoiceId="alice-1a2b3c" />);
        expect(starting).toMatch(/<button[^>]*disabled=""[^>]*>voice_library\.replace<\/button>/);
        expect(starting).toMatch(/<button[^>]*disabled=""[^>]*>.*voice_library\.record_again<\/button>/);
        expect(starting).toMatch(/<button[^>]*disabled=""[^>]*>.*voice_library\.discard<\/button>/);
    });

    it("re-recording a library voice replaces its clip under its name and keeps its language", () => {
        voiceLibraryUi.voices = [voice("alice-1a2b3c", "Alice", { language: "de" }), voice("kit-7a8b9c", "Kit", { language: "yue" })];
        voiceLibraryUi.loaded = true;
        voiceLibraryUi.take = take([], true);
        const html = render(<ReferenceVoiceRecorder compact replaceVoiceId="alice-1a2b3c" onClose={() => {}} />);
        expect(html).toContain('voice_library.rerecord_title{"name":"Alice"}');
        expect(html).not.toContain("voice_library.name<");
        expect(html).toMatch(/<option value="de" selected="">/);
        expect(html).toMatch(/<button[^>]*>voice_library\.replace<\/button>/);
        expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>voice_library\.replace<\/button>/);
        // A detected language the list does not offer stays selected rather than reading "Detect".
        const kit = render(<ReferenceVoiceRecorder compact replaceVoiceId="kit-7a8b9c" />);
        expect(kit).toMatch(/<option value="yue" selected="">/);
    });
});

describe("<ReferenceVoicePicker>", () => {
    it("lists the library's voices with the selected one, its preview and editing", () => {
        voiceLibraryUi.voices = [voice("alice-1a2b3c", "Alice", { transcript: "hello there" }), voice("bob-4d5e6f", "Bob", { durationMs: 9100 })];
        voiceLibraryUi.loaded = true;
        const html = render(<ReferenceVoicePicker value="bob-4d5e6f" onChange={() => {}} needsTranscript />);
        expect(html).toContain("voice_library.picker_label");
        expect(html).toContain(">voice_library.picker_none<");
        expect(html).toContain('<option value="alice-1a2b3c">voice_library.voice_option{"name":"Alice","seconds":"6.4"}</option>');
        expect(html).toContain('<option value="bob-4d5e6f" selected="">voice_library.voice_option{"name":"Bob","seconds":"9.1"}</option>');
        expect(html).toContain("voice_library.play_voice");
        expect(html).toContain("voice_library.new_voice");
        expect(html).toContain("voice_library.manage");
        // Bob has no transcript and TTS cloning wants one.
        expect(html).toContain("voice_library.transcript_needed");
        expect(render(<ReferenceVoicePicker value="alice-1a2b3c" onChange={() => {}} needsTranscript />)).not.toContain("voice_library.transcript_needed");
        voiceLibraryUi.voices = [voice("bob-4d5e6f", "Bob", { transcribing: true })];
        expect(render(<ReferenceVoicePicker value="bob-4d5e6f" onChange={() => {}} needsTranscript />)).toContain("voice_library.transcribing");
    });

    it("a voice that left the library, an empty library, and no 'None' choice", () => {
        voiceLibraryUi.voices = [voice("alice-1a2b3c", "Alice")];
        voiceLibraryUi.loaded = true;
        const missing = render(<ReferenceVoicePicker value="gone-000001" onChange={() => {}} />);
        expect(missing).toContain('<option value="gone-000001" selected="">voice_library.picker_missing{"id":"gone-000001"}</option>');
        expect(missing).toContain("voice_library.picker_missing_hint");
        expect(missing).not.toContain("voice_library.manage");

        // No "None" and nothing chosen: a placeholder is shown, never a voice the consumer has not got (no preview,
        // no editing, no transcript hint for it).
        const required = render(<ReferenceVoicePicker value="" allowNone={false} onChange={() => {}} needsTranscript />);
        expect(required).not.toContain("voice_library.picker_none");
        expect(required).toContain('<option value="__choose__" selected="">voice_library.picker_choose</option>');
        expect(required).toContain('<option value="alice-1a2b3c">');
        expect(required).not.toContain("voice_library.play_voice");
        expect(required).not.toContain("voice_library.manage");
        expect(required).not.toContain("voice_library.transcript_needed");
        // Once one is chosen the placeholder goes.
        const chosen = render(<ReferenceVoicePicker value="alice-1a2b3c" allowNone={false} onChange={() => {}} />);
        expect(chosen).not.toContain("voice_library.picker_choose");
        expect(chosen).toContain('<option value="alice-1a2b3c" selected="">');

        voiceLibraryUi.voices = [];
        const empty = render(<ReferenceVoicePicker value="" onChange={() => {}} />);
        expect(empty).toContain("voice_library.picker_empty");
        expect(empty).not.toContain("voice_library.play_voice");
        // Before the first listing nothing is called missing or empty.
        voiceLibraryUi.loaded = false;
        const loading = render(<ReferenceVoicePicker value="gone-000001" onChange={() => {}} />);
        expect(loading).not.toContain("voice_library.picker_missing");
        expect(loading).not.toContain("voice_library.picker_empty");
    });
});

describe("ClipPreview (the preview player of both panels)", () => {
    /** A webview stand-in: object URLs and audio elements that record what happened to them. */
    function fakeWebview() {
        const created: string[] = [];
        const revoked: string[] = [];
        const audios: { url: string; playing: boolean; ended: (() => void) | null }[] = [];
        const env: ClipPreviewEnv = {
            createUrl: () => {
                const url = `blob:${created.length + 1}`;
                created.push(url);
                return url;
            },
            revokeUrl: (url) => {
                revoked.push(url);
            },
            createAudio: (url) => {
                const a = { url, playing: false, ended: null as (() => void) | null };
                audios.push(a);
                return {
                    play: async () => {
                        a.playing = true;
                    },
                    pause: () => {
                        a.playing = false;
                    },
                    onEnded: (callback) => {
                        a.ended = callback;
                    },
                };
            },
        };
        return { env, created, revoked, audios };
    }

    function pending() {
        let resolve: (bytes: ArrayBuffer) => void = () => {};
        let reject: (e: unknown) => void = () => {};
        const promise = new Promise<ArrayBuffer>((res, rej) => {
            resolve = res;
            reject = rej;
        });
        return { promise, resolve, reject };
    }

    it("plays one clip at a time: a clip whose bytes arrive after a newer play is dropped", async () => {
        const web = fakeWebview();
        const states: (string | null)[] = [];
        const player = new ClipPreview((key) => states.push(key), web.env);
        const a = pending();
        const b = pending();
        const playA = player.play("voice-a", () => a.promise);
        const playB = player.play("voice-b", () => b.promise);
        b.resolve(new ArrayBuffer(8));
        await playB;
        a.resolve(new ArrayBuffer(8));
        await playA;
        expect(web.audios.map((x) => [x.url, x.playing])).toEqual([["blob:1", true]]);
        expect(states.at(-1)).toBe("voice-b");
        player.stop();
        expect(web.audios[0].playing).toBe(false);
        expect(web.revoked).toEqual(web.created);
        expect(states.at(-1)).toBe(null);
    });

    it("nothing plays when a stop (or the panel closing) comes while the clip loads", async () => {
        const web = fakeWebview();
        const states: (string | null)[] = [];
        const player = new ClipPreview((key) => states.push(key), web.env);
        const a = pending();
        const playA = player.play("take-1", () => a.promise);
        player.stop();
        a.resolve(new ArrayBuffer(8));
        await playA;
        expect(web.audios).toEqual([]);
        expect(web.created).toEqual([]);
        expect(states).not.toContain("take-1");
    });

    it("a clip that ends stops; a superseded load error is dropped, a current one is reported", async () => {
        const web = fakeWebview();
        const player = new ClipPreview(() => {}, web.env);
        await player.play("voice-a", async () => new ArrayBuffer(8));
        expect(web.audios[0].playing).toBe(true);
        web.audios[0].ended?.();
        expect(web.audios[0].playing).toBe(false);
        expect(web.revoked).toEqual(["blob:1"]);

        const stale = pending();
        const playStale = player.play("voice-b", () => stale.promise);
        const current = player.play("voice-c", async () => new ArrayBuffer(8));
        stale.reject(new Error("cannot read the clip"));
        await expect(playStale).resolves.toBeUndefined();
        await current;
        await expect(player.play("voice-d", async () => Promise.reject(new Error("cannot read the clip")))).rejects.toThrow("cannot read the clip");
        // A late end event of an earlier clip does not stop the newer one.
        await player.play("voice-e", async () => new ArrayBuffer(8));
        const lateEnd = web.audios[2].ended;
        await player.play("voice-f", async () => new ArrayBuffer(8));
        lateEnd?.();
        expect(web.audios.map((x) => x.playing)).toEqual([false, false, false, true]);
    });
});
