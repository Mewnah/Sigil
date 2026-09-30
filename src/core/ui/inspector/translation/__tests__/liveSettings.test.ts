import i18n from "i18next";
import { proxy } from "valtio";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ServiceNetworkState, TextEventType } from "@/types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { createTranslationPipeline, type TranslationEngineMap } from "@/core/services/translation/pipeline";
import { Service_Translation_Schema, Translation_Backends, type Translation_State } from "@/core/services/translation/schema";
import type { ITranslationEngine, TranslationRequest } from "@/core/services/translation/types";
import { mergeGlossary, newGlossaryEntry, parseGlossaryText } from "../logic";

/*
 * Acceptance check of P4-F: the glossary and context editors sit outside the inspector's Deactivatable and write the
 * same settings proxy the running pipeline reads per line, so an edit applies to the next line without a restart.
 * The edits below use the editor's own operations (add, bulk paste, edit a row, switch a row off).
 */

class RecordingEngine implements ITranslationEngine {
  static starts = 0;
  static requests: TranslationRequest[] = [];
  async start(): Promise<void> {
    RecordingEngine.starts++;
  }
  async translate(request: TranslationRequest): Promise<string> {
    RecordingEngine.requests.push(request);
    return `T(${request.text})`;
  }
  stop(): void {}
}

const engines = Object.fromEntries(Object.values(Translation_Backends).map((b) => [b, RecordingEngine])) as unknown as TranslationEngineMap;
const flush = async () => {
  for (let i = 0; i < 5; i++) await new Promise<void>((r) => setTimeout(r, 0));
};

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
});

describe("live settings", () => {
  it("glossary and context edits reach the next line of a running pipeline without a restart", async () => {
    const settings: Translation_State = proxy(Service_Translation_Schema.parse({}));
    settings.backend = Translation_Backends.llm;
    settings.sourceLanguage = "en";
    settings.targetLanguage = "ja";
    const published: string[] = [];
    const pipeline = createTranslationPipeline({
      name: "tr",
      settings: () => settings,
      engines,
      publish: (e) => e.type === TextEventType.final && published.push(e.value),
    });
    pipeline.start();
    await flush();
    expect(pipeline.state.status).toBe(ServiceNetworkState.connected);
    const startsBefore = RecordingEngine.starts;
    let n = 0;
    const line = async (text: string) => {
      n++;
      pipeline.push("stt", { type: TextEventType.final, value: text, emotes: {}, segmentId: `mic:1:${n}`, revision: 1 });
      await flush();
      return RecordingEngine.requests[RecordingEngine.requests.length - 1];
    };

    expect((await line("Welcome to Tokyo")).glossary).toBeUndefined();

    // Add one term (the add row).
    settings.glossary.push(newGlossaryEntry("Tokyo", "東京"));
    expect((await line("Welcome to Tokyo")).glossary).toEqual([{ source: "Tokyo", target: "東京" }]);

    // Paste a list (merged, duplicates skipped).
    const pasted = mergeGlossary(settings.glossary, parseGlossaryText("tokyo\tトーキョー\nMewnah = ミューナ").entries);
    expect(pasted.duplicates).toBe(1);
    settings.glossary = pasted.entries;
    expect((await line("Mewnah is in Tokyo")).glossary).toEqual([
      { source: "Tokyo", target: "東京" },
      { source: "Mewnah", target: "ミューナ" },
    ]);

    // Edit a row in place, then switch it off.
    settings.glossary[0].target = "とうきょう";
    expect((await line("Tokyo again")).glossary).toEqual([{ source: "Tokyo", target: "とうきょう" }]);
    settings.glossary[0].enabled = false;
    expect((await line("Tokyo again")).glossary).toBeUndefined();

    // Limit a term to another target language: it no longer applies to en → ja.
    settings.glossary[1].to = "ko";
    expect((await line("Mewnah says hi")).glossary).toBeUndefined();

    // Context background and turns apply live too.
    settings.context.background = "A cozy VRChat stream.";
    expect((await line("Hello")).background).toBe("A cozy VRChat stream.");
    settings.context.enabled = false;
    const off = await line("Hello again");
    expect(off.background).toBeUndefined();
    expect(off.context).toBeUndefined();

    expect(RecordingEngine.starts).toBe(startsBefore);
    expect(published).toHaveLength(n);
    pipeline.dispose();
  });
});
