import { describe, expect, it } from "vitest";
import { Service_STT_Schema } from "../schema";
import { deepgramListenUrl } from "../services/deepgram";
import { deepGramLangs, normalizeDeepgramLanguage } from "../stt_data";

const listenParams = (deepgram: Record<string, unknown>) => {
  const state = Service_STT_Schema.parse({ deepgram });
  return new URL(deepgramListenUrl(state)).searchParams;
};

describe("Deepgram language codes", () => {
  it("lists only BCP-47 codes Deepgram accepts (no `zn-*` typo)", () => {
    const codes = Object.values(deepGramLangs).flatMap((group) => group.map(([code]) => code));
    expect(codes).toContain("zh-CN");
    expect(codes).toContain("zh-TW");
    for (const code of codes) expect(code, code).toMatch(/^(?!zn)[a-z]{2,3}(-[A-Z]{2})?$/);
  });

  it("maps the legacy `zn-*` values, keeps everything else", () => {
    expect(normalizeDeepgramLanguage("zn-CN")).toBe("zh-CN");
    expect(normalizeDeepgramLanguage("zn-TW")).toBe("zh-TW");
    expect(normalizeDeepgramLanguage(" zn ")).toBe("zh");
    expect(normalizeDeepgramLanguage("en-US")).toBe("en-US");
    expect(normalizeDeepgramLanguage("nl")).toBe("nl");
    expect(normalizeDeepgramLanguage("")).toBe("en-US");
  });

  it("a stored Chinese selection from an older build loads and connects as zh-*", () => {
    expect(Service_STT_Schema.parse({ deepgram: { language: "zn-CN" } }).deepgram.language).toBe("zh-CN");
    const params = listenParams({ language: "zn-TW", model: "nova-3" });
    expect(params.get("language")).toBe("zh-TW");
    expect(params.get("model")).toBe("nova-3");
  });

  it("sends the model with the chosen options", () => {
    const params = listenParams({ language: "en-GB", punctuate: false, interim: true, profanity: false });
    expect(Object.fromEntries(params)).toEqual({
      model: "nova-3",
      language: "en-GB",
      punctuate: "false",
      interim_results: "true",
      profanity_filter: "false",
    });
  });
});
