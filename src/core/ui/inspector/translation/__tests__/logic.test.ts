import { filterGlossary, glossaryLanguageMatches, termOccurs } from "@/core/services/translation/glossary";
import { canonicalLanguage, TRANSLATION_LANGUAGES } from "@/core/services/translation/languages";
import { PROMPT_FORMAT_TRAITS } from "@/core/services/translation/prompt";
import {
  Service_Translation_Schema,
  TRANSLATION_ENGINE_IDS,
  TRANSLATION_GLOSSARY_MAX,
  TRANSLATION_PROMPT_FORMATS,
  Translation_Backends,
  type TranslationEngineId,
} from "@/core/services/translation/schema";
import { describe, expect, it } from "vitest";
import {
  addToChain,
  backendLabelKey,
  bergamotDirections,
  bergamotPlanFromPairs,
  customPromptIssues,
  deeplFormalityChoice,
  engineSetupIssue,
  enginesNotInChain,
  formatGlossaryText,
  formatScaled,
  glossaryIssues,
  isInternetAddress,
  isOnlineEngine,
  isSameLanguage,
  keptRowAfterRemove,
  languageDisplayName,
  languageOptions,
  LIBRE_PRESETS,
  libreServerKind,
  listsModel,
  LLM_SERVER_PRESETS,
  llmServerPresetOf,
  mergeGlossary,
  moveItem,
  newGlossaryEntry,
  parseBoundedNumber,
  parseGlossaryText,
  pickerValue,
  planBergamot,
  promptFormatUses,
  removeFromChain,
  searchGlossary,
  usagePercent,
  visibleGlossaryRows,
  type GlossaryEntry,
} from "../logic";

const defaults = () => Service_Translation_Schema.parse({});

describe("parseBoundedNumber", () => {
  it("accepts integers in range and rejects partial, fractional or out-of-range text", () => {
    expect(parseBoundedNumber("512", { min: 16, max: 8192 })).toBe(512);
    expect(parseBoundedNumber(" 16 ", { min: 16, max: 8192 })).toBe(16);
    expect(parseBoundedNumber("", { min: 0, max: 10 })).toBeNull();
    expect(parseBoundedNumber("-", { min: -1, max: 10 })).toBeNull();
    expect(parseBoundedNumber("1.5", { min: 0, max: 10 })).toBeNull();
    expect(parseBoundedNumber("15", { min: 16, max: 8192 })).toBeNull();
    expect(parseBoundedNumber("9000", { min: 16, max: 8192 })).toBeNull();
    expect(parseBoundedNumber("1e3", { min: 0, max: 10_000 })).toBeNull();
  });

  it("converts shown seconds to stored milliseconds", () => {
    expect(parseBoundedNumber("1.5", { min: 500, max: 60_000, scale: 1000 })).toBe(1500);
    expect(parseBoundedNumber("0,5", { min: 500, max: 60_000, scale: 1000 })).toBe(500);
    expect(parseBoundedNumber("0.4", { min: 500, max: 60_000, scale: 1000 })).toBeNull();
    expect(parseBoundedNumber("61", { min: 500, max: 60_000, scale: 1000 })).toBeNull();
  });

  it("keeps decimals for non-integer settings", () => {
    expect(parseBoundedNumber("0.25", { min: 0, max: 2, integer: false })).toBe(0.25);
    expect(parseBoundedNumber("2.5", { min: 0, max: 2, integer: false })).toBeNull();
  });

  it("formats stored values without float noise", () => {
    expect(formatScaled(1500, 1000)).toBe("1.5");
    expect(formatScaled(6000, 1000)).toBe("6");
    expect(formatScaled(0.30000000000000004)).toBe("0.3");
  });
});

describe("languages", () => {
  it("offers every table language once, sorted by name, with its code", () => {
    const opts = languageOptions("en");
    expect(opts).toHaveLength(TRANSLATION_LANGUAGES.length);
    expect(new Set(opts.map((o) => o.value)).size).toBe(opts.length);
    const ja = opts.find((o) => o.value === "ja");
    expect(ja?.label).toBe("Japanese (ja)");
    const names = opts.map((o) => o.label.replace(/ \([^)]*\)$/, ""));
    const collator = new Intl.Collator(["en", "en"]);
    for (let i = 1; i < names.length; i++) expect(collator.compare(names[i - 1], names[i])).toBeLessThanOrEqual(0);
  });

  it("names canonical zh as Simplified Chinese and falls back to the table's name", () => {
    expect(languageDisplayName("zh", "en")).toMatch(/simplified/i);
    expect(languageDisplayName("zh-Hant", "en")).toMatch(/traditional/i);
    expect(languageDisplayName("jpn_Jpan", "en")).toBe("Japanese");
    expect(languageDisplayName("not-a-language", "en")).toBe("not-a-language");
  });

  it("keeps every label unique, even where the runtime gives two codes one name", () => {
    for (const ui of ["en", "de", "ja", "ar", "zh"]) {
      const labels = languageOptions(ui).map((o) => o.label.replace(/ \([^)]*\)$/, ""));
      expect(new Set(labels).size).toBe(labels.length);
    }
  });

  it("localizes names to the UI language", () => {
    expect(languageDisplayName("ja", "de")).toBe("Japanisch");
    expect(languageOptions("de").find((o) => o.value === "de")?.label).toBe("Deutsch (de)");
  });

  it("shows canonical codes for hand-edited values and keeps unknown text", () => {
    expect(pickerValue("ja-JP")).toBe("ja");
    expect(pickerValue(" EN_us ")).toBe("en");
    expect(pickerValue("zh-TW")).toBe("zh-Hant");
    expect(pickerValue("xx-custom")).toBe("xx-custom");
    expect(pickerValue("")).toBe("");
    expect(isSameLanguage("en-US", "en")).toBe(true);
    expect(isSameLanguage("", "")).toBe(false);
  });
});

describe("glossary", () => {
  it("parses tabs, arrows, equals and CSV, skipping comments and half lines", () => {
    const { entries, skipped } = parseGlossaryText(
      [
        "﻿# game terms",
        "Tokyo\t東京",
        "Mewnah = ミューナ",
        "HP -> 体力",
        "MP → 魔力",
        '"Hello, world",こんにちは',
        "ignored line without separator",
        "orphan =",
        "",
        "Osaka\t大阪\tEN\tja",
      ].join("\r\n"),
    );
    expect(entries).toEqual([
      { source: "Tokyo", target: "東京", from: "", to: "" },
      { source: "Mewnah", target: "ミューナ", from: "", to: "" },
      { source: "HP", target: "体力", from: "", to: "" },
      { source: "MP", target: "魔力", from: "", to: "" },
      { source: "Hello, world", target: "こんにちは", from: "", to: "" },
      { source: "Osaka", target: "大阪", from: "en", to: "ja" },
    ]);
    expect(skipped).toBe(2);
  });

  it("keeps only language codes from the optional columns", () => {
    const parsed = parseGlossaryText(["Tokyo\t東京\tcity name", "Osaka\t大阪\tja-JP\tnotes", "Kyoto,京都,en,zh-TW"].join("\n"));
    expect(parsed.entries).toEqual([
      { source: "Tokyo", target: "東京", from: "", to: "" },
      { source: "Osaka", target: "大阪", from: "ja", to: "" },
      { source: "Kyoto", target: "京都", from: "en", to: "zh-Hant" },
    ]);
    expect(parsed.ignoredColumns).toBe(2);
    expect(parsed.skipped).toBe(0);
    // The pasted term is sent: a notes column no longer limits it to a pair that never occurs.
    const stored = parsed.entries.map((p) => newGlossaryEntry(p.source, p.target, p.from, p.to));
    expect(filterGlossary(stored, "Welcome to Tokyo", "en", "ja")).toEqual([{ source: "Tokyo", target: "東京" }]);
  });

  it("round-trips through the copied text", () => {
    const list = [newGlossaryEntry("Tokyo", "東京"), newGlossaryEntry("Osaka", "大阪", "en", "ja")];
    const text = formatGlossaryText(list);
    expect(text).toBe("Tokyo\t東京\nOsaka\t大阪\ten\tja");
    const back = parseGlossaryText(text).entries;
    expect(back.map((e) => [e.source, e.target, e.from, e.to])).toEqual([
      ["Tokyo", "東京", "", ""],
      ["Osaka", "大阪", "en", "ja"],
    ]);
    expect(formatGlossaryText([newGlossaryEntry("a\tb", "c\nd")])).toBe("a b\tc d");
  });

  it("round-trips terms that start with # or are wrapped in quotes", () => {
    const list = [
      newGlossaryEntry("#VRChat", "#VRChat JP"),
      newGlossaryEntry("Tokyo", "東京"),
      newGlossaryEntry('"Quoted"', "'single'"),
      newGlossaryEntry('#say "hi"', "#hi"),
    ];
    const text = formatGlossaryText(list);
    const parsed = parseGlossaryText(text);
    expect(parsed.skipped).toBe(0);
    expect(parsed.entries.map((e) => [e.source, e.target])).toEqual(list.map((e) => [e.source, e.target]));
    // A real comment line is still skipped.
    expect(parseGlossaryText("# comment\n\"#tag\" = #tag").entries).toEqual([{ source: "#tag", target: "#tag", from: "", to: "" }]);
  });

  it("merges without duplicates (case-insensitive, same pair) and respects the limit", () => {
    const existing = [newGlossaryEntry("Tokyo", "東京")];
    const merged = mergeGlossary(
      existing,
      [
        { source: "tokyo", target: "トーキョー", from: "", to: "" },
        { source: "Tokyo", target: "東京都", from: "en", to: "ja" },
        { source: "Kyoto", target: "京都", from: "", to: "" },
        { source: "KYOTO", target: "京都", from: "", to: "" },
      ],
      3,
    );
    expect(merged.added).toBe(2);
    expect(merged.duplicates).toBe(2);
    expect(merged.overflow).toBe(0);
    expect(merged.entries.map((e) => e.source)).toEqual(["Tokyo", "Tokyo", "Kyoto"]);
    expect(existing).toHaveLength(1);

    const full = mergeGlossary(merged.entries, [{ source: "Nara", target: "奈良", from: "", to: "" }], 3);
    expect(full.added).toBe(0);
    expect(full.overflow).toBe(1);
  });

  it("flags incomplete, shadowed and unknown-language entries", () => {
    const list = [
      newGlossaryEntry("Tokyo", "東京"),
      newGlossaryEntry("TOKYO", "トウキョウ"),
      newGlossaryEntry("Tokyo", "東京都", "en", ""),
      newGlossaryEntry("", "x"),
      { ...newGlossaryEntry("Tokyo", "Tokio"), caseSensitive: true },
      newGlossaryEntry("Osaka", "大阪", "city name", ""),
    ];
    const issues = glossaryIssues(list);
    // The pipeline sends one entry per term whatever its case or languages: the first entry covers all the others.
    expect([...issues.shadowed.keys()]).toEqual([1, 2, 4]);
    for (const i of [1, 2, 4]) expect(issues.shadowed.get(i)).toEqual({ kind: "always", by: 0 });
    expect([...issues.incomplete]).toEqual([3]);
    expect([...issues.unknownLanguage.entries()]).toEqual([[5, ["city name"]]]);
  });

  const entry = (source: string, target: string, extra: Partial<GlossaryEntry> = {}): GlossaryEntry => ({
    ...newGlossaryEntry(source, target),
    ...extra,
  });

  it("ignores switched-off entries (review probe a)", () => {
    const list = [entry("Tokyo", "東京", { enabled: false }), entry("tokyo", "トーキョー")];
    expect(glossaryIssues(list).shadowed.size).toBe(0);
    expect(filterGlossary(list, "Tokyo", "en", "ja")).toEqual([{ source: "tokyo", target: "トーキョー" }]);
  });

  it("flags a language-scoped entry after an any-language one (review probe b)", () => {
    const list = [entry("Tokyo", "東京"), entry("Tokyo", "도쿄", { to: "ko" })];
    expect(glossaryIssues(list).shadowed.get(1)).toEqual({ kind: "always", by: 0 });
    expect(filterGlossary(list, "Tokyo", "en", "ko")).toEqual([{ source: "Tokyo", target: "東京" }]);
  });

  it("reports the languages and spellings an earlier entry takes over", () => {
    const partly = glossaryIssues([entry("Tokyo", "東京", { to: "ja" }), entry("Tokyo", "Tokio")]).shadowed.get(1);
    expect(partly).toEqual({ kind: "partly", overlaps: [{ by: 0, from: "", to: "ja", spelling: null }] });
    const cased = glossaryIssues([entry("Tokyo", "東京", { caseSensitive: true }), entry("tokyo", "トーキョー", { from: "en" })]).shadowed.get(1);
    expect(cased).toEqual({ kind: "partly", overlaps: [{ by: 0, from: "en", to: "", spelling: "Tokyo" }] });
    expect(glossaryIssues([entry("Tokyo", "東京", { to: "ja" }), entry("Tokyo", "도쿄", { to: "ko" })]).shadowed.size).toBe(0);
  });

  describe("agrees with the pipeline's filterGlossary", () => {
    const lists: GlossaryEntry[][] = [
      [entry("Tokyo", "東京", { enabled: false }), entry("tokyo", "トーキョー")],
      [entry("Tokyo", "東京"), entry("Tokyo", "도쿄", { to: "ko" })],
      [entry("Tokyo", "東京", { to: "ja" }), entry("Tokyo", "Tokio")],
      [entry("Tokyo", "東京", { from: "en", to: "ja" }), entry("TOKYO", "トウキョウ", { from: "en-US", to: "ja-JP" })],
      [entry("Tokyo", "東京", { to: "ja" }), entry("Tokyo", "도쿄", { to: "ko" }), entry("tokyo", "Tokio")],
      [entry("Tokyo", "東京", { from: "de" }), entry("Tokyo", "トーキョー", { to: "ja" }), entry("Tokyo", "Tokio")],
      [entry("Tokyo", "東京", { caseSensitive: true }), entry("tokyo", "トーキョー")],
      [entry("tokyo", "トーキョー"), entry("TOKYO", "東京", { caseSensitive: true })],
      [entry("Tokyo", "東京", { caseSensitive: true }), entry("TOKYO", "トウキョウ", { caseSensitive: true })],
      [entry("Tokyo", "東京", { caseSensitive: true, to: "ja" }), entry("tokyo", "トーキョー", { from: "en" })],
      [entry("Tokyo", ""), entry("Tokyo", "東京")],
      [entry("Ｔｏｋｙｏ", "東京"), entry("Tokyo", "トーキョー", { enabled: true })],
      [entry("Tokyo", "東京", { to: "ja" }), entry("Tokyo", "Tokio", { to: "ja", enabled: false }), entry("Tokyo", "도쿄")],
    ];
    const pairs: [string, string][] = [
      ["en", "ja"],
      ["en", "ko"],
      ["de", "ja"],
      ["en", "de"],
      ["de", "ko"],
    ];
    const spellings = ["Tokyo", "tokyo", "TOKYO", "ToKyO"];

    it.each(lists.map((list, n) => [n, list] as const))("list %i", (_n, list) => {
      const issues = glossaryIssues(list);
      let checks = 0;
      list.forEach((e, i) => {
        if (!e.enabled || !e.source.trim() || !e.target.trim()) {
          expect(issues.shadowed.has(i)).toBe(false);
          return;
        }
        for (const [from, to] of pairs) {
          if (!glossaryLanguageMatches(e.from, from) || !glossaryLanguageMatches(e.to, to)) continue;
          for (const spelling of spellings) {
            const text = `Welcome to ${spelling} tonight`;
            if (!termOccurs(text, e.source, e.caseSensitive, from)) continue;
            const sent = filterGlossary(list, text, from, to).some((t) => t.target === e.target.trim());
            const shadow = issues.shadowed.get(i);
            let predicted = true;
            if (shadow?.kind === "always") predicted = false;
            else if (shadow?.kind === "partly") {
              predicted = !shadow.overlaps.some(
                (o) =>
                  (!o.from || canonicalLanguage(o.from) === canonicalLanguage(from)) &&
                  (!o.to || canonicalLanguage(o.to) === canonicalLanguage(to)) &&
                  (o.spelling === null || termOccurs(text, o.spelling, true, from)),
              );
            }
            expect({ i, from, to, spelling, sent }).toEqual({ i, from, to, spelling, sent: predicted });
            checks++;
          }
        }
      });
      expect(checks).toBeGreaterThan(0);
    });
  });

  it("searches terms and translations", () => {
    const list = [newGlossaryEntry("Tokyo", "東京"), newGlossaryEntry("Osaka", "大阪")];
    expect(searchGlossary(list, "")).toEqual([0, 1]);
    expect(searchGlossary(list, "osa")).toEqual([1]);
    expect(searchGlossary(list, "東")).toEqual([0]);
    expect(searchGlossary(list, "nara")).toEqual([]);
  });

  it("keeps the row being edited listed while a search runs", () => {
    const list = [newGlossaryEntry("Tokyo", "東京"), newGlossaryEntry("Kyoto", "京都"), newGlossaryEntry("Osaka", "大阪")];
    expect(visibleGlossaryRows(list, "Tok", null)).toEqual([0]);
    // Row 0 was retyped to "Kyoto2": it no longer matches but stays while it is being edited.
    const edited = [newGlossaryEntry("Kyoto2", "京都"), ...list.slice(1)];
    expect(visibleGlossaryRows(edited, "Tok", 0)).toEqual([0]);
    expect(visibleGlossaryRows(edited, "osa", 0)).toEqual([0, 2]);
    expect(visibleGlossaryRows(edited, "", 0)).toEqual([0, 1, 2]);
    expect(visibleGlossaryRows(edited, "Tok", 7)).toEqual([]);
    expect(keptRowAfterRemove(2, 0)).toBe(1);
    expect(keptRowAfterRemove(2, 2)).toBeNull();
    expect(keptRowAfterRemove(1, 2)).toBe(1);
    expect(keptRowAfterRemove(null, 0)).toBeNull();
  });

  it("new entries match the stored schema", () => {
    const parsed = Service_Translation_Schema.parse({ glossary: [newGlossaryEntry(" a ", " b ")] });
    expect(parsed.glossary).toEqual([{ source: "a", target: "b", from: "", to: "", caseSensitive: false, enabled: true }]);
    expect(TRANSLATION_GLOSSARY_MAX).toBeGreaterThan(0);
  });
});

describe("router chains", () => {
  const chain: TranslationEngineId[] = [Translation_Backends.llm, Translation_Backends.bergamot, Translation_Backends.nllb];

  it("moves, removes and adds engines without mutating the input", () => {
    expect(moveItem(chain, 2, -1)).toEqual(["llm", "nllb", "bergamot"]);
    expect(moveItem(chain, 0, -1)).toEqual(chain);
    expect(moveItem(chain, 2, 5)).toEqual(chain);
    expect(removeFromChain(chain, Translation_Backends.bergamot)).toEqual(["llm", "nllb"]);
    expect(removeFromChain([Translation_Backends.llm], Translation_Backends.llm)).toEqual(["llm"]);
    expect(addToChain(chain, Translation_Backends.deepl)).toEqual(["llm", "bergamot", "nllb", "deepl"]);
    expect(addToChain(chain, Translation_Backends.llm)).toEqual(chain);
    expect(addToChain(chain, "auto" as TranslationEngineId)).toEqual(chain);
    expect(chain).toEqual(["llm", "bergamot", "nllb"]);
  });

  it("lists the engines a chain does not use, and every chain the editor makes is valid", () => {
    const rest = enginesNotInChain(chain);
    expect(rest).toEqual(TRANSLATION_ENGINE_IDS.filter((id) => !chain.includes(id)));
    const all = rest.reduce((c, id) => addToChain(c, id), chain);
    expect(Service_Translation_Schema.parse({ auto: { chain: all } }).auto.chain).toEqual(all);
  });

  it("names engines with the provider list's keys", () => {
    expect(backendLabelKey(Translation_Backends.onDevice)).toBe("transl.backend_ondevice");
    expect(backendLabelKey(Translation_Backends.nllb)).toBe("transl.backend_nllb");
  });

  it("reports what an engine still needs", () => {
    const s = defaults();
    expect(engineSetupIssue(Translation_Backends.llm, s)).toBeNull();
    expect(engineSetupIssue(Translation_Backends.deepl, s)).toBe("transl.route_needs_key");
    expect(engineSetupIssue(Translation_Backends.azure, s)).toBe("transl.route_needs_key");
    expect(engineSetupIssue(Translation_Backends.bergamot, s)).toBeNull();
    expect(engineSetupIssue(Translation_Backends.nllb, s)).toBeNull();
    expect(engineSetupIssue(Translation_Backends.libretranslate, s)).toBeNull();

    s.llm.provider = "gemini";
    expect(engineSetupIssue(Translation_Backends.llm, s)).toBe("transl.route_needs_key");
    s.llm.geminiKey = "AIza";
    expect(engineSetupIssue(Translation_Backends.llm, s)).toBeNull();
    s.llm.provider = "custom";
    expect(engineSetupIssue(Translation_Backends.llm, s)).toBe("transl.route_needs_url");
    s.llm.customUrl = "http://localhost:11434/v1";
    expect(engineSetupIssue(Translation_Backends.llm, s)).toBe("transl.route_needs_model");
    s.libretranslate.endpoint = " ";
    expect(engineSetupIssue(Translation_Backends.libretranslate, s)).toBe("transl.route_needs_url");
  });

  it("marks engines that send text online, including self-hosted servers on the internet", () => {
    const s = defaults();
    expect(isOnlineEngine(Translation_Backends.deepl, s)).toBe(true);
    expect(isOnlineEngine(Translation_Backends.azure, s)).toBe(true);
    expect(isOnlineEngine(Translation_Backends.llm, s)).toBe(false);
    s.llm.provider = "openrouter";
    expect(isOnlineEngine(Translation_Backends.llm, s)).toBe(true);
    expect(isOnlineEngine(Translation_Backends.bergamot, s)).toBe(false);
    expect(isOnlineEngine(Translation_Backends.nllb, s)).toBe(false);

    expect(isOnlineEngine(Translation_Backends.libretranslate, s)).toBe(false);
    s.libretranslate.endpoint = "https://libretranslate.com";
    expect(isOnlineEngine(Translation_Backends.libretranslate, s)).toBe(true);
    s.libretranslate.endpoint = "http://192.168.1.20:5000";
    expect(isOnlineEngine(Translation_Backends.libretranslate, s)).toBe(false);

    s.llm.provider = "custom";
    s.llm.customUrl = "http://localhost:11434/v1";
    expect(isOnlineEngine(Translation_Backends.llm, s)).toBe(false);
    s.llm.customUrl = "https://my-vllm.example.com/v1";
    expect(isOnlineEngine(Translation_Backends.llm, s)).toBe(true);
    s.llm.customUrl = "";
    expect(isOnlineEngine(Translation_Backends.llm, s)).toBe(false);
  });

  it("tells internet addresses from this PC and the local network", () => {
    for (const local of [
      "http://localhost:5000",
      "localhost:5050",
      "http://127.0.0.1:8080/v1",
      "http://127.8.0.1",
      "http://[::1]:11434/v1",
      "http://10.0.0.5:5000",
      "http://172.16.4.2",
      "http://172.31.255.1",
      "http://192.168.1.20:5000",
      "http://169.254.10.1",
      "http://100.101.102.103:11434",
      "http://[fd12:3456::1]:8000",
      "http://[fe80::1]:8000",
      "http://[::ffff:192.168.1.2]:5000",
      "http://gpu-box:11434/v1",
      "http://nas.local:5000",
      "http://server.lan",
      "http://llm.home.arpa",
      "http://app.localhost:1234",
      "http://2130706433:5000",
    ]) {
      expect(isInternetAddress(local), local).toBe(false);
    }
    for (const remote of [
      "https://libretranslate.com",
      "https://my-vllm.example.com/v1",
      "http://8.8.8.8:5000",
      "http://172.32.0.1",
      "http://100.128.0.1",
      "http://[2001:db8::1]:5000",
      "http://[::ffff:8.8.8.8]",
      "api.openai.com",
    ]) {
      expect(isInternetAddress(remote), remote).toBe(true);
    }
    expect(isInternetAddress("")).toBe(false);
    expect(isInternetAddress("http://")).toBe(false);
  });
});

describe("bergamot directions", () => {
  const catalog = [
    { id: "mt-bergamot/ja_en", family: "mt-bergamot" },
    { id: "mt-bergamot/en_ja", family: "mt-bergamot" },
    { id: "mt-bergamot/en_de", family: "mt-bergamot" },
    { id: "mt-bergamot/de_en", family: "mt-bergamot" },
    { id: "mt-bergamot/en_zh-hant", family: "mt-bergamot" },
    { id: "runtime/bergamot", family: "runtime" },
    { id: "mt-bergamot/broken", family: "mt-bergamot" },
  ];
  const directions = bergamotDirections(catalog);

  it("reads directions from catalog ids", () => {
    expect(directions).toEqual([
      { source: "ja", target: "en", id: "mt-bergamot/ja_en" },
      { source: "en", target: "ja", id: "mt-bergamot/en_ja" },
      { source: "en", target: "de", id: "mt-bergamot/en_de" },
      { source: "de", target: "en", id: "mt-bergamot/de_en" },
      { source: "en", target: "zh-Hant", id: "mt-bergamot/en_zh-hant" },
    ]);
  });

  it("plans direct, pivot and unsupported pairs", () => {
    expect(planBergamot(directions, "ja", "en")).toEqual({ kind: "direct", ids: ["mt-bergamot/ja_en"] });
    expect(planBergamot(directions, "ja-JP", "de")).toEqual({ kind: "pivot", ids: ["mt-bergamot/ja_en", "mt-bergamot/en_de"] });
    expect(planBergamot(directions, "en", "zh-TW")).toEqual({ kind: "direct", ids: ["mt-bergamot/en_zh-hant"] });
    expect(planBergamot(directions, "ja", "ko")).toEqual({ kind: "unsupported" });
    expect(planBergamot(directions, "en", "ko")).toEqual({ kind: "unsupported" });
    expect(planBergamot(directions, "", "en")).toEqual({ kind: "unknown_source" });
    expect(planBergamot(directions, "en", "")).toEqual({ kind: "no_target" });
    expect(planBergamot(directions, "en", "en-GB")).toEqual({ kind: "same" });
  });

  it("prefers the engine's own pair list when it has the pair", () => {
    const pairs = [{ source: "ja", target: "de", ids: ["mt-bergamot/ja_en", "mt-bergamot/en_de"], pivot: true, installed: false }];
    expect(bergamotPlanFromPairs(pairs, "ja", "de")).toEqual({ kind: "pivot", ids: ["mt-bergamot/ja_en", "mt-bergamot/en_de"] });
    expect(bergamotPlanFromPairs(pairs, "ja", "en")).toBeNull();
    expect(bergamotPlanFromPairs(pairs, "", "de")).toBeNull();
    expect(bergamotPlanFromPairs([], "ja", "de")).toBeNull();
  });
});

describe("server presets", () => {
  it("recognizes LibreTranslate and LTEngine endpoints", () => {
    expect(libreServerKind(defaults().libretranslate.endpoint)).toBe("libretranslate");
    expect(libreServerKind("http://127.0.0.1:5050/")).toBe("ltengine");
    expect(libreServerKind("localhost:5050")).toBe("ltengine");
    expect(libreServerKind("http://192.168.1.20:5000")).toBe("custom");
    expect(libreServerKind("https://libretranslate.com")).toBe("custom");
    expect(libreServerKind("")).toBe("custom");
    expect(libreServerKind(LIBRE_PRESETS.ltengine.endpoint ?? "")).toBe("ltengine");
  });

  it("LTEngine lets the server detect the source and keeps a stored API key", () => {
    expect(LIBRE_PRESETS.ltengine).toMatchObject({ languageFrom: "auto", autoDetect: false });
    // Applying a preset must not erase the key of a hosted LibreTranslate server the user may switch back to.
    for (const preset of Object.values(LIBRE_PRESETS)) expect(preset).not.toHaveProperty("apiKey");
    const parsed = Service_Translation_Schema.parse({ libretranslate: LIBRE_PRESETS.ltengine });
    expect(parsed.libretranslate.endpoint).toBe("http://localhost:5050");
  });

  it("finds a model in a server's list, with or without an id prefix", () => {
    expect(listsModel(["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash"], "gemini-2.5-flash-lite")).toBe(true);
    expect(listsModel(["google/gemini-2.5-flash-lite"], "google/gemini-2.5-flash-lite")).toBe(true);
    expect(listsModel(["qwen3:8b"], "qwen3")).toBe(false);
    expect(listsModel(["qwen3:8b"], " ")).toBe(false);
  });

  it("matches custom LLM server URLs to presets", () => {
    for (const p of LLM_SERVER_PRESETS) expect(llmServerPresetOf(p.url)).toBe(p.id);
    expect(llmServerPresetOf("http://127.0.0.1:11434/v1/")).toBe("ollama");
    expect(llmServerPresetOf("http://192.168.1.5:11434/v1")).toBe("other");
    expect(llmServerPresetOf("")).toBe("other");
    expect(llmServerPresetOf("http://localhost:9999/v1")).toBe("other");
  });
});

describe("prompts and DeepL", () => {
  it("reports what each prompt family sends", () => {
    expect(promptFormatUses("hy-mt2")).toEqual({ context: true, glossary: true, background: true });
    expect(promptFormatUses("milmmt")).toEqual({ context: false, glossary: false, background: false });
    for (const f of TRANSLATION_PROMPT_FORMATS) if (f !== "auto") expect(PROMPT_FORMAT_TRAITS[f]).toBeDefined();
  });

  it("checks custom templates", () => {
    expect(customPromptIssues("")).toEqual({ missingText: false, unknown: [] });
    expect(customPromptIssues("Translate {text} into {target_lang}")).toEqual({ missingText: false, unknown: [] });
    expect(customPromptIssues("Into {target_lang} using {glosary} and {glosary}")).toEqual({ missingText: true, unknown: ["glosary"] });
    expect(customPromptIssues("{background}{context}{source_lang}{text}").unknown).toEqual([]);
  });

  it("maps DeepL formality and usage", () => {
    expect(deeplFormalityChoice("prefer_more")).toBe("more");
    expect(deeplFormalityChoice("less")).toBe("less");
    expect(deeplFormalityChoice("default")).toBe("default");
    expect(usagePercent({ characterCount: 450_000, characterLimit: 500_000 })).toBe(90);
    expect(usagePercent({ characterCount: 600_000, characterLimit: 500_000 })).toBe(100);
    expect(usagePercent({ characterCount: 10, characterLimit: 0 })).toBeNull();
    expect(usagePercent(null)).toBeNull();
  });
});
