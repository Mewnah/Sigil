import { describe, expect, it, vi } from "vitest";
import {
  OthersEventSource,
  ReadingsEventSchema,
  TextEventSource,
  TextEventType,
  type ReadingsEvent,
  type RustReadingsRequest,
  type RustReadingsResult,
  type TextEvent,
} from "@/types";
import {
  JAPANESE_DICTIONARY_ID,
  MAX_PENDING,
  MAX_READINGS_TEXT,
  ReadingsAnnotator,
  isDictionaryError,
  readingsLanguage,
  readingsRequest,
  sourceSelected,
  toReadingsEvent,
  utteranceIdOf,
  type ReadingsAnnotatorDeps,
} from "../annotator";
import { Readings_Schema, type ReadingsState } from "../schema";

function settings(over: Partial<ReadingsState> = {}): ReadingsState {
  return { ...Readings_Schema.parse({}), enable: true, ...over };
}

function line(value: string, over: Partial<TextEvent> = {}): TextEvent {
  return { type: TextEventType.final, value, emotes: {}, ...over };
}

function result(over: Partial<RustReadingsResult> = {}): RustReadingsResult {
  return { lang: "ja", hiragana: "わたし", romaji: "watashi", pinyin: null, segments: [{ text: "私", reading: "わたし" }], ...over };
}

/** A promise the test resolves by hand. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

function harness(over: Partial<ReadingsAnnotatorDeps> = {}, s: ReadingsState = settings()) {
  const published: ReadingsEvent[] = [];
  const requests: RustReadingsRequest[] = [];
  const deps: ReadingsAnnotatorDeps = {
    annotate: async (r) => {
      requests.push(r);
      return result({ lang: r.lang });
    },
    publish: (e) => published.push(e),
    settings: () => s,
    japaneseReady: () => true,
    now: () => 0,
    ...over,
  };
  const annotator = new ReadingsAnnotator(deps);
  return { annotator, published, requests, settings: s };
}

describe("readings language", () => {
  it("uses the line's lang for Japanese and Chinese only", () => {
    expect(readingsLanguage("ja", "東京")).toBe("ja");
    expect(readingsLanguage("ja-JP", "東京")).toBe("ja");
    expect(readingsLanguage("zh", "你好")).toBe("zh");
    expect(readingsLanguage("zh-Hant", "你好")).toBe("zh");
    expect(readingsLanguage("yue", "你好")).toBe("zh");
    expect(readingsLanguage("cmn", "你好")).toBe("zh");
    expect(readingsLanguage("en", "ひらがな")).toBeNull();
    expect(readingsLanguage("ko", "한국어")).toBeNull();
  });

  it("guesses Japanese from kana but never guesses Han-only text", () => {
    expect(readingsLanguage(undefined, "今日はいい天気")).toBe("ja");
    expect(readingsLanguage(undefined, "カラオケ")).toBe("ja");
    expect(readingsLanguage(undefined, "東京大学")).toBeNull();
    expect(readingsLanguage(undefined, "你好世界")).toBeNull();
    expect(readingsLanguage(undefined, "hello")).toBeNull();
  });
});

describe("source selection", () => {
  it("selects a topic and the topics below a parent", () => {
    expect(sourceSelected(["text.stt"], "text.stt")).toBe(true);
    expect(sourceSelected(["others"], "others.translation")).toBe(true);
    expect(sourceSelected(["text"], "text.stt")).toBe(true);
    expect(sourceSelected(["text.stt"], "text.translation")).toBe(false);
    expect(sourceSelected(["text"], "textfield")).toBe(false);
    expect(sourceSelected([], "text.stt")).toBe(false);
  });
});

describe("readings request", () => {
  it("asks only for the forms that are on", () => {
    expect(readingsRequest("ja", "東京", settings({ romaji: false }))).toEqual({
      text: "東京",
      lang: "ja",
      hiragana: true,
      romaji: false,
      pinyin: false,
    });
    expect(readingsRequest("ja", "東京", settings({ hiragana: false, romaji: false }))).toBeNull();
    expect(readingsRequest("zh", "你好", settings({ pinyinStyle: "numbers" }))).toEqual({
      text: "你好",
      lang: "zh",
      hiragana: false,
      romaji: false,
      pinyin: true,
      pinyinStyle: "numbers",
    });
    expect(readingsRequest("zh", "你好", settings({ pinyin: false }))).toBeNull();
  });
});

describe("utterance ids and events", () => {
  it("keys lines by segmentId, then by numeric id", () => {
    expect(utteranceIdOf({ segmentId: "mic-ab12:3:7", id: 5 })).toBe("mic-ab12:3:7");
    expect(utteranceIdOf({ id: 42 })).toBe("42");
    expect(utteranceIdOf({ segmentId: "bad id!" })).toBeNull();
    expect(utteranceIdOf({})).toBeNull();
  });

  it("builds a valid readings.stt payload", () => {
    const event = toReadingsEvent({
      utteranceId: "mic-1:1:1",
      revision: 3,
      source: TextEventSource.stt,
      value: "私です",
      result: result({ segments: [{ text: "私", reading: "わたし" }, { text: "です", reading: null }] }),
    });
    expect(event).toEqual({
      utteranceId: "mic-1:1:1",
      revision: 3,
      source: "text.stt",
      lang: "ja",
      value: "私です",
      hiragana: "わたし",
      romaji: "watashi",
      segments: [{ text: "私", reading: "わたし" }, { text: "です" }],
    });
    expect(ReadingsEventSchema.safeParse(event).success).toBe(true);
    // Only the requested forms; no empty segment list.
    const zh = toReadingsEvent({
      utteranceId: "7",
      source: OthersEventSource.stt,
      value: "你好",
      result: { lang: "zh", hiragana: null, romaji: null, pinyin: "nǐ hǎo", segments: [] },
    });
    expect(zh).toEqual({ utteranceId: "7", source: "others.stt", lang: "zh", value: "你好", pinyin: "nǐ hǎo" });
  });

  it("refuses payloads over the schema limits", () => {
    expect(
      toReadingsEvent({ utteranceId: "1", source: TextEventSource.stt, value: "x", result: result({ romaji: "a".repeat(4_001) }) }),
    ).toBeNull();
  });

  it("recognizes dictionary errors by the catalog id", () => {
    expect(isDictionaryError(`The Japanese dictionary (${JAPANESE_DICTIONARY_ID}) is not installed`)).toBe(true);
    expect(isDictionaryError("Readings are available for Japanese and Chinese, not 'en'")).toBe(false);
  });
});

describe("ReadingsAnnotator", () => {
  it("publishes readings of a selected source's final", async () => {
    const h = harness();
    h.annotator.push(TextEventSource.stt, line("私は学生です", { lang: "ja", segmentId: "mic-x:1:1", revision: 2 }));
    await flush();
    expect(h.requests).toEqual([{ text: "私は学生です", lang: "ja", hiragana: true, romaji: true, pinyin: false }]);
    expect(h.published).toHaveLength(1);
    expect(h.published[0]).toMatchObject({ utteranceId: "mic-x:1:1", revision: 2, source: "text.stt", value: "私は学生です" });
    expect(h.annotator.state.published).toBe(1);
    expect(h.annotator.state.last?.utteranceId).toBe("mic-x:1:1");
  });

  it("ignores lines when off, from other sources, interims unless asked, and other languages", async () => {
    const off = harness({}, settings({ enable: false }));
    off.annotator.push(TextEventSource.stt, line("東京", { lang: "ja" }));
    const h = harness();
    h.annotator.push(TextEventSource.translation, line("東京", { lang: "ja" }));
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja", type: TextEventType.interim }));
    h.annotator.push(TextEventSource.stt, line("hello", { lang: "en" }));
    h.annotator.push(TextEventSource.stt, line("   ", { lang: "ja" }));
    h.annotator.push(TextEventSource.stt, line("あ".repeat(MAX_READINGS_TEXT + 1), { lang: "ja" }));
    h.annotator.push(TextEventSource.stt, undefined);
    await flush();
    expect(off.requests).toHaveLength(0);
    expect(h.requests).toHaveLength(0);

    const interim = harness({}, settings({ interim: true }));
    interim.annotator.push(TextEventSource.stt, line("東京", { lang: "ja", type: TextEventType.interim }));
    await flush();
    expect(interim.requests).toHaveLength(1);
  });

  it("skips Japanese while the dictionary is missing but still reads Chinese", async () => {
    const h = harness({ japaneseReady: () => false }, settings({ sources: [TextEventSource.stt] }));
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja" }));
    h.annotator.push(TextEventSource.stt, line("你好", { lang: "zh", id: 9 }));
    await flush();
    expect(h.requests.map((r) => r.lang)).toEqual(["zh"]);
    expect(h.published[0].utteranceId).toBe("9");
  });

  it("coalesces revisions of a waiting line and drops results a newer revision replaces", async () => {
    const first = deferred<RustReadingsResult>();
    const calls: RustReadingsRequest[] = [];
    const h = harness(
      {
        annotate: (r) => {
          calls.push(r);
          return calls.length === 1 ? first.promise : Promise.resolve(result({ hiragana: r.text }));
        },
      },
      settings({ interim: true }),
    );
    const seg = { lang: "ja", segmentId: "mic-y:1:4" } as const;
    h.annotator.push(TextEventSource.stt, line("き", { ...seg, revision: 1, type: TextEventType.interim }));
    // While the first is running, two newer revisions arrive: only the last one is annotated.
    h.annotator.push(TextEventSource.stt, line("きょう", { ...seg, revision: 2, type: TextEventType.interim }));
    h.annotator.push(TextEventSource.stt, line("今日は", { ...seg, revision: 3 }));
    expect(h.annotator.pendingCount).toBe(1);
    first.resolve(result({ hiragana: "き" }));
    await flush();
    await flush();
    expect(calls.map((c) => c.text)).toEqual(["き", "今日は"]);
    // The first result was superseded while it ran: only revision 3 is published.
    expect(h.published.map((e) => [e.revision, e.hiragana])).toEqual([[3, "今日は"]]);
  });

  it("keeps lines of different sources and segments apart", async () => {
    const h = harness({}, settings({ sources: [TextEventSource.stt, OthersEventSource.stt] }));
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja", segmentId: "a:1:1" }));
    h.annotator.push(OthersEventSource.stt, line("東京", { lang: "ja", segmentId: "a:1:1" }));
    h.annotator.push(TextEventSource.stt, line("大阪", { lang: "ja", segmentId: "a:1:2" }));
    await flush();
    await flush();
    expect(h.published.map((e) => `${e.source}|${e.utteranceId}`)).toEqual(["text.stt|a:1:1", "others.stt|a:1:1", "text.stt|a:1:2"]);
  });

  it("bounds the queue, dropping the oldest interims first", async () => {
    const gate = deferred<RustReadingsResult>();
    let n = 0;
    const h = harness({ annotate: () => (n++ === 0 ? gate.promise : Promise.resolve(result())) }, settings({ interim: true }));
    h.annotator.push(TextEventSource.stt, line("最初", { lang: "ja", segmentId: "s:0:0" }));
    for (let i = 1; i <= MAX_PENDING + 4; i++) {
      const interim = i % 2 === 0;
      h.annotator.push(TextEventSource.stt, line(`行${i}`, { lang: "ja", segmentId: `s:1:${i}`, type: interim ? TextEventType.interim : TextEventType.final }));
    }
    expect(h.annotator.pendingCount).toBe(MAX_PENDING);
    gate.resolve(result());
    for (let i = 0; i < 40; i++) await flush();
    const ids = h.published.map((e) => e.utteranceId);
    // Every final survived; interims made room.
    for (let i = 1; i <= MAX_PENDING + 4; i += 2) expect(ids).toContain(`s:1:${i}`);
    expect(ids.length).toBe(MAX_PENDING + 1);
  });

  it("reports errors once per message and flags the dictionary", async () => {
    const log = vi.fn();
    const onDictionaryError = vi.fn();
    const message = `The Japanese dictionary (${JAPANESE_DICTIONARY_ID}) is not installed`;
    const h = harness({ annotate: () => Promise.reject(message), log, onDictionaryError });
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja", segmentId: "e:1:1" }));
    h.annotator.push(TextEventSource.stt, line("大阪", { lang: "ja", segmentId: "e:1:2" }));
    await flush();
    await flush();
    expect(h.published).toHaveLength(0);
    expect(h.annotator.state.lastError).toBe(message);
    expect(onDictionaryError).toHaveBeenCalledTimes(2);
    expect(log).toHaveBeenCalledTimes(1);
  });

  it("drops results that arrive after readings were switched off", async () => {
    const gate = deferred<RustReadingsResult>();
    const h = harness({ annotate: () => gate.promise });
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja", segmentId: "c:1:1" }));
    h.annotator.clear();
    gate.resolve(result());
    await flush();
    expect(h.published).toHaveLength(0);
  });

  it("gives lines without ids their own fallback key", async () => {
    let next = 0;
    const h = harness({ newId: () => `fallback-${++next}` });
    h.annotator.push(TextEventSource.stt, line("東京", { lang: "ja" }));
    h.annotator.push(TextEventSource.stt, line("大阪", { lang: "ja" }));
    await flush();
    await flush();
    expect(h.published.map((e) => e.utteranceId)).toEqual(["fallback-1", "fallback-2"]);
  });
});
