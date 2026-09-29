import { describe, expect, it } from "vitest";
import { azureVoiceLocale, buildAzureSsml, escapeXml, normalizeAzureVoiceName } from "../ssml";

/**
 * Minimal well-formedness check (no XML parser in the dependency tree): every `&` starts one of the
 * five predefined entities, and after removing tags no `<` or `>` remains in text content.
 */
function assertWellFormedText(xml: string) {
  const stray = xml.match(/&(?!(amp|lt|gt|quot|apos);)/g);
  expect(stray, `stray & in ${xml}`).toBeNull();
  const text = xml.replace(/<[^<>]*>/g, "");
  expect(text).not.toMatch(/[<>]/);
}

describe("escapeXml", () => {
  it("escapes the five XML special characters", () => {
    expect(escapeXml(`Tom & Jerry <3 "quotes" 'single'`)).toBe(
      "Tom &amp; Jerry &lt;3 &quot;quotes&quot; &apos;single&apos;",
    );
  });

  it("does not double-escape and leaves ordinary text alone", () => {
    expect(escapeXml("&amp;")).toBe("&amp;amp;");
    expect(escapeXml("こんにちは、世界！ 😀")).toBe("こんにちは、世界！ 😀");
  });

  it("drops characters XML 1.0 cannot contain", () => {
    expect(escapeXml("a\u0000b\u0007c\u000Bd\u001Fe")).toBe("abcde");
    expect(escapeXml("tab\tnew\nline\rok")).toBe("tab\tnew\nline\rok");
    expect(escapeXml("x￾y￿z")).toBe("xyz");
    // Unpaired surrogates are dropped, valid pairs (emoji) are kept.
    expect(escapeXml("a\uD800b")).toBe("ab");
    expect(escapeXml("a\uDC00b")).toBe("ab");
    expect(escapeXml("😀")).toBe("😀");
  });
});

describe("normalizeAzureVoiceName", () => {
  it("drops trailing commas and whitespace from old voice ids", () => {
    expect(normalizeAzureVoiceName("zh-CN-XiaoyiNeural,")).toBe("zh-CN-XiaoyiNeural");
    expect(normalizeAzureVoiceName("  en-US-JennyNeural , ")).toBe("en-US-JennyNeural");
    expect(normalizeAzureVoiceName("en-US-JennyNeural")).toBe("en-US-JennyNeural");
  });
});

describe("azureVoiceLocale", () => {
  it.each([
    ["en-US-JennyNeural", "en-US"],
    ["zh-CN-XiaoxiaoNeural", "zh-CN"],
    ["ja-JP-NanamiNeural", "ja-JP"],
    ["zh-CN-henan-YundengNeural", "zh-CN"],
    ["wuu-CN-XiaotongNeural", "wuu-CN"],
    ["sr-Latn-RS-NicholasNeural", "sr-Latn-RS"],
    ["iu-Cans-CA-SiqiniqNeural", "iu-Cans-CA"],
    ["zh-CN-XiaoyiNeural,", "zh-CN"],
  ])("%s -> %s", (voice, locale) => {
    expect(azureVoiceLocale(voice)).toBe(locale);
  });

  it("returns undefined for names without a locale", () => {
    expect(azureVoiceLocale("JennyNeural")).toBeUndefined();
    expect(azureVoiceLocale("")).toBeUndefined();
  });
});

describe("buildAzureSsml", () => {
  it("escapes text so '&' and '<' no longer break the document", () => {
    const ssml = buildAzureSsml("Fish & chips < 5 > 3", { voice: "en-US-JennyNeural" });
    expect(ssml).toContain(">Fish &amp; chips &lt; 5 &gt; 3<");
    assertWellFormedText(ssml);
  });

  it("uses the http SSML namespace, the mstts namespace and the voice's language", () => {
    const ssml = buildAzureSsml("你好", { voice: "zh-CN-XiaoxiaoNeural" });
    expect(ssml).toMatch(/^<speak version="1\.0" /);
    expect(ssml).toContain('xmlns="http://www.w3.org/2001/10/synthesis"');
    expect(ssml).not.toContain('xmlns="https://');
    expect(ssml).toContain('xmlns:mstts="https://www.w3.org/2001/mstts"');
    expect(ssml).toContain('xml:lang="zh-CN"');
    expect(ssml).toContain('<voice name="zh-CN-XiaoxiaoNeural">你好</voice>');
  });

  it("falls back to en-US when the voice has no locale prefix", () => {
    expect(buildAzureSsml("hi", { voice: "CustomVoice" })).toContain('xml:lang="en-US"');
  });

  it("normalizes voice ids with a trailing comma", () => {
    const ssml = buildAzureSsml("hi", { voice: "zh-CN-YunxiaNeural," });
    expect(ssml).toContain('<voice name="zh-CN-YunxiaNeural">');
    expect(ssml).toContain('xml:lang="zh-CN"');
  });

  it("puts style and role on mstts:express-as, not on <voice>", () => {
    const ssml = buildAzureSsml("hi", { voice: "zh-CN-XiaomoNeural", style: "cheerful", role: "Girl" });
    expect(ssml).toContain('<voice name="zh-CN-XiaomoNeural"><mstts:express-as style="cheerful" role="Girl">hi</mstts:express-as></voice>');
  });

  it("omits prosody when every value is default or empty", () => {
    const ssml = buildAzureSsml("hi", {
      voice: "en-US-JennyNeural",
      volume: "default",
      rate: "",
      pitch: "default",
      range: undefined,
    });
    expect(ssml).not.toContain("<prosody");
    expect(ssml).not.toContain("express-as");
  });

  it("keeps only non-default prosody attributes, escaped", () => {
    const ssml = buildAzureSsml("hi", {
      voice: "en-US-JennyNeural",
      style: "whispering",
      volume: "loud",
      rate: "default",
      pitch: 'x-high" onload="x',
      range: "low",
    });
    expect(ssml).toContain(
      '<mstts:express-as style="whispering"><prosody volume="loud" pitch="x-high&quot; onload=&quot;x" range="low">hi</prosody></mstts:express-as>',
    );
    expect(ssml).not.toContain("rate=");
    assertWellFormedText(ssml);
  });

  it("returns an empty string when there is nothing to say or no voice", () => {
    expect(buildAzureSsml("   ", { voice: "en-US-JennyNeural" })).toBe("");
    expect(buildAzureSsml("\u0000\u0001", { voice: "en-US-JennyNeural" })).toBe("");
    expect(buildAzureSsml("hello", { voice: " , " })).toBe("");
  });

  it("speaks markup literally instead of interpreting it", () => {
    const ssml = buildAzureSsml('<break time="5s"/> hi', { voice: "en-US-JennyNeural" });
    expect(ssml).not.toContain("<break");
    expect(ssml).toContain("&lt;break time=&quot;5s&quot;/&gt; hi");
  });
});
