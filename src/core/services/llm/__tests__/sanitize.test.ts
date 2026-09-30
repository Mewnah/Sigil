import { describe, expect, it } from "vitest";
import { cleanLlmOutput, contentText, redactKeys, visibleStreamText } from "../sanitize";

describe("cleanLlmOutput", () => {
  it("strips reasoning blocks and stray special tokens", () => {
    expect(cleanLlmOutput("<think>plan the pun</think>\nhewwo!")).toBe("hewwo!");
    expect(cleanLlmOutput("<s> hi </s>")).toBe("hi");
    expect(cleanLlmOutput("<THINK>a</THINK>b<think>c</think>d")).toBe("bd");
  });

  it("returns nothing for a reply cut off while still thinking", () => {
    expect(cleanLlmOutput("<think>long thoughts that never end")).toBe("");
  });

  it("strips reasoning whose <think> was opened by the chat template (only </think> in the reply)", () => {
    // DeepSeek-R1 distill on vLLM without --reasoning-parser.
    expect(cleanLlmOutput("Okay, the user wants uwu style... </think>\n\nhewwo")).toBe("hewwo");
    expect(cleanLlmOutput("plan</THINK>answer")).toBe("answer");
    expect(cleanLlmOutput("a</think>b<think>c</think>d")).toBe("bd");
    expect(cleanLlmOutput("thinking</think> reply <think>cut off")).toBe("reply");
    expect(cleanLlmOutput("only thoughts</think>")).toBe("");
  });

  it("drops an echoed input fence", () => {
    expect(cleanLlmOutput("<text>\nhewwo owo\n</text>")).toBe("hewwo owo");
    expect(cleanLlmOutput("<think>x</think><text>hi</text>")).toBe("hi");
  });

  it("drops chat-template control tokens that leak from raw completions", () => {
    expect(cleanLlmOutput("Bonjour<|im_end|>")).toBe("Bonjour");
    expect(cleanLlmOutput("<|im_start|>assistant\nHallo<|im_end|>")).toBe("Hallo");
    expect(cleanLlmOutput("こんにちは<|eot_id|>")).toBe("こんにちは");
    expect(cleanLlmOutput("Hola<end_of_turn>")).toBe("Hola");
    expect(cleanLlmOutput("Ciao</s>")).toBe("Ciao");
    expect(cleanLlmOutput("<|hy_place▁holder▁no▁2|>你好")).toBe("你好");
    // Ordinary text with angle brackets or pipes stays.
    expect(cleanLlmOutput("a <b> c | d")).toBe("a <b> c | d");
    expect(cleanLlmOutput("I <3 you")).toBe("I <3 you");
    // "assistant" is only a leaked role name right after a stripped control token.
    expect(cleanLlmOutput("assistant\nhelps")).toBe("assistant\nhelps");
  });
});

describe("visibleStreamText", () => {
  it("never shows a half-received tag", () => {
    expect(visibleStreamText("Hello <")).toBe("Hello");
    expect(visibleStreamText("Hello <thi")).toBe("Hello");
    expect(visibleStreamText("Hello <think>")).toBe("Hello");
    expect(visibleStreamText("Hello <think>hmm")).toBe("Hello");
    expect(visibleStreamText("Hi<|im_e")).toBe("Hi");
    expect(visibleStreamText("Hi</te")).toBe("Hi");
  });

  it("shows the reply once the reasoning is closed", () => {
    const stream = ["<th", "ink>plan", "ning</th", "ink>Hel", "lo wor", "ld"];
    let raw = "";
    const seen: string[] = [];
    for (const piece of stream) {
      raw += piece;
      seen.push(visibleStreamText(raw));
    }
    expect(seen).toEqual(["", "", "", "Hel", "Hello wor", "Hello world"]);
  });

  it("keeps text that only looks like a tag start once it cannot become one", () => {
    expect(visibleStreamText("I <3")).toBe("I <3");
    expect(visibleStreamText("x < y")).toBe("x < y");
  });
});

describe("contentText", () => {
  it("reads string and array message contents", () => {
    expect(contentText("x")).toBe("x");
    expect(contentText([{ type: "text", text: "a" }, "b", { type: "image" }])).toBe("ab");
    expect(contentText(null)).toBe("");
    expect(contentText(undefined)).toBe("");
  });
});

describe("redactKeys", () => {
  it("masks key-like tokens in provider errors", () => {
    expect(redactKeys("401 Incorrect API key provided: sk-proj-abc*****************wxyz.")).not.toContain("wxyz");
    expect(redactKeys("key AIzaSyA1234567890abcdefghijklmnop rejected")).toBe("key [redacted] rejected");
    expect(redactKeys("Authorization: Bearer abcdefgh12345678")).toBe("Authorization: Bearer [redacted]");
    expect(redactKeys("bad gsk_abcdefghijklmnop")).toBe("bad gsk_[redacted]");
    expect(redactKeys('{"x-goog-api-key": "abcdefgh12345678"}')).toBe('{"x-goog-api-key": "[redacted]"}');
    expect(redactKeys("api_key=abcdefgh12345678&x=1")).toBe("api_key=[redacted]&x=1");
    expect(redactKeys("model not found")).toBe("model not found");
  });

  it("masks the endpoint's own key whatever its format", () => {
    const serverKey = "f3a9c2e1d0b8a7f6";
    expect(redactKeys(`invalid key ${serverKey} (expected another)`, [serverKey])).toBe("invalid key [redacted] (expected another)");
    // Short or empty secrets are ignored rather than masking common words.
    expect(redactKeys("the model", ["the", ""])).toBe("the model");
    // Regex characters in the secret are literal.
    expect(redactKeys("k=a.b*c+d?e(f)", ["a.b*c+d?e(f)"])).toBe("k=[redacted]");
  });
});
