import { describe, expect, it } from "vitest";
import { mapFinishReason, parseChatPayload, parseCompletionPayload, SseEvent, SseParser } from "../stream";

function parseAll(chunks: string[]): SseEvent[] {
  const p = new SseParser();
  const out: SseEvent[] = [];
  for (const c of chunks) out.push(...p.push(c));
  out.push(...p.end());
  return out;
}

describe("SseParser", () => {
  const body = 'data: {"a":1}\n\n: keep-alive\n\nevent: ping\ndata: x\n\ndata: [DONE]\n\n';

  it("parses events, ignoring comments", () => {
    expect(parseAll([body])).toEqual([
      { event: "message", data: '{"a":1}' },
      { event: "ping", data: "x" },
      { event: "message", data: "[DONE]" },
    ]);
  });

  it("gives the same events whatever the chunk boundaries (every split point)", () => {
    const expected = parseAll([body]);
    for (let i = 1; i < body.length; i++) {
      expect(parseAll([body.slice(0, i), body.slice(i)])).toEqual(expected);
    }
    expect(parseAll(body.split(""))).toEqual(expected);
  });

  it("handles CRLF and CR line endings, including a CRLF split across chunks", () => {
    const crlf = 'data: {"a":1}\r\n\r\ndata: b\r\n\r\n';
    expect(parseAll([crlf])).toEqual([
      { event: "message", data: '{"a":1}' },
      { event: "message", data: "b" },
    ]);
    expect(parseAll(['data: a\r', '\n\r', "\n"])).toEqual([{ event: "message", data: "a" }]);
    expect(parseAll(["data: a\r\rdata: b\r\r"])).toEqual([
      { event: "message", data: "a" },
      { event: "message", data: "b" },
    ]);
  });

  it("joins multi-line data and strips only one leading space", () => {
    expect(parseAll(["data:  two\ndata:x\n\n"])).toEqual([{ event: "message", data: " two\nx" }]);
  });

  it("flushes a last event without the closing blank line", () => {
    expect(parseAll(['data: {"z":0}'])).toEqual([{ event: "message", data: '{"z":0}' }]);
  });

  it("reports llama-server's error lines as error events", () => {
    expect(parseAll(['error: {"code":500,"message":"boom"}\n\n'])).toEqual([
      { event: "error", data: '{"code":500,"message":"boom"}' },
    ]);
  });
});

describe("payload parsing", () => {
  it("reads chat deltas, whole messages, finish reasons, usage and inline errors", () => {
    expect(parseChatPayload({ choices: [{ delta: { content: "Hi" } }], model: "m" })).toMatchObject({ content: "Hi", model: "m" });
    expect(parseChatPayload({ choices: [{ message: { content: [{ type: "text", text: "A" }] }, finish_reason: "length" }] })).toMatchObject({
      content: "A",
      finishReason: "length",
    });
    expect(parseChatPayload({ choices: [], usage: { prompt_tokens: 5, completion_tokens: 7 } }).usage).toEqual({ promptTokens: 5, completionTokens: 7 });
    expect(parseChatPayload({ error: { message: "quota", code: 429 } }).error).toEqual({ message: "quota", status: 429 });
    // OpenRouter mid-stream failure: error next to a choice.
    expect(parseChatPayload({ error: { message: "upstream", code: 502 }, choices: [{ delta: { content: "" }, finish_reason: "error" }] }).error).toEqual({
      message: "upstream",
      status: 502,
    });
    expect(parseChatPayload({ error: "model not found" }).error).toEqual({ message: "model not found" });
    // Reasoning deltas are not content.
    expect(parseChatPayload({ choices: [{ delta: { reasoning_content: "hmm" } }] }).content).toBe("");
    expect(parseChatPayload("nope").content).toBe("");
  });

  it("reads OpenAI legacy completions and llama-server native completions", () => {
    expect(parseCompletionPayload({ choices: [{ text: "Bonjour", finish_reason: "stop" }] })).toMatchObject({ content: "Bonjour", finishReason: "stop" });
    expect(parseCompletionPayload({ content: "Hal", stop: false })).toEqual({ content: "Hal", finishReason: undefined, model: undefined, usage: undefined });
    expect(parseCompletionPayload({ content: "", stop: true, stopped_limit: true, tokens_predicted: 64, tokens_evaluated: 10 })).toMatchObject({
      finishReason: "length",
      usage: { promptTokens: 10, completionTokens: 64 },
    });
    expect(parseCompletionPayload({ content: "", stop: true, stopped_eos: true }).finishReason).toBe("stop");
    expect(parseCompletionPayload({ content: "", stop_type: "limit" }).finishReason).toBe("length");
    expect(parseCompletionPayload({ error: { message: "bad", code: 400 } }).error).toEqual({ message: "bad", status: 400 });
  });

  it("maps provider finish reasons", () => {
    expect(mapFinishReason("stop")).toBe("stop");
    expect(mapFinishReason("length")).toBe("length");
    expect(mapFinishReason("max_tokens")).toBe("length");
    expect(mapFinishReason("content_filter")).toBe("content_filter");
    expect(mapFinishReason("error")).toBe("unknown");
    expect(mapFinishReason(null)).toBeUndefined();
  });
});
