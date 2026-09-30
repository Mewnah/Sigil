import { describe, expect, it } from "vitest";
import { TRANSFORM_REASONING_EFFORTS } from "@/core/services/transform/schema";
import { LLM_DEFAULT_URLS, LLM_PROVIDERS, LLM_REASONING_EFFORTS, LlmError } from "../types";

// P4-S0 contract tests for the shared LLM client types.

describe("LLM client contract", () => {
  it("keeps the persisted provider and reasoning values", () => {
    expect([...LLM_PROVIDERS]).toEqual(["bundled", "gemini", "openrouter", "openai", "custom"]);
    // The transform settings store the same reasoning values (P4-A moves transform onto this client).
    expect([...LLM_REASONING_EFFORTS]).toEqual([...TRANSFORM_REASONING_EFFORTS]);
    expect(LLM_DEFAULT_URLS.gemini).toBe("https://generativelanguage.googleapis.com/v1beta/openai");
    for (const url of Object.values(LLM_DEFAULT_URLS)) expect(url.endsWith("/")).toBe(false);
  });

  it("classifies retryable LLM errors by kind unless told otherwise", () => {
    expect(new LlmError("rate_limited", "429").retryable).toBe(true);
    expect(new LlmError("server", "502", { status: 502 }).retryable).toBe(true);
    expect(new LlmError("unauthorized", "401", { status: 401 }).retryable).toBe(false);
    expect(new LlmError("bad_request", "400").retryable).toBe(false);
    expect(new LlmError("network", "reset", { retryable: false }).retryable).toBe(false);
    const e = new LlmError("timeout", "slow");
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe("LlmError");
    expect(e.kind).toBe("timeout");
  });
});
