/**
 * Output and error-text sanitizing shared by every LLM consumer (transform, translation, the llama-server host):
 * - `cleanLlmOutput` strips reasoning (`<think>` blocks, including template-opened ones), stray special tokens and
 *   an echoed input fence, so only the reply is published;
 * - `visibleStreamText` is the same for a reply that is still streaming (a half-received tag is hidden too);
 * - `redactKeys` masks API-key-like tokens (and the endpoint's own key) in text shown to users or written to logs;
 * - `contentText` reads a message `content` that some servers send as an array of parts.
 *
 * Pure functions: no window, no Tauri (unit-tested in `__tests__/sanitize.test.ts`).
 */

/**
 * Chat-template control tokens that leak into replies when a server renders special tokens (llama.cpp raw
 * completions, misconfigured templates): ChatML / Llama 3 / Hunyuan `<|...|>`, Gemma turn markers, and the
 * sentencepiece `<s>` / `</s>` / `<eos>` / `<bos>` / `<pad>` / `<unk>`. They never belong in a caption.
 */
const SPECIAL_TOKEN_RE = /<\|[^<>|\s]{1,64}\|>|<\/?(?:start_of_turn|end_of_turn)>|<\/?s>|<(?:eos|bos|pad|unk)>/gi;
/** Text a ChatML-style template can leave after a stripped `<|im_start|>`: the role name on its own line. */
const LEADING_ROLE_RE = /^\s*(?:assistant|model)\s*\n/i;

/**
 * Removes reasoning and stray special tokens so only the reply is published. Local reasoning models (Qwen3,
 * DeepSeek-R1 distills, ...) put their thinking inline in `<think>` blocks; a reply cut off while still thinking has
 * no usable text at all. Chat templates that open the block in the prompt (DeepSeek-R1 distills, Qwen3 thinking
 * templates served without a reasoning parser, e.g. vLLM without --reasoning-parser or llama.cpp
 * --reasoning-format none) return the reasoning with only a closing `</think>`: everything up to the last unmatched
 * one is reasoning too.
 */
export function cleanLlmOutput(raw: string): string {
  let text = raw.replace(/<think>[\s\S]*?<\/think>/gi, "");
  // Any </think> left has no opening tag before it: the block was opened by the template.
  let reasoningEnd = -1;
  for (const m of text.matchAll(/<\/think>/gi)) reasoningEnd = (m.index ?? 0) + m[0].length;
  if (reasoningEnd >= 0) text = text.slice(reasoningEnd);
  const unclosed = text.search(/<think>/i);
  if (unclosed >= 0) text = text.slice(0, unclosed);
  // `search` ignores the regex's global state.
  const hadSpecialTokens = text.search(SPECIAL_TOKEN_RE) >= 0;
  text = text.replace(SPECIAL_TOKEN_RE, "");
  if (hadSpecialTokens) text = text.replace(LEADING_ROLE_RE, "");
  // Models sometimes echo the input fence (transform's `<text>` data fence).
  text = text.replace(/^\s*<text>/i, "").replace(/<\/text>\s*$/i, "");
  return text.trim();
}

/** Tags whose half-received start must not flash in a streaming caption. */
const PARTIAL_TAGS = ["<think>", "</think>", "<text>", "</text>", "<|", "<end_of_turn>", "<start_of_turn>", "</s>", "<eos>"];

/**
 * Length of a trailing fragment of `raw` that could still become one of the tags `cleanLlmOutput` removes
 * (`"...Hello <thi"` → 4), so a streaming caption never shows it for a frame.
 */
function partialTagLength(raw: string): number {
  const lastOpen = raw.lastIndexOf("<");
  if (lastOpen < 0) return 0;
  const tail = raw.slice(lastOpen).toLowerCase();
  if (tail.includes(">")) return 0;
  // `<|im_end` style tokens: anything after `<|` without a closing `|>` yet.
  if (tail.startsWith("<|") && !/\s/.test(tail) && tail.length <= 66) return tail.length;
  return PARTIAL_TAGS.some((tag) => tag.length > tail.length && tag.startsWith(tail)) ? tail.length : 0;
}

/**
 * Sanitized text of a reply that is still streaming: `cleanLlmOutput` of what arrived so far, where an unclosed
 * `<think>` hides everything after it and a half-received tag at the end is not shown yet.
 *
 * Limitation: reasoning whose `<think>` was opened by the chat template (no opening tag in the reply) is shown
 * until its `</think>` arrives, then disappears; configure the server's reasoning parser for such models.
 */
export function visibleStreamText(raw: string): string {
  const cut = partialTagLength(raw);
  return cleanLlmOutput(cut > 0 ? raw.slice(0, raw.length - cut) : raw);
}

/** Text of a message `content`, which some OpenAI-compatible servers send as an array of parts. */
export function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : typeof part?.text === "string" ? part.text : ""))
      .join("");
  }
  return "";
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Masks API-key-like tokens in provider error text before it reaches toasts and the file log (OpenAI's 401 message,
 * for one, echoes part of the key). `secrets` are masked wherever they appear (the endpoint's own key, whatever its
 * format, e.g. the bundled server's random key); values shorter than 6 characters are ignored.
 */
export function redactKeys(text: string, secrets: readonly string[] = []): string {
  let out = text;
  for (const secret of secrets) {
    const s = secret?.trim();
    if (s && s.length >= 6) out = out.replace(new RegExp(escapeRegExp(s), "g"), "[redacted]");
  }
  return out
    .replace(/\b(sk|rk|pk)-[A-Za-z0-9_*.-]{6,}/g, "$1-[redacted]")
    .replace(/\bAIza[0-9A-Za-z_-]{20,}/g, "[redacted]")
    .replace(/\bgsk_[A-Za-z0-9]{10,}/g, "gsk_[redacted]")
    .replace(/\b(Bearer)\s+[A-Za-z0-9._~+/=*-]{8,}/gi, "$1 [redacted]")
    .replace(/\b(x-goog-api-key|api[-_]?key|DeepL-Auth-Key)(["']?\s*[:=]\s*["']?)[A-Za-z0-9._~+/=*:-]{8,}/gi, "$1$2[redacted]");
}
