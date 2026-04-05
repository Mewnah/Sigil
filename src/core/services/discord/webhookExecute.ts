import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import type { Discord_State } from "./schema";

/** Discord Execute Webhook `content` max length. */
export const DISCORD_CONTENT_MAX = 2000;

const CHUNK_GAP_MS = 650;

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export function chunkDiscordContent(text: string): string[] {
  if (text.length <= DISCORD_CONTENT_MAX) return [text];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += DISCORD_CONTENT_MAX) {
    out.push(text.slice(i, i + DISCORD_CONTENT_MAX));
  }
  return out;
}

/** Reject webhook names that Discord API rejects (case-insensitive). */
export function sanitizeDiscordWebhookUsername(raw: string): string {
  let s = raw.trim().slice(0, 80) || "Sigil";
  const lower = s.toLowerCase();
  if (lower.includes("clyde") || lower.includes("discord")) return "Sigil";
  return s;
}

/**
 * Accepts only https Discord incoming webhook URLs (discord.com or discordapp.com).
 */
export function normalizeDiscordWebhookUrl(raw: string): string | null {
  const t = raw.trim();
  if (!t) return null;
  const withScheme = /^https?:\/\//i.test(t) ? t : `https://${t}`;
  try {
    const u = new URL(withScheme);
    const host = u.hostname.toLowerCase();
    if (host !== "discord.com" && host !== "discordapp.com") return null;
    if (!u.pathname.includes("/webhooks/")) return null;
    return u.toString();
  } catch {
    return null;
  }
}

export function buildDiscordExecuteUrl(
  webhookBaseUrl: string,
  opts: { threadId?: string; wait?: boolean },
): string {
  const u = new URL(webhookBaseUrl);
  const tid = opts.threadId?.trim();
  if (tid) u.searchParams.set("thread_id", tid);
  if (opts.wait) u.searchParams.set("wait", "true");
  return u.toString();
}

export function buildDiscordWebhookJsonBody(
  content: string,
  data: Pick<
    Discord_State,
    "channelBotName" | "channelAvatarUrl" | "webhookThreadName"
  >,
): Record<string, unknown> {
  const username = sanitizeDiscordWebhookUsername(data.channelBotName || "Sigil");
  const body: Record<string, unknown> = {
    content: content.slice(0, DISCORD_CONTENT_MAX),
    username,
    allowed_mentions: { parse: [] as string[] },
  };
  const av = data.channelAvatarUrl?.trim();
  if (av) body.avatar_url = av;
  const threadName = data.webhookThreadName?.trim();
  if (threadName) body.thread_name = threadName.slice(0, 100);
  return body;
}

function parseRetryAfterSeconds(res: Response): number {
  const h = res.headers.get("retry-after");
  if (h) {
    const n = parseFloat(h);
    if (Number.isFinite(n)) return Math.min(Math.max(n, 0.5), 10);
  }
  return 2;
}

async function parse429BodyRetryAfter(res: Response): Promise<number | undefined> {
  try {
    const j = (await res.clone().json()) as { retry_after?: number };
    if (typeof j.retry_after === "number" && Number.isFinite(j.retry_after))
      return Math.min(Math.max(j.retry_after, 0.5), 10);
  } catch {
    /* ignore */
  }
  return undefined;
}

/**
 * POST JSON to Discord execute webhook; retries on 429 using Retry-After / body.
 */
export async function postDiscordWebhookJson(
  executeUrl: string,
  body: Record<string, unknown>,
  timeoutMs = 25_000,
  max429Attempts = 3,
): Promise<Response> {
  let last!: Response;
  for (let attempt = 0; attempt < max429Attempts; attempt++) {
    last = await fetchWithTimeout(executeUrl, {
      method: "POST",
      timeoutMs,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (last.status !== 429) return last;
    let wait = parseRetryAfterSeconds(last);
    const fromBody = await parse429BodyRetryAfter(last);
    if (fromBody !== undefined) wait = fromBody;
    await sleep(wait * 1000);
  }
  return last;
}

export async function postDiscordContentChunks(
  executeUrl: string,
  chunks: string[],
  buildBody: (chunk: string) => Record<string, unknown>,
): Promise<Response> {
  let last!: Response;
  for (let i = 0; i < chunks.length; i++) {
    if (i > 0) await sleep(CHUNK_GAP_MS);
    last = await postDiscordWebhookJson(executeUrl, buildBody(chunks[i]));
    if (!last.ok) return last;
  }
  return last;
}
