/**
 * SSML for Azure neural voices.
 *
 * Earlier code had three bugs:
 * - Text was interpolated without escaping, so one `&` (or `<`) made the document invalid and Azure
 *   cancelled the utterance.
 * - The default namespace was `https://www.w3.org/2001/10/synthesis`. SSML requires `http://`.
 * - `xml:lang` was hard-coded to `en-US`, whatever the voice's language.
 *
 * It also put `style`/`role` on `<voice>`. Azure reads speaking styles and roles only from
 * `<mstts:express-as>`.
 */

const SSML_NAMESPACE = "http://www.w3.org/2001/10/synthesis";
/** Microsoft's extension namespace; Azure documents it with `https://`. */
const MSTTS_NAMESPACE = "https://www.w3.org/2001/mstts";
const FALLBACK_LOCALE = "en-US";

/**
 * Characters XML 1.0 forbids (C0 controls other than tab/LF/CR, U+FFFE/U+FFFF, unpaired surrogates).
 * Speech-to-text and chat text can carry them, and one is enough for Azure to reject the document.
 */
const INVALID_XML_CHARS =
  /[\u0000-\u0008\u000B\u000C\u000E-\u001F￾￿]|[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

const XML_ENTITIES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&apos;",
};

/** Escape text for XML content or a quoted attribute value, dropping characters XML cannot hold. */
export function escapeXml(text: string): string {
  return text.replace(INVALID_XML_CHARS, "").replace(/[&<>"']/g, (c) => XML_ENTITIES[c]);
}

/**
 * Voice short name as Azure expects it. Some ids shipped with a trailing comma (`zh-CN-XiaoyiNeural,`),
 * and users may have that value persisted.
 */
export function normalizeAzureVoiceName(voice: string): string {
  return voice.trim().replace(/[\s,]+$/, "");
}

// language (2-3 letters), optional script (Latn, Cans, ...), region (2 letters or 3 digits).
const VOICE_LOCALE = /^([a-z]{2,3})-(?:([A-Z][a-z]{3})-)?([A-Z]{2}|\d{3})(?=-|$)/;

/**
 * BCP-47 locale of an Azure voice short name: `zh-CN-XiaoxiaoNeural` gives `zh-CN`,
 * `sr-Latn-RS-NicholasNeural` gives `sr-Latn-RS`, and `zh-CN-henan-YundengNeural` gives `zh-CN`.
 * Returns `undefined` when the name does not start with a locale.
 */
export function azureVoiceLocale(voice: string): string | undefined {
  const m = VOICE_LOCALE.exec(normalizeAzureVoiceName(voice));
  if (!m) return undefined;
  return [m[1], m[2], m[3]].filter(Boolean).join("-");
}

export type AzureSsmlOptions = {
  /** Voice short name, e.g. `en-US-JennyNeural`. */
  voice: string;
  /** Speaking style (`cheerful`, ...); empty for none. */
  style?: string;
  /** Role-play role (`Girl`, ...); empty for none. */
  role?: string;
  /** Prosody values; empty or `default` leaves the voice's default. */
  volume?: string;
  rate?: string;
  pitch?: string;
  range?: string;
};

function attrs(pairs: [string, string | undefined][]): string {
  return pairs
    .map(([name, value]) => [name, value?.trim() ?? ""] as const)
    .filter(([, value]) => value !== "" && value.toLowerCase() !== "default")
    .map(([name, value]) => ` ${name}="${escapeXml(value)}"`)
    .join("");
}

/**
 * Build the SSML document for one utterance. Returns `""` when there is nothing to speak or no
 * voice. `text` is plain text; any markup in it is spoken literally.
 */
export function buildAzureSsml(text: string, options: AzureSsmlOptions): string {
  const body = escapeXml(text).trim();
  const voice = normalizeAzureVoiceName(options.voice);
  if (!body || !voice) return "";

  const lang = azureVoiceLocale(voice) ?? FALLBACK_LOCALE;
  const prosody = attrs([
    ["volume", options.volume],
    ["rate", options.rate],
    ["pitch", options.pitch],
    ["range", options.range],
  ]);
  const expressAs = attrs([
    ["style", options.style],
    ["role", options.role],
  ]);

  let inner = prosody ? `<prosody${prosody}>${body}</prosody>` : body;
  if (expressAs) inner = `<mstts:express-as${expressAs}>${inner}</mstts:express-as>`;

  return (
    `<speak version="1.0" xmlns="${SSML_NAMESPACE}" xmlns:mstts="${MSTTS_NAMESPACE}" xml:lang="${escapeXml(lang)}">` +
    `<voice name="${escapeXml(voice)}">${inner}</voice>` +
    `</speak>`
  );
}
