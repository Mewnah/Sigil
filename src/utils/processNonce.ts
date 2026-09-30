/**
 * Per-process random tags for ids that would otherwise restart in every Sigil process.
 *
 * STT segment ids (`<stream>:<epoch>:<n>`), translation line ids and the AI transform's line `id` all count up from
 * the same start in every process. A restarted host, or a linked host relaying the same `text.*` topics, would reuse
 * ids that another process already used for a different line, so captions could merge into, give up or block each
 * other's lines. These values are made once per module load.
 */

/** Random [0-9a-z]{6} tag for string ids (put it in the id's stream part, never after the first ':'). */
export const PROCESS_NONCE: string = makeNonce(6);

/**
 * Random first value (an integer below 2^40) for numeric ids that count up from it in this process (the AI
 * transform's line `id`). Consumers only compare these ids for equality, never order them.
 */
export const PROCESS_ID_BASE: number = randomIdBase();

export function makeNonce(length: number): string {
  const bytes = new Uint8Array(length);
  globalThis.crypto.getRandomValues(bytes);
  let out = "";
  for (const b of bytes) out += (b % 36).toString(36);
  return out;
}

function randomIdBase(): number {
  const bytes = new Uint8Array(5);
  globalThis.crypto.getRandomValues(bytes);
  let n = 0;
  for (const b of bytes) n = n * 256 + b;
  return n;
}
