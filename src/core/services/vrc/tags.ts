/**
 * SenseVoice emotion and audio-event tags for VRChat (P6-C).
 *
 * Finals of engines that know them (SenseVoice in the local speech pipeline) carry `TextEvent.tags` such as
 * `emotion:happy` and `event:laughter` (vocabulary frozen by `speech::engine_api`). Two optional outputs:
 * - **Avatar parameters** ({@link TagParameterDriver}): two Int parameters set to the values below on every tagged
 *   final of your own speech, and back to 0 after a while, so avatar animations can react (an event works as a
 *   pulse). Values never change once published: avatars are built against them.
 * - **Chatbox emoji** ({@link decorateWithTags}): the emotion and event as an emoji in front of the chatbox text.
 *
 * Pure logic: no `window`, no Tauri.
 */
import type { TextEvent } from "@/types";

/** Int values of the emotion parameter (0 = neutral / none). Published contract: never renumber. */
export const EMOTION_VALUES: Readonly<Record<string, number>> = Object.freeze({
  neutral: 0,
  happy: 1,
  sad: 2,
  angry: 3,
  fearful: 4,
  disgusted: 5,
  surprised: 6,
});

/** Int values of the audio-event parameter (0 = none). Published contract: never renumber. */
export const EVENT_VALUES: Readonly<Record<string, number>> = Object.freeze({
  laughter: 1,
  applause: 2,
  cry: 3,
  cough: 4,
  sneeze: 5,
  breath: 6,
  bgm: 7,
});

const EMOTION_EMOJI: Readonly<Record<string, string>> = {
  happy: "😊",
  sad: "😢",
  angry: "😠",
  fearful: "😨",
  disgusted: "🤢",
  surprised: "😮",
};

const EVENT_EMOJI: Readonly<Record<string, string>> = {
  laughter: "😂",
  applause: "👏",
  cry: "😭",
  cough: "😷",
  sneeze: "🤧",
  breath: "😮‍💨",
  bgm: "🎵",
};

export interface ParsedTags {
  /** Known emotion (`happy`, ...), if any. */
  emotion?: string;
  /** Known audio event (`laughter`, ...), if any. */
  event?: string;
}

/** The first known emotion and event of `tags` (unknown kinds and values are ignored). */
export function parseTags(tags: readonly string[] | undefined | null): ParsedTags {
  const out: ParsedTags = {};
  for (const tag of tags ?? []) {
    const i = tag.indexOf(":");
    if (i <= 0) continue;
    const kind = tag.slice(0, i);
    const value = tag.slice(i + 1).toLowerCase();
    if (kind === "emotion" && out.emotion === undefined && value in EMOTION_VALUES) out.emotion = value;
    else if (kind === "event" && out.event === undefined && value in EVENT_VALUES) out.event = value;
  }
  return out;
}

/** Whether `tags` hold anything the avatar parameters or emoji show. */
export function hasKnownTags(tags: readonly string[] | undefined | null): boolean {
  const t = parseTags(tags);
  return t.emotion !== undefined || t.event !== undefined;
}

/** Emoji for `tags`: the emotion (not neutral) then the event, or "". */
export function tagEmoji(tags: readonly string[] | undefined | null): string {
  const t = parseTags(tags);
  return (t.emotion ? EMOTION_EMOJI[t.emotion] ?? "" : "") + (t.event ? EVENT_EMOJI[t.event] ?? "" : "");
}

/** `text` with the tags' emoji in front (unchanged without known tags). */
export function decorateWithTags(text: string, tags: readonly string[] | undefined | null): string {
  const emoji = tagEmoji(tags);
  return emoji ? `${emoji} ${text}` : text;
}

/**
 * A usable avatar parameter name, or null: trimmed, without OSC pattern characters or whitespace (VRChat names may
 * contain `/` and `.` for nested parameters).
 */
export function sanitizeParameterName(name: string | undefined | null): string | null {
  const n = (name ?? "").trim().replace(/^\/+|\/+$/g, "");
  if (!n || n.length > 128) return null;
  if (/[\s#*,?[\]{}]/u.test(n) || n.includes("//")) return null;
  return n;
}

/** OSC address of an avatar parameter. */
export function avatarParameterAddress(name: string): string {
  return `/avatar/parameters/${name}`;
}

export interface TagDriverClock {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const systemClock: TagDriverClock = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

/** Default time the parameters keep a tag before returning to 0 (ms). */
export const TAG_RESET_MS = 5_000;

export interface TagParameterDriverOptions {
  /** Sends one Int avatar parameter (`/avatar/parameters/<name>`). */
  send(name: string, value: number): void;
  /** Emotion parameter name ("" = not sent). */
  emotionParameter(): string;
  /** Audio-event parameter name ("" = not sent). */
  eventParameter(): string;
  /** Time until both go back to 0 (ms; 0 = they stay until the next tagged final). */
  resetMs(): number;
  clock?: TagDriverClock;
}

/** Drives the two Int avatar parameters from tagged finals; see the module comment. */
export class TagParameterDriver {
  readonly #options: TagParameterDriverOptions;
  readonly #clock: TagDriverClock;
  #timer: unknown = null;
  /** Parameters set to a non-zero value since the last reset (name → value). */
  #set = new Map<string, number>();

  constructor(options: TagParameterDriverOptions) {
    this.#options = options;
    this.#clock = options.clock ?? systemClock;
  }

  /** A final of your own speech: sets the parameters when it carries known tags. */
  onFinal(event: Pick<TextEvent, "tags"> | undefined): void {
    const tags = parseTags(event?.tags);
    if (tags.emotion === undefined && tags.event === undefined) return;
    const emotion = sanitizeParameterName(this.#options.emotionParameter());
    const audioEvent = sanitizeParameterName(this.#options.eventParameter());
    if (emotion) this.#write(emotion, tags.emotion !== undefined ? EMOTION_VALUES[tags.emotion] : 0);
    if (audioEvent && audioEvent !== emotion) this.#write(audioEvent, tags.event !== undefined ? EVENT_VALUES[tags.event] : 0);
    this.#armReset();
  }

  /** Sets every parameter this driver changed back to 0 now. */
  reset(): void {
    this.#disarm();
    const set = [...this.#set.keys()];
    this.#set.clear();
    for (const name of set) this.#options.send(name, 0);
  }

  /** Stops the reset timer without sending anything. */
  dispose(): void {
    this.#disarm();
    this.#set.clear();
  }

  #write(name: string, value: number): void {
    this.#options.send(name, value);
    if (value !== 0) this.#set.set(name, value);
    else this.#set.delete(name);
  }

  #armReset(): void {
    this.#disarm();
    const ms = this.#options.resetMs();
    if (!Number.isFinite(ms) || ms <= 0 || this.#set.size === 0) return;
    this.#timer = this.#clock.setTimeout(() => {
      this.#timer = null;
      this.reset();
    }, ms);
  }

  #disarm(): void {
    if (this.#timer !== null) {
      this.#clock.clearTimeout(this.#timer);
      this.#timer = null;
    }
  }
}

/**
 * Tags of recent finals by `segmentId`, so a translated line (whose `segmentId` is its first source segment's) can
 * show the emoji of the speech it came from.
 */
export class RecentTags {
  readonly #max: number;
  readonly #map = new Map<string, string[]>();

  constructor(max = 64) {
    this.#max = Math.max(1, max);
  }

  remember(event: Pick<TextEvent, "segmentId" | "tags"> | undefined): void {
    if (!event?.segmentId || !event.tags || event.tags.length === 0) return;
    this.#map.delete(event.segmentId);
    this.#map.set(event.segmentId, [...event.tags]);
    while (this.#map.size > this.#max) {
      const oldest = this.#map.keys().next().value;
      if (oldest === undefined) break;
      this.#map.delete(oldest);
    }
  }

  get(segmentId: string | undefined): string[] | undefined {
    return segmentId ? this.#map.get(segmentId) : undefined;
  }

  clear(): void {
    this.#map.clear();
  }
}
