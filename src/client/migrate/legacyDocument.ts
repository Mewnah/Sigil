import { DocumentSchema, type DocumentState } from "@/client/schema";
import { FileStateSchema } from "@/client/services/files/schema";
import { TextEventSource } from "@/types";

const VALID_TEXT_SOURCES = new Set<string>(Object.values(TextEventSource));

const DEFAULT_CANVAS = { x: 0, y: 0, w: 500, h: 400, r: 0 };
const DEFAULT_TRANSITION = { type: "none" as const, duration: 300 };
const DEFAULT_SCENES = {
  main: { id: "main", name: "Default scene" },
};

function cloneJson<T>(v: T): T {
  return JSON.parse(JSON.stringify(v)) as T;
}

function filterElementsByType(elements: unknown): { filtered: Record<string, unknown>; removed: number } {
  if (!elements || typeof elements !== "object" || Array.isArray(elements)) {
    return { filtered: {}, removed: 0 };
  }
  const src = elements as Record<string, unknown>;
  const filtered: Record<string, unknown> = {};
  let removed = 0;
  for (const [id, el] of Object.entries(src)) {
    if (!el || typeof el !== "object") {
      removed++;
      continue;
    }
    const t = (el as { type?: string }).type;
    if (t === "text" || t === "image") filtered[id] = cloneJson(el);
    else removed++;
  }
  return { filtered, removed };
}

function syncElementIds(originalIds: unknown, elementKeys: Set<string>): string[] {
  const ordered: string[] = [];
  if (Array.isArray(originalIds)) {
    for (const id of originalIds) {
      if (typeof id === "string" && elementKeys.has(id)) ordered.push(id);
    }
  }
  for (const id of elementKeys) {
    if (!ordered.includes(id)) ordered.push(id);
  }
  return ordered;
}

function sanitizeFilesMeta(raw: unknown): unknown[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((item) => FileStateSchema.safeParse(item).success);
}

function ensureSceneAnimations(elements: Record<string, unknown>): boolean {
  let touched = false;
  const defaultAnim = {
    enter: { type: "none", duration: 300, delay: 0, ease: "easeOut" },
    exit: { type: "none", duration: 300, delay: 0, ease: "easeOut" },
  };
  for (const el of Object.values(elements)) {
    if (!el || typeof el !== "object") continue;
    const scenes = (el as { scenes?: Record<string, unknown> }).scenes;
    if (!scenes || typeof scenes !== "object") continue;
    for (const scene of Object.values(scenes)) {
      if (!scene || typeof scene !== "object") continue;
      const s = scene as Record<string, unknown>;
      if (s.animation === undefined || s.animation === null) {
        s.animation = defaultAnim;
        touched = true;
      }
    }
  }
  return touched;
}

function coerceTextSources(elements: Record<string, unknown>): boolean {
  let touched = false;
  for (const el of Object.values(elements)) {
    if (!el || typeof el !== "object" || (el as { type?: string }).type !== "text") continue;
    const scenes = (el as { scenes?: Record<string, Record<string, unknown>> }).scenes;
    if (!scenes) continue;
    for (const scene of Object.values(scenes)) {
      if (!scene?.data || typeof scene.data !== "object") continue;
      const data = scene.data as Record<string, unknown>;
      const sm = data.sourceMain;
      if (typeof sm === "string" && !VALID_TEXT_SOURCES.has(sm)) {
        data.sourceMain = TextEventSource.stt;
        touched = true;
      }
    }
  }
  return touched;
}

function mergeScenes(raw: unknown): Record<string, unknown> {
  const base = cloneJson(DEFAULT_SCENES);
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return base;
  return { ...base, ...(raw as Record<string, unknown>) };
}

function pickActiveScene(active: unknown, sceneKeys: string[]): string {
  if (typeof active === "string" && sceneKeys.includes(active)) return active;
  if (sceneKeys.includes("main")) return "main";
  return sceneKeys[0] ?? "main";
}

export type MigratedDocument =
  | {
      ok: true;
      doc: DocumentState;
      notes: string[];
      /** When {@link migrateLegacyJsonDocument} was called with `fileBinaries`, aligned to `doc.filesMeta` indices. */
      alignedFileBinaries?: Uint8Array[];
    }
  | { ok: false; error: string };

/**
 * Normalizes JSON from Sigil, Curses, or close forks, then validates with {@link DocumentSchema}.
 * Curses exports omitted `snapToGrid`, `transition`, per-scene `animation`, and newer text sources.
 *
 * Pass `fileBinaries` (same order as `filesMeta` in the template) when importing a Yjs template (e.g. `.cursestmp`)
 * so dropped invalid file rows stay in sync with embedded asset blobs.
 */
export function migrateLegacyJsonDocument(input: unknown, fileBinaries?: Uint8Array[]): MigratedDocument {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    return { ok: false, error: "File is not a JSON object." };
  }

  const raw = cloneJson(input) as Record<string, unknown>;
  const notes: string[] = [];

  const { filtered: elements, removed } = filterElementsByType(raw.elements);
  if (removed > 0) {
    notes.push(`Removed ${removed} unsupported or invalid element(s) (e.g. legacy types).`);
  }

  const keys = new Set(Object.keys(elements));
  raw.elements = elements;
  raw.elementsIds = syncElementIds(raw.elementsIds, keys);

  const hadSnap = typeof raw.snapToGrid === "boolean";
  if (!hadSnap) {
    raw.snapToGrid = true;
    notes.push("Applied default snap-to-grid.");
  }

  const hadTransition =
    raw.transition &&
    typeof raw.transition === "object" &&
    !Array.isArray(raw.transition) &&
    typeof (raw.transition as { type?: unknown }).type === "string";
  if (!hadTransition) {
    raw.transition = { ...DEFAULT_TRANSITION };
    notes.push("Applied default scene transition settings.");
  } else {
    const tr = raw.transition as Record<string, unknown>;
    raw.transition = {
      type: typeof tr.type === "string" ? tr.type : "none",
      duration: typeof tr.duration === "number" && !Number.isNaN(tr.duration) ? tr.duration : 300,
    };
  }

  raw.canvas =
    raw.canvas && typeof raw.canvas === "object" && !Array.isArray(raw.canvas)
      ? { ...DEFAULT_CANVAS, ...(raw.canvas as object) }
      : { ...DEFAULT_CANVAS };

  raw.scenes = mergeScenes(raw.scenes);
  const sceneKeyList = Object.keys(raw.scenes as object);
  raw.activeScene = pickActiveScene(raw.activeScene, sceneKeyList);

  const prevFiles = Array.isArray(raw.filesMeta) ? raw.filesMeta : [];
  let alignedFileBinaries: Uint8Array[] | undefined;

  if (fileBinaries !== undefined) {
    const nextMeta: unknown[] = [];
    const aligned: Uint8Array[] = [];
    let dropped = 0;
    for (let i = 0; i < prevFiles.length; i++) {
      if (!FileStateSchema.safeParse(prevFiles[i]).success) {
        dropped++;
        continue;
      }
      nextMeta.push(cloneJson(prevFiles[i]));
      const b = i < fileBinaries.length ? fileBinaries[i] : new Uint8Array();
      aligned.push(b instanceof Uint8Array ? new Uint8Array(b) : new Uint8Array());
    }
    raw.filesMeta = nextMeta;
    if (dropped > 0) {
      notes.push(`Dropped ${dropped} invalid file metadata entr(y/ies) and matching asset data.`);
    }
    alignedFileBinaries = aligned;
  } else {
    const prevFilesLen = prevFiles.length;
    const nextFiles = sanitizeFilesMeta(raw.filesMeta);
    raw.filesMeta = nextFiles;
    if (prevFilesLen > nextFiles.length) {
      notes.push(`Dropped ${prevFilesLen - nextFiles.length} invalid file metadata entr(y/ies).`);
    }
  }

  if (ensureSceneAnimations(elements)) {
    notes.push("Added default enter/exit animation settings where missing (Curses compatibility).");
  }

  if (coerceTextSources(elements)) {
    notes.push("Updated text source fields that are not valid in Sigil (set to Speech-to-text).");
  }

  const parsed = DocumentSchema.safeParse(raw);
  if (!parsed.success) {
    const msg = parsed.error.issues.map((e) => `${e.path.join(".")}: ${e.message}`).join("; ");
    return { ok: false, error: `Invalid project data after migration: ${msg}` };
  }

  return {
    ok: true,
    doc: parsed.data,
    notes,
    ...(alignedFileBinaries !== undefined ? { alignedFileBinaries } : {}),
  };
}
