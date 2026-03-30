import { zSafe } from "@/utils";
import z from "zod";
import { TransformRectSchema, UnionElementStateSchema } from "../elements/schema";
import { FileStateSchema } from "../services/files/schema";
import { SceneStateSchema } from "../services/scenes/schema";

/** Remove deprecated canvas elements so older saves still load. */
function stripLegacyAudioVizElements(input: unknown): unknown {
  if (input == null || typeof input !== "object") return input;
  const doc = input as Record<string, unknown>;
  const elements = doc.elements;
  if (!elements || typeof elements !== "object" || Array.isArray(elements)) return input;

  const removeIds = new Set<string>();
  for (const [id, el] of Object.entries(elements)) {
    if (el && typeof el === "object" && (el as { type?: string }).type === "audioViz") {
      removeIds.add(id);
    }
  }
  if (removeIds.size === 0) return input;

  const nextElements = { ...(elements as Record<string, unknown>) };
  removeIds.forEach((id) => {
    delete nextElements[id];
  });

  const ids = doc.elementsIds;
  const nextIds = Array.isArray(ids) ? ids.filter((id: string) => !removeIds.has(id)) : ids;

  return { ...doc, elements: nextElements, elementsIds: nextIds };
}

export const TransitionSchema = z.object({
  type: zSafe(z.string(), "none"),
  duration: zSafe(z.number(), 300),
}).default({});

export type TransitionState = z.infer<typeof TransitionSchema>;

export const DocumentSchema = z.preprocess(
  stripLegacyAudioVizElements,
  z.object({
    author: zSafe(z.string(), ""),
    canvas: zSafe(TransformRectSchema, { x: 0, y: 0, w: 500, h: 400, r: 0 }),
    snapToGrid: zSafe(z.boolean(), true),
    activeScene: zSafe(z.string(), "main"),
    transition: TransitionSchema,
    scenes: zSafe(z.record(z.string(), SceneStateSchema), {
      main: { id: "main", name: "Default scene" },
    }),
    filesMeta: zSafe(FileStateSchema.array(), []),
    elementsIds: zSafe(z.string().array(), []),
    elements: zSafe(z.record(z.string(), UnionElementStateSchema), {}),
  })
);
export type DocumentState = z.infer<typeof DocumentSchema>;