import { Services } from "@/services-registry";
import type { InspectorTabPath } from "@/types";

/**
 * Embedded panels: a few services live inside another service's inspector instead of having a left-menu entry of
 * their own. Rewrite (`Services.transform`) is the second tab of the Speech to Text inspector. Its service, state,
 * pubsub topics and ids are unchanged; only where it opens changed.
 */

/** `InspectorTabPath.value` of the STT inspector that opens its Rewrite tab. */
export const STT_REWRITE_SECTION = "rewrite";

/**
 * Resolves a requested inspector path to the one the store keeps: `{ tab: transform }` (the status-bar card and any
 * other `changeTab` caller) opens Speech to Text on its Rewrite tab. Every other path passes through untouched.
 */
export function resolveInspectorTab(v?: InspectorTabPath): InspectorTabPath | undefined {
  return v?.tab === Services.transform ? { tab: Services.stt, value: STT_REWRITE_SECTION } : v;
}
