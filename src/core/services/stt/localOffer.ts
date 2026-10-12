import type { LocalOfferKind } from "./migrate";

/**
 * The one-time prompt that offers the built-in engine (`STT_Backends.local`, "Built-in (runs on your PC)") to an
 * existing desktop profile: its numbers, its wording per engine, its buttons. Pure; `Service_STT.offerLocalBackend`
 * shows it in a native dialog.
 */

/**
 * Word errors per 100 words in Sigil's English test, as the prompt says them (percent, rounded to a whole number).
 * Source: `docs/benchmarks/SPEECH_BENCHMARKS.md`, "Browser speech recognition (Chrome and Edge Web Speech, the old
 * default), English", the `Finals, clean` column (30 utterances): Parakeet Ultra 6.1, Chrome 20.8, Edge 9.8. These are
 * the only place the prompt's numbers live; `localOffer.test.ts` reads that table and fails when they drift from it.
 */
export const LOCAL_OFFER_ENGLISH_ERRORS = { ours: 6, chrome: 21, edge: 10 } as const;

/** Size of the speech models the first Start downloads for English, as the prompt says it (about). */
export const LOCAL_OFFER_DOWNLOAD_SIZE = "1.7 GB";

/** The engines the prompt has its own wording for, plus `other` (any other engine, when the offer is asked for by hand). */
export type LocalOfferDialogKind = LocalOfferKind | "other";

export interface LocalOfferDialog {
  title: string;
  message: string;
  /** Switches to the built-in engine. */
  okLabel: string;
  /** Keeps the current engine ("Keep Chrome", "Keep Edge", "Not now"). */
  cancelLabel: string;
}

export type OfferTranslate = (key: string, options?: Record<string, unknown>) => string;

/**
 * The prompt for the engine a profile is on. The numbers are interpolated into the strings (`{{ours}}`, `{{theirs}}`,
 * `{{size}}`) so no wording repeats them.
 */
export function localOfferDialog(kind: LocalOfferDialogKind, t: OfferTranslate): LocalOfferDialog {
  const common = { ours: LOCAL_OFFER_ENGLISH_ERRORS.ours, size: LOCAL_OFFER_DOWNLOAD_SIZE };
  const title = t("stt.local_offer_title");
  const okLabel = t("stt.local_offer_yes");
  switch (kind) {
    case "chrome":
      return {
        title,
        okLabel,
        message: t("stt.local_offer_message_chrome", { ...common, theirs: LOCAL_OFFER_ENGLISH_ERRORS.chrome }),
        cancelLabel: t("stt.local_offer_keep_chrome"),
      };
    case "edge":
      return {
        title,
        okLabel,
        message: t("stt.local_offer_message_edge", { ...common, theirs: LOCAL_OFFER_ENGLISH_ERRORS.edge }),
        cancelLabel: t("stt.local_offer_keep_edge"),
      };
    case "native":
      return { title, okLabel, message: t("stt.local_offer_message", common), cancelLabel: t("stt.local_offer_no") };
    default:
      return { title, okLabel, message: t("stt.local_offer_message_other", common), cancelLabel: t("stt.local_offer_no") };
  }
}
