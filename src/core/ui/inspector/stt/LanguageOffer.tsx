import { FC, ReactNode, useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { RiCheckLine, RiCloseLine, RiTranslate2 } from "react-icons/ri";
import { toast } from "react-toastify";
import { isTauri } from "@tauri-apps/api/core";
import { useSnapshot } from "valtio";
import type { LanguageOffer } from "@/core/services/stt/languageOffer";
import type { LockOutcome, LanguageOfferSource } from "@/core/services/stt/languageSuggestion";
import { isAskable, planSystemHint, systemLanguage } from "@/core/services/stt/languageSuggestion";
import { Services } from "@/services-registry";
import { useAppUIStore } from "../../store";
import { STT_REWRITE_SECTION } from "../../inspectorRoutes";
import { useLanguageName } from "./common";

/**
 * "Lock captions to your language?" (the rules: `core/services/stt/languageSuggestion.ts`; the state and the answers:
 * `Service_STT.languageOffer`). One card, three places:
 * - the STT panel, above the Start button (outside the part that greys out while STT runs, so Lock works live);
 * - a persistent toast while the panel is closed. A toast is not a dialog: it never takes focus and the app stays
 *   usable. Closing it with the X counts as "Keep automatic";
 * - the panel also shows it for a profile that never answered, from the Windows language (`source: "system"`).
 *
 * Every button is an explicit answer; "Detect automatically" stays one click away in the language select.
 */

const TOAST_ID = "stt-language-offer";

export interface LanguageOfferCardProps {
  source: LanguageOfferSource;
  /** Display name of the offered language. */
  language: string;
  busy: boolean;
  onLock(): void;
  onKeep(): void;
  onNever(): void;
}

/** The question: what was heard (or the Windows language), why it matters, and the three answers. */
export const LanguageOfferCard: FC<LanguageOfferCardProps> = ({ source, language, busy, onLock, onKeep, onNever }) => {
  const { t } = useTranslation();
  const question = t(source === "system" ? "stt.lang_offer_system" : "stt.lang_offer_detected", { language });
  return (
    <div role="group" aria-label={question} className="flex flex-col gap-2 text-xs leading-snug">
      <div className="flex items-start gap-1.5">
        <RiTranslate2 className="shrink-0 mt-0.5 text-primary" aria-hidden />
        <span className="font-semibold break-words">{question}</span>
      </div>
      <span className="opacity-70">{t("stt.lang_offer_why", { language })}</span>
      <div className="flex flex-wrap gap-1.5">
        <button type="button" className="btn btn-primary btn-xs" onClick={onLock} disabled={busy}>
          {busy ? t("stt.lang_offer_locking") : t("stt.lang_offer_lock", { language })}
        </button>
        <button type="button" className="btn btn-ghost btn-xs" onClick={onKeep} disabled={busy}>{t("stt.lang_offer_keep")}</button>
        <button type="button" className="btn btn-ghost btn-xs" onClick={onNever} disabled={busy}>{t("stt.lang_offer_never")}</button>
      </div>
    </div>
  );
};

const RESULT_KEYS: Record<LockOutcome, string> = {
  restarted: "stt.lang_offer_result_restarted",
  next_start: "stt.lang_offer_result_next_start",
  download: "stt.lang_offer_result_download",
};

export interface LanguageOfferResultProps {
  language: string;
  outcome: LockOutcome;
  /** Closes the line; absent inside the toast, which has its own X. */
  onClose?(): void;
}

/** What the lock did, in one line, with how it takes effect. */
export const LanguageOfferResult: FC<LanguageOfferResultProps> = ({ language, outcome, onClose }) => {
  const { t } = useTranslation();
  return (
    <div role="status" aria-live="polite" className="flex items-start gap-1.5 text-xs leading-snug">
      <RiCheckLine className="shrink-0 mt-0.5 text-success" aria-hidden />
      <span className="flex-1 min-w-0 break-words">{t(RESULT_KEYS[outcome], { language })}</span>
      {onClose && (
        <button type="button" className="btn btn-ghost btn-xs btn-square" onClick={onClose} aria-label={t("stt.lang_offer_close")} title={t("stt.lang_offer_close")}>
          <RiCloseLine aria-hidden />
        </button>
      )}
    </div>
  );
};

/** The card or the result for the current state of `Service_STT.languageOffer`; null when there is nothing to show. */
const OfferBody: FC<{ systemHint?: string | null; inToast?: boolean }> = ({ systemHint = null, inToast = false }) => {
  const controller = window.ApiServer.stt.languageOffer;
  const offer = useSnapshot(controller.view);
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  const languageName = useLanguageName();
  if (offer.result) {
    return (
      <LanguageOfferResult
        language={languageName(offer.result.language)}
        outcome={offer.result.outcome}
        onClose={inToast ? undefined : () => controller.dismissResult()}
      />
    );
  }
  // An offer made for a profile that has since left the automatic backend or language is not shown.
  const offered = offer.language !== null && isAskable(stt.backend, stt.local.language) ? offer.language : null;
  const language = offered ?? systemHint;
  if (!language) return null;
  const source: LanguageOfferSource = offered ? (offer.source ?? "detected") : "system";
  return (
    <LanguageOfferCard
      source={source}
      language={languageName(language)}
      busy={offer.busy}
      onLock={() => void controller.lock(language)}
      onKeep={() => controller.keepAutomatic(language)}
      onNever={() => controller.neverAsk(language)}
    />
  );
};

const panelClass = "rounded-lg border border-base-content/10 bg-base-200/40 p-2.5";

/**
 * The card in the STT panel. Shows an offer from this session, else (for a profile on the automatic backend that never
 * answered) the Windows language; the feedback line after a lock stays for a few seconds.
 */
export const LanguageOfferPanel: FC = () => {
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  const offer = useSnapshot(window.ApiServer.stt.languageOffer.view);
  const system = useMemo(() => systemLanguage(), []);
  if (!isTauri()) return null;
  const hint = planSystemHint({
    backend: stt.backend,
    language: stt.local.language,
    decision: stt.local.languageDecision,
    now: Date.now(),
    system,
  });
  const systemHint = hint.show ? hint.language : null;
  const offered = offer.language !== null && isAskable(stt.backend, stt.local.language);
  if (!offer.result && !offered && !systemHint) return null;
  return (
    <div className={panelClass} data-testid="language-offer">
      <OfferBody systemHint={systemHint} />
    </div>
  );
};

/** The slice of react-toastify the offer toast uses (injectable for tests). */
export interface OfferToastApi {
  isActive(id: string): boolean;
  show(content: ReactNode, options: { toastId: string; icon: false; autoClose: false; closeOnClick: false; draggable: false; onClose: () => void }): void;
  dismiss(id: string): void;
}

/** The part of `Service_STT.languageOffer` the toast answers with. */
export type OfferToastController = Pick<LanguageOffer, "view" | "keepAutomatic" | "dismissResult">;

/**
 * Keeps one persistent toast in step with whether the card should be on screen. A toast is not a dialog: it takes no
 * focus and the app stays usable. It does not close by itself or on a click (an unanswered question must not vanish);
 * closing it with its X is an answer: "Keep automatic" for an offer, a dismissed line for the feedback. Closing it
 * because the panel opened (which shows the card itself) or the question was answered elsewhere is not an answer.
 */
export function createOfferToastSync(api: OfferToastApi, controller: OfferToastController, body: () => ReactNode): (wanted: boolean) => void {
  let programmaticClose = false;
  let wantedNow = false;
  const sync = (wanted: boolean): void => {
    wantedNow = wanted;
    try {
      if (wanted) {
        // A toast taken down a moment ago is still on its way out (the library reports the close after its exit
        // animation): it is shown again once that close arrives.
        if (api.isActive(TOAST_ID)) return;
        api.show(body(), {
          toastId: TOAST_ID,
          icon: false,
          autoClose: false,
          closeOnClick: false,
          draggable: false,
          onClose: () => {
            if (programmaticClose) {
              programmaticClose = false;
              if (wantedNow) setTimeout(() => sync(wantedNow), 0);
              return;
            }
            if (controller.view.language) controller.keepAutomatic();
            else controller.dismissResult();
          },
        });
      } else if (api.isActive(TOAST_ID)) {
        programmaticClose = true;
        api.dismiss(TOAST_ID);
      }
    } catch {
      /* toasts are best effort */
    }
  };
  return sync;
}

const reactToastifyApi: OfferToastApi = {
  isActive: (id) => toast.isActive(id),
  show: (content, options) => void toast(content, options),
  dismiss: (id) => toast.dismiss(id),
};

/**
 * Shows the card as a toast when an offer (or the feedback after a lock) exists and the STT panel, which shows it
 * itself, is closed. Mounted once in the app root.
 */
export const LanguageOfferToastHost: FC = () => {
  const offer = useSnapshot(window.ApiServer.stt.languageOffer.view);
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  // Open on its Speech tab: the Rewrite tab does not show the card.
  const panelOpen = useAppUIStore((s) => s.sidebar.show && s.sidebar.tab?.tab === Services.stt && s.sidebar.tab.value !== STT_REWRITE_SECTION);
  const offered = offer.language !== null && isAskable(stt.backend, stt.local.language);
  const wanted = (offered || offer.result !== null) && !panelOpen;
  const sync = useMemo(
    () => createOfferToastSync(reactToastifyApi, window.ApiServer.stt.languageOffer, () => <div data-testid="language-offer-toast"><OfferBody inToast /></div>),
    [],
  );
  useEffect(() => sync(wanted), [sync, wanted]);
  return null;
};
