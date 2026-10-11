import type { RustMtEngineStatus, RustMtError, RustMtRequest, RustMtResponse } from "@/types";
import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { canonicalLanguage, guessLanguage, languageName, sameLanguage, toFlores } from "../languages";
import { NLLB_MODELS, Translation_State } from "../schema";
import {
  ITranslationEngine,
  TranslationEngineHost,
  TranslationError,
  TranslationRequest,
} from "../types";

/** Rejection value of every `plugin:mt-local|*` command (Rust `mt_local::MtError`). */
export function isMtLocalError(e: unknown): e is RustMtError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as { kind?: unknown }).kind === "string" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

function messageOf(e: unknown): string {
  if (isMtLocalError(e)) return e.message.trim();
  if (e instanceof Error) return e.message.trim();
  return e === undefined || e === null ? "" : String(e).trim();
}

/**
 * A user-facing `TranslationError` for a failed NLLB command. `not_installed` names the catalog ids to download
 * (`missing`, for the inspector's model requirement), `unsupported_pair` / `invalid_input` never succeed on a retry,
 * `busy` does, and a runtime failure gets the one retry finals are allowed.
 */
export function nllbError(e: unknown): TranslationError {
  const detail = messageOf(e) || i18n.t("common.error_no_details") || "no details";
  if (!isMtLocalError(e)) {
    return new TranslationError(i18n.t("transl.nllb_error_failed", { error: detail }) || detail, { retryable: true });
  }
  switch (e.kind) {
    case "not_installed":
      return new TranslationError(i18n.t("transl.nllb_error_not_installed") || detail, {
        retryable: false,
        missing: e.missing?.length ? [...e.missing] : [NLLB_MODELS[0]],
      });
    case "unsupported_pair":
      return new TranslationError(i18n.t("transl.nllb_error_pair", { error: detail }) || detail, { retryable: false });
    case "invalid_input":
    case "not_implemented":
      return new TranslationError(i18n.t("transl.nllb_error_failed", { error: detail }) || detail, { retryable: false });
    case "busy":
      return new TranslationError(i18n.t("transl.nllb_error_failed", { error: detail }) || detail, { retryable: true });
    case "cancelled":
      return new TranslationError(i18n.t("transl.nllb_error_cancelled") || detail, { retryable: false });
    default:
      return new TranslationError(i18n.t("transl.nllb_error_failed", { error: detail }) || detail, { retryable: true });
  }
}

interface NllbConfig {
  model: string;
  threads: number;
  beam: number;
}

/** Widest beam the slice allows (`Translation_Nllb_Schema.beam`, Rust `nllb::MAX_BEAM`). */
const NLLB_MAX_BEAM = 4;

/**
 * NLLB-200 engine (`nllb` backend) over `plugin:mt-local|nllb_*` (Rust `mt_local/nllb.rs`: NLLB-200 distilled on
 * Sigil's ONNX Runtime, CPU background profile). About 200 languages in any direction, fully offline; the router
 * uses it for the long tail.
 *
 * - `start` loads the model (`nllb_load`, a few seconds; the status line says so). A missing model rejects with a
 *   `TranslationError` whose `missing` names the catalog id: nothing downloads without the user's consent.
 * - `translate` sends canonical codes; Rust maps them to FLORES-200. An unknown source (`from` "") is guessed from
 *   the script here (`guessLanguage`) and, for Latin text, in Rust. Finals use the slice's beam width, interims are
 *   greedy. An aborted request is cancelled in Rust (`nllb_cancel`) at its next decoding step.
 * - `stop` unloads the model (about 1 GB of memory) only when no other engine instance still uses it: the main
 *   translation pipeline and the Other players' reverse pipeline (P6-A) can run NLLB at the same time, so the users
 *   are counted like Bergamot's (P6-I0).
 *
 * Owned by P4-D.
 */
export class Translation_NllbService implements ITranslationEngine {
  /** Engine instances that started (or are starting) the model and have not stopped: the last one to stop unloads it. */
  static readonly #users = new Set<Translation_NllbService>();

  #host: TranslationEngineHost | null = null;
  #config: NllbConfig | null = null;
  #source = "";
  #target = "";
  #stopped = false;
  #counter = 0;

  attach(host: TranslationEngineHost): void {
    this.#host = host;
  }

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const cfg = state.nllb;
    const target = canonicalLanguage(state.targetLanguage);
    if (!target) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    if (!toFlores(target)) {
      throw new TranslationError(i18n.t("transl.nllb_error_language", { language: languageName(state.targetLanguage) }), {
        retryable: false,
      });
    }
    const source = canonicalLanguage(state.sourceLanguage);
    if (state.sourceLanguage.trim() && (!source || !toFlores(source))) {
      throw new TranslationError(i18n.t("transl.nllb_error_language", { language: languageName(state.sourceLanguage) }), {
        retryable: false,
      });
    }
    const config: NllbConfig = {
      model: (NLLB_MODELS as readonly string[]).includes(cfg.model) ? cfg.model : NLLB_MODELS[0],
      threads: cfg.threads,
      beam: cfg.beam,
    };

    Translation_NllbService.#users.add(this);
    this.#host?.status({ text: i18n.t("transl.nllb_status_loading") || "", level: "info", progress: null });
    try {
      await raceWithSignal(
        invoke<RustMtEngineStatus>("plugin:mt-local|nllb_load", { model: config.model, threads: config.threads }),
        signal,
      );
    } catch (e) {
      this.#host?.status({ text: "" });
      if (signal.aborted) {
        this.#release();
        throw e;
      }
      const error = nllbError(e);
      if (error.missing.length) this.#host?.status({ text: error.message, level: "warning", missing: error.missing });
      throw error;
    }
    this.#host?.status({ text: "" });
    // Stopped (or superseded) while the model loaded: give the memory back.
    if (signal.aborted || this.#stopped) {
      this.#release();
      throw signal.reason ?? new DOMException("Aborted", "AbortError");
    }
    this.#config = config;
    this.#source = source;
    this.#target = target;
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const config = this.#config;
    if (!config) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
    const target = canonicalLanguage(request.to ?? this.#target) || this.#target;
    let source = canonicalLanguage(request.from ?? this.#source);
    if (source && sameLanguage(source, target)) return request.text;
    // Rust guesses too (Latin → English); the TS guess tells Ukrainian from Russian, Persian from Arabic, ...
    if (!source) source = guessLanguage(request.text);
    const requestId = request.requestId?.trim() || `nllb-${++this.#counter}`;
    const payload: RustMtRequest = {
      text: request.text,
      source: source && toFlores(source) ? source : "",
      target,
      kind: request.kind,
      requestId,
      model: config.model,
      // A built-in route (`routingTable.ts`) can ask for a wider beam for a final; the slice's own width is a floor.
      beam: request.kind === "final" ? Math.max(config.beam, Math.min(NLLB_MAX_BEAM, Math.trunc(request.beam ?? 1))) : 1,
      threads: config.threads,
    };
    const onAbort = () => {
      void invoke("plugin:mt-local|nllb_cancel", { requestId }).catch(() => {});
    };
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      const response = await raceWithSignal(
        invoke<RustMtResponse>("plugin:mt-local|nllb_translate", { request: payload }),
        signal,
      );
      return response.text;
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      throw nllbError(e);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  stop(): void {
    this.#stopped = true;
    this.#config = null;
    this.#release();
  }

  supports(from: string, to: string): boolean {
    if (!toFlores(to)) return false;
    return from === "" || toFlores(from) !== null;
  }

  /** Forget this instance; the last user unloads the model. */
  #release(): void {
    if (!Translation_NllbService.#users.delete(this)) return;
    if (Translation_NllbService.#users.size === 0) void invoke("plugin:mt-local|nllb_unload").catch(() => {});
  }
}
