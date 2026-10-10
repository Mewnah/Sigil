import type { STT_State } from "@/core/services/stt/schema";
import {
  GOVERNOR_REASONS,
  isTwoPassPlan,
  planRoles,
  requiredModelIds,
  routedLanguages,
  specOnGpu,
  type PlanRole,
} from "@/core/services/stt/router";
import { buildLocalRouteRequest, localSpeechState } from "@/core/services/stt/services/local";
import { WHISPER_LANGUAGES } from "@/core/services/stt/whisperModels";
import { entryTitle } from "@/core/services/models/format";
import { formatBytes } from "@/core/services/diagnostics/format";
import { ServiceNetworkState, type RustCatalogEntry, type RustSpeechEngineSpec, type RustSpeechRoutePlan } from "@/types";
import { invoke, isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiCpuLine, RiInformationLine, RiRefreshLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { InputCheckbox, InputRustAudioInput, InputSelect, InputText } from "../components/input";
import { Advanced, useLanguageName } from "./common";
import { VoiceDetection } from "./VoiceDetection";
import { MusicGateRequirement } from "@/core/services/readings/ReadingsSection";

/**
 * Panel of the `local` backend (P2-E): Sigil's in-process speech pipeline with automatic model routing. Language (auto
 * or explicit), hardware tier ("Best for my PC"), two-pass, Smart Turn, hotwords, the route preview
 * (`plugin:speech|preview_route`; while running, the plan in use) and `<ModelRequirement>` for consent and downloads.
 */

/** Whisper-only languages already covered under another code by the router (`no` → `nb`, `tl` → `fil`). */
const WHISPER_ALIASED = new Set(["no", "tl"]);

/** How often the preview is refreshed while the panel is open (GPU memory and the voice changer change). */
const PREVIEW_REFRESH_MS = 30_000;

/** Languages offered: every language the router has a row for, plus the ones Whisper handles well. */
function languageCodes(): string[] {
  const codes = new Set(routedLanguages());
  for (const code of WHISPER_LANGUAGES) if (!WHISPER_ALIASED.has(code)) codes.add(code);
  return [...codes];
}

interface PreviewState {
  plan: RustSpeechRoutePlan | null;
  error: string | null;
  loading: boolean;
}

/** The plan Rust would run now for these settings (refreshed on changes, installs, and periodically). */
function useRoutePreview(language: string, tier: string, twoPass: boolean, interim: boolean, refreshKey: string): PreviewState & { refresh(): void } {
  const [state, setState] = useState<PreviewState>({ plan: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  const tauri = isTauri();

  useEffect(() => {
    if (!tauri) return;
    const timer = setInterval(() => setTick((n) => n + 1), PREVIEW_REFRESH_MS);
    return () => clearInterval(timer);
  }, [tauri]);

  useEffect(() => {
    if (!tauri) return;
    let cancelled = false;
    setState((s) => ({ ...s, loading: true }));
    // Debounced: typing in other fields re-renders; only these inputs matter.
    const timer = setTimeout(() => {
      const request = buildLocalRouteRequest({ language, tier: tier as STT_State["local"]["tier"], twoPass, interim });
      invoke<RustSpeechRoutePlan>("plugin:speech|preview_route", { request })
        .then((plan) => {
          if (!cancelled) setState({ plan, error: null, loading: false });
        })
        .catch((e: unknown) => {
          if (!cancelled) setState({ plan: null, error: e instanceof Error ? e.message : String(e), loading: false });
        });
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tauri, language, tier, twoPass, interim, refreshKey, tick]);

  return { ...state, refresh: () => setTick((n) => n + 1) };
}

const ROLE_KEYS: Record<PlanRole, string> = {
  streaming: "stt.local_role_streaming",
  final: "stt.local_role_final",
  fallback: "stt.local_role_fallback",
  languageId: "stt.local_role_language_id",
};

/** The engines of a plan, what each does and where it runs, and why the plan looks the way it does. */
const PlanView: FC<{ plan: RustSpeechRoutePlan; catalog: readonly RustCatalogEntry[] }> = ({ plan, catalog }) => {
  const { t } = useTranslation();
  const title = (spec: RustSpeechEngineSpec) => {
    const id = spec.model ?? "";
    const entry = catalog.find((e) => e.id === id);
    return entry ? entryTitle(entry) : id;
  };
  const roles = planRoles(plan);
  const summary = isTwoPassPlan(plan)
    ? t("stt.local_route_summary_two_pass")
    : plan.streaming
      ? t("stt.local_route_summary_stream")
      : plan.final
        ? t("stt.local_route_summary_finals")
        : t("stt.local_route_none");
  const noteText = (code: string) =>
    GOVERNOR_REASONS.includes(code)
      ? t(`models.gpu.rec_reason_${code}`, { defaultValue: code })
      : t(`stt.route_note_${code}`, { defaultValue: code });
  // "auto_language" repeats what the language field says; the rest explain the choice.
  const notes = plan.notes.filter((n) => n !== "auto_language");

  return <div className="flex flex-col gap-1.5 text-xs leading-snug">
    <span className="opacity-80">{summary}</span>
    {roles.length > 0 && <ul className="flex flex-col gap-1">
      {roles.map(({ role, spec }) => (
        <li key={role} className="flex items-start gap-1.5">
          <RiCpuLine className="shrink-0 mt-0.5 opacity-60" aria-hidden />
          <span className="break-words">
            <span className="opacity-70">{t(ROLE_KEYS[role])}: </span>
            <span className="font-semibold" title={spec.model ?? undefined}>{title(spec)}</span>
            <span className="opacity-60"> · {specOnGpu(spec) ? t("stt.local_device_gpu") : t("stt.local_device_cpu")}</span>
          </span>
        </li>
      ))}
    </ul>}
    {plan.vramMb > 0 && <span className="opacity-60">{t("stt.local_route_vram", { size: formatBytes(plan.vramMb * 1024 * 1024) })}</span>}
    {notes.length > 0 && <ul className="flex flex-col gap-1">
      {notes.map((code) => (
        <li key={code} className="flex items-start gap-1.5 opacity-70">
          <RiInformationLine className="shrink-0 mt-0.5" aria-hidden />
          <span className="break-words">{noteText(code)}</span>
        </li>
      ))}
    </ul>}
  </div>;
};

export const Local: FC = () => {
  const { t } = useTranslation();
  const languageName = useLanguageName();
  const tauri = isTauri();
  useModelsLoaded();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.local);
  const sttState = useSnapshot(window.ApiServer.stt.serviceState);
  const running = useSnapshot(localSpeechState);
  const modelsSnap = useSnapshot(window.ApiServer.models.state);
  const catalog = modelsSnap.catalog as RustCatalogEntry[];
  const up = <K extends keyof STT_State["local"]>(key: K, v: STT_State["local"][K]) =>
    window.ApiServer.patchService("stt", s => { s.data.local[key] = v; });

  // Installing or removing a model changes what the plan is missing (and, for GPU packs, the plan itself).
  const installedKey = useMemo(
    () => Object.values(modelsSnap.states).filter((s) => s?.state === "installed").map((s) => s.id).sort().join("|"),
    [modelsSnap.states],
  );
  const preview = useRoutePreview(pr.language, pr.tier, pr.twoPass, pr.interim !== false, installedKey);

  const languageOptions = useMemo(() => {
    const codes = languageCodes();
    const options = codes
      .map((code) => ({ value: code, label: languageName(code) }))
      .sort((a, b) => a.label.localeCompare(b.label));
    if (pr.language && pr.language !== "auto" && !codes.includes(pr.language))
      options.unshift({ value: pr.language, label: languageName(pr.language) });
    return [{ value: "auto", label: t("stt.local_language_auto") }, ...options];
  }, [languageName, pr.language, t]);

  const tierOptions = [
    { value: "auto", label: t("stt.local_tier_auto") },
    { value: "cpu", label: t("stt.local_tier_cpu") },
    { value: "gpu", label: t("stt.local_tier_gpu") },
  ];

  const live = sttState.status !== ServiceNetworkState.disconnected && running.plan !== null;
  const shownPlan = (live ? running.plan : preview.plan) as RustSpeechRoutePlan | null;
  const requirementIds = useMemo(
    () => (preview.plan ? requiredModelIds(preview.plan, pr.smartTurn) : []),
    [preview.plan, pr.smartTurn],
  );

  return <>
    <Inspector.SubHeader>{t("stt.local_title")}</Inspector.SubHeader>
    <Inspector.Description>{t("stt.local_desc")}</Inspector.Description>

    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />

    <InputSelect label="common.field_language" value={pr.language || "auto"} options={languageOptions} onValueChange={e => up("language", e)} />
    <Inspector.Description>{t("stt.local_language_desc")}</Inspector.Description>

    <InputSelect label="stt.local_tier" value={pr.tier} options={tierOptions} onValueChange={e => up("tier", e as STT_State["local"]["tier"])} />
    <Inspector.Description>{t("stt.local_tier_desc")}</Inspector.Description>

    <InputCheckbox label="stt.local_two_pass" value={pr.twoPass} onChange={e => up("twoPass", e)} />
    <Inspector.Description>{t("stt.local_two_pass_desc")}</Inspector.Description>

    <InputCheckbox label="stt.field_enable_interim_results" value={pr.interim !== false} onChange={e => up("interim", e)} />
    <Inspector.Description>{t("stt.local_interim_desc")}</Inspector.Description>

    <InputCheckbox label="stt.local_smart_turn" value={pr.smartTurn} onChange={e => up("smartTurn", e)} />
    <Inspector.Description>{t("stt.local_smart_turn_desc")}</Inspector.Description>

    <InputText label="stt.local_hotwords" value={pr.hotwords} placeholder={t("stt.local_hotwords_placeholder")} onChange={e => up("hotwords", e.target.value)} />
    <Inspector.Description>{t("stt.local_hotwords_desc")}</Inspector.Description>

    <div className="rounded-lg border border-base-content/10 bg-base-200/40 p-2.5 flex flex-col gap-2" aria-live="polite">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold flex-1 min-w-0">{live ? t("stt.local_route_running") : t("stt.local_route_title")}</span>
        {tauri && !live && (
          <button
            type="button"
            className={classNames("btn btn-ghost btn-xs gap-1", preview.loading && "loading")}
            onClick={() => preview.refresh()}
            disabled={preview.loading}
            title={t("stt.local_route_refresh")}
          >
            {!preview.loading && <RiRefreshLine aria-hidden />}
            <span className="sr-only">{t("stt.local_route_refresh")}</span>
          </button>
        )}
      </div>
      {!tauri ? (
        <span className="text-xs opacity-70">{t("stt.local_route_desktop_only")}</span>
      ) : shownPlan ? (
        <PlanView plan={shownPlan} catalog={catalog} />
      ) : preview.error ? (
        <span className="text-xs text-warning break-words">{t("stt.local_route_error", { error: preview.error })}</span>
      ) : (
        <span className="text-xs opacity-70">{t("stt.local_route_loading")}</span>
      )}
      {live && running.restarts > 0 && (
        <span className="text-xs opacity-60">{t("stt.local_route_restarts", { n: running.restarts })}</span>
      )}
    </div>

    {tauri && requirementIds.length > 0 && (
      <ModelRequirement ids={requirementIds} reason={t("stt.local_models_reason")} />
    )}

    <Advanced title={t("stt.whisper_accuracy_title")}>
      <InputCheckbox label="stt.whisper_hallucination_filter" value={pr.hallucinationFilter} onChange={e => up("hallucinationFilter", e)} />
      <Inspector.Description>{t("stt.whisper_hallucination_filter_desc")}</Inspector.Description>
      {/* P6-L A/B (scaffolded by P6-S0): FireRedVAD music / singing gate, off by default. */}
      <InputCheckbox label="stt.local_music_gate" value={pr.musicGate} onChange={e => up("musicGate", e)} />
      <Inspector.Description>{t("stt.local_music_gate_desc")}</Inspector.Description>
      <MusicGateRequirement />
    </Advanced>
    <VoiceDetection />
  </>;
};
