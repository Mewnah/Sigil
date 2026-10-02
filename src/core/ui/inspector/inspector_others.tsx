import { othersRouteRequest, planNote } from "@/core/services/others/driver";
import { OTHERS_DEFAULT_SOURCE, type Others_State } from "@/core/services/others/schema";
import { othersTranslationPair } from "@/core/services/others/translation";
import {
  isProcessSource,
  LIST_LOOPBACK_PROCESSES,
  parsePlan,
  processExe,
  processSourceId,
  sourceLabel,
  type OthersProcessList,
} from "@/core/services/others/source";
import { entryTitle } from "@/core/services/models/format";
import { isTwoPassPlan, planRoles, requiredModelIds, routedLanguages, specOnGpu } from "@/core/services/stt/router";
import { WHISPER_LANGUAGES } from "@/core/services/stt/whisperModels";
import { TRANSLATION_LANGUAGES } from "@/core/services/translation/languages";
import { ServiceNetworkState, type RustCatalogEntry, type RustSpeechRoutePlan } from "@/types";
import { buildRustSourceOptions } from "@/utils/rustAudioInput";
import { invoke, isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiCpuLine, RiRefreshLine, RiTeamFill } from "react-icons/ri";
import { useSnapshot } from "valtio";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { ModelRequirement, useModelsLoaded } from "./components/ModelRequirement";
import { InputCheckbox, InputSelect, type InputSelectOption } from "./components/input";
import { Advanced, useLanguageName } from "./stt/common";

/**
 * Other players (P6-A): what other players say (the game's own sound, or an output device), as captions on
 * `others.stt` and translated into the user's language on `others.translation`. Capture and recognition settings
 * apply on the next start; translation and live text apply at once.
 */

/** Whisper-only languages already covered under another code by the router (`no` → `nb`, `tl` → `fil`). */
const WHISPER_ALIASED = new Set(["no", "tl"]);

function languageCodes(): string[] {
  const codes = new Set(routedLanguages());
  for (const code of WHISPER_LANGUAGES) if (!WHISPER_ALIASED.has(code)) codes.add(code);
  return [...codes];
}

/** The program list and source plan (refreshed on demand and every few seconds while the panel is open). */
function usePrograms(source: string, fallback: boolean): { list: OthersProcessList | null; refresh(): void; loading: boolean } {
  const tauri = isTauri();
  const [list, setList] = useState<OthersProcessList | null>(null);
  const [tick, setTick] = useState(0);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!tauri) return;
    const timer = setInterval(() => setTick((n) => n + 1), 10_000);
    return () => clearInterval(timer);
  }, [tauri]);
  useEffect(() => {
    if (!tauri) return;
    let cancelled = false;
    setLoading(true);
    invoke<OthersProcessList>(LIST_LOOPBACK_PROCESSES, { source, fallback })
      .then((l) => !cancelled && setList(l))
      .catch(() => !cancelled && setList(null))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [tauri, source, fallback, tick]);
  return { list, loading, refresh: () => setTick((n) => n + 1) };
}

/** The route Rust would run for the Others settings now. */
function useRoutePreview(s: Pick<Others_State, "language" | "tier" | "twoPass">, refreshKey: string): { plan: RustSpeechRoutePlan | null; error: string | null } {
  const tauri = isTauri();
  const [state, setState] = useState<{ plan: RustSpeechRoutePlan | null; error: string | null }>({ plan: null, error: null });
  useEffect(() => {
    if (!tauri) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      invoke<RustSpeechRoutePlan>("plugin:speech|preview_route", { request: othersRouteRequest(s) })
        .then((plan) => !cancelled && setState({ plan, error: null }))
        .catch((e: unknown) => !cancelled && setState({ plan: null, error: e instanceof Error ? e.message : String(e) }));
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tauri, s.language, s.tier, s.twoPass, refreshKey]);
  return state;
}

const PlanSummary: FC<{ plan: RustSpeechRoutePlan; catalog: readonly RustCatalogEntry[] }> = ({ plan, catalog }) => {
  const { t } = useTranslation();
  const title = (id: string) => {
    const entry = catalog.find((e) => e.id === id);
    return entry ? entryTitle(entry) : id;
  };
  const roles = planRoles(plan);
  return <div className="flex flex-col gap-1 text-xs leading-snug">
    <span className="opacity-80">
      {isTwoPassPlan(plan) ? t("others.route_summary_two_pass") : plan.streaming ? t("others.route_summary_stream") : plan.final ? t("others.route_summary_finals") : t("others.route_none")}
    </span>
    {roles.length > 0 && <ul className="flex flex-col gap-1">
      {roles.map(({ role, spec }) => (
        <li key={role} className="flex items-start gap-1.5">
          <RiCpuLine className="shrink-0 mt-0.5 opacity-60" aria-hidden />
          <span className="break-words">
            <span className="font-semibold" title={spec.model ?? undefined}>{title(spec.model ?? "")}</span>
            <span className="opacity-60"> · {specOnGpu(spec) ? t("stt.local_device_gpu") : t("stt.local_device_cpu")}</span>
          </span>
        </li>
      ))}
    </ul>}
  </div>;
};

const Inspector_Others: FC = () => {
  const { t, i18n } = useTranslation();
  const tauri = isTauri();
  const languageName = useLanguageName();
  useModelsLoaded();
  const pr = useSnapshot(window.ApiServer.state.services.others.data);
  const state = useSnapshot(window.ApiServer.others.serviceState);
  const tr = useSnapshot(window.ApiServer.others.translationState);
  const devices = useSnapshot(window.ApiServer.audioDevices.state);
  const modelsSnap = useSnapshot(window.ApiServer.models.state);
  const catalog = modelsSnap.catalog as RustCatalogEntry[];
  const up = <K extends keyof Others_State>(key: K, v: Others_State[K]) =>
    window.ApiServer.patchService("others", (s) => { s.data[key] = v; });

  useEffect(() => {
    void window.ApiServer.audioDevices.ensureLoaded();
  }, []);

  const stopped = state.status === ServiceNetworkState.disconnected;
  const source = pr.source || OTHERS_DEFAULT_SOURCE;
  const programs = usePrograms(source, pr.fallback);
  const supported = programs.list?.supported ?? true;
  const processes = programs.list?.processes ?? [];

  const installedKey = useMemo(
    () => Object.values(modelsSnap.states).filter((s) => s?.state === "installed").map((s) => s.id).sort().join("|"),
    [modelsSnap.states],
  );
  const preview = useRoutePreview(pr, installedKey);
  const requirementIds = useMemo(() => (preview.plan ? requiredModelIds(preview.plan, false) : []), [preview.plan]);
  const shownPlan = (!stopped && state.routePlan ? state.routePlan : preview.plan) as RustSpeechRoutePlan | null;

  const labels = useMemo(() => ({
    program: (exe: string) => t("others.source_program", { exe }),
    loopbackDefault: t("others.source_default_output"),
    loopback: (name: string) => t("others.source_output", { name }),
    device: (name: string) => name,
  }), [t]);

  const sourceOptions = useMemo(() => {
    const programOptions: InputSelectOption[] = [];
    const seen = new Set<string>();
    const add = (value: string, label: string) => {
      const key = value.toLowerCase();
      if (seen.has(key)) return;
      seen.add(key);
      programOptions.push({ value, label });
    };
    const vrchat = processes.find((p) => processExe(OTHERS_DEFAULT_SOURCE).toLowerCase() === p.exe.toLowerCase());
    add(OTHERS_DEFAULT_SOURCE, vrchat ? t("others.source_vrchat") : t("others.source_program_not_running", { exe: t("others.source_vrchat") }));
    for (const p of processes) {
      add(processSourceId(p.exe), p.title ? t("others.source_program_title", { exe: p.exe, title: p.title }) : t("others.source_program", { exe: p.exe }));
    }
    if (isProcessSource(source)) add(source, t("others.source_program_not_running", { exe: processExe(source) }));
    const { loopback } = buildRustSourceOptions(
      { inputs: devices.inputs, outputs: devices.outputs },
      isProcessSource(source) ? "" : source,
      true,
      {
        systemDefault: t("common.device_system_default"),
        loopbackDefault: labels.loopbackDefault,
        loopback: labels.loopback,
        unavailable: (name) => t("audio_devices.device_unavailable", { name }),
      },
    );
    const outputOptions: InputSelectOption[] = [...loopback];
    const listed = [...programOptions, ...outputOptions].some((o) => o.value === source);
    if (!listed) outputOptions.push({ value: source, label: sourceLabel(source, labels) });
    return [
      { label: t("others.source_group_programs"), options: programOptions },
      { label: t("others.source_group_outputs"), options: outputOptions },
    ];
  }, [processes, devices.inputs, devices.outputs, source, labels, t]);

  const languageOptions = useMemo(() => {
    const codes = languageCodes();
    const options = codes.map((code) => ({ value: code, label: languageName(code) })).sort((a, b) => a.label.localeCompare(b.label));
    if (pr.language && pr.language !== "auto" && !codes.includes(pr.language)) options.unshift({ value: pr.language, label: languageName(pr.language) });
    return [{ value: "auto", label: t("stt.local_language_auto") }, ...options];
  }, [languageName, pr.language, t]);

  const tierOptions = [
    { value: "auto", label: t("stt.local_tier_auto") },
    { value: "cpu", label: t("stt.local_tier_cpu") },
    { value: "gpu", label: t("stt.local_tier_gpu") },
  ];

  // "" = the user's own language: shown with the language it resolves to now.
  const mainTranslation = useSnapshot(window.ApiServer.state.services.translation.data);
  const sttLocal = useSnapshot(window.ApiServer.state.services.stt.data.local);
  const myLanguage = othersTranslationPair({
    othersLanguage: pr.language,
    translateTo: "",
    mainSource: mainTranslation.sourceLanguage,
    sttLanguage: sttLocal?.language,
    appLanguage: i18n.language,
  }).to;
  const targetOptions = useMemo(() => {
    const options = TRANSLATION_LANGUAGES.map((l) => ({ value: l.code, label: languageName(l.code) })).sort((a, b) => a.label.localeCompare(b.label));
    if (pr.translateTo && !TRANSLATION_LANGUAGES.some((l) => l.code === pr.translateTo)) options.unshift({ value: pr.translateTo, label: languageName(pr.translateTo) });
    return [{ value: "mine", label: t("others.translate_to_mine", { language: languageName(myLanguage) }) }, ...options];
  }, [languageName, myLanguage, pr.translateTo, t]);

  const listeningTo = state.source ? sourceLabel(state.source, labels) : "";
  // What a start would capture now (the program is running, or the fallback / wait it would use).
  const plannedNow = parsePlan(programs.list?.plan);
  const previewNote = plannedNow?.running
    ? t("others.preview_running", { program: processExe(plannedNow.source) })
    : planNote(plannedNow).text;
  const processChosen = isProcessSource(source);

  return <Inspector.Body>
    <Inspector.Header><RiTeamFill /> {t("others.title")}</Inspector.Header>
    <Inspector.Content>
      <Inspector.Description>{t("others.description")}</Inspector.Description>

      <Inspector.SubHeader>{t("others.section_capture")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={stopped}>
        <div className="flex items-end gap-1">
          <div className="flex-1 min-w-0">
            <InputSelect label="others.field_source" value={source} options={sourceOptions} onValueChange={(v) => up("source", v)} />
          </div>
          {tauri && <button
            type="button"
            className={classNames("btn btn-ghost btn-xs mb-1", programs.loading && "loading")}
            onClick={() => programs.refresh()}
            title={t("others.refresh_programs")}
            aria-label={t("others.refresh_programs")}
          >
            {!programs.loading && <RiRefreshLine aria-hidden />}
          </button>}
        </div>
        <Inspector.Description>{processChosen ? t("others.source_hint_program") : t("others.source_hint_output")}</Inspector.Description>
        {processChosen && !supported && (
          // With the fallback off a start fails instead of capturing the default output: say so.
          <span className="text-xs text-warning">
            {pr.fallback ? t("others.source_unsupported", { build: programs.list?.windowsBuild ?? "?" }) : t("others.error_unsupported")}
          </span>
        )}
        {processChosen && supported && stopped && previewNote && (
          <span className="text-xs text-base-content/70">{previewNote}</span>
        )}
        {processChosen && <>
          <InputCheckbox label="others.fallback" value={pr.fallback} onChange={(v) => up("fallback", v)} />
          <Inspector.Description>{t("others.fallback_desc")}</Inspector.Description>
        </>}
      </Inspector.Deactivatable>

      <Inspector.SubHeader>{t("others.section_recognition")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={stopped}>
        <InputSelect label="common.field_language" value={pr.language || "auto"} options={languageOptions} onValueChange={(v) => up("language", v)} />
        <Inspector.Description>{t("others.language_desc")}</Inspector.Description>
        <InputSelect label="stt.local_tier" value={pr.tier} options={tierOptions} onValueChange={(v) => up("tier", v as Others_State["tier"])} />
        <Inspector.Description>{t("others.tier_desc")}</Inspector.Description>
        <Advanced title={t("others.advanced")}>
          <InputCheckbox label="stt.local_two_pass" value={pr.twoPass} onChange={(v) => up("twoPass", v)} />
          <Inspector.Description>{t("others.two_pass_desc")}</Inspector.Description>
          <InputCheckbox label="stt.local_music_gate" value={pr.musicGate} onChange={(v) => up("musicGate", v)} />
          <Inspector.Description>{t("stt.local_music_gate_desc")}</Inspector.Description>
        </Advanced>
      </Inspector.Deactivatable>

      <div className="rounded-lg border border-base-content/10 bg-base-200/40 p-2.5 flex flex-col gap-2" aria-live="polite">
        <span className="text-xs font-semibold">{stopped ? t("others.route_title") : t("others.route_running")}</span>
        {!tauri ? (
          <span className="text-xs opacity-70">{t("others.error_desktop_only")}</span>
        ) : shownPlan ? (
          <PlanSummary plan={shownPlan} catalog={catalog} />
        ) : preview.error ? (
          <span className="text-xs text-warning break-words">{t("stt.local_route_error", { error: preview.error })}</span>
        ) : (
          <span className="text-xs opacity-70">{t("stt.local_route_loading")}</span>
        )}
      </div>
      {tauri && requirementIds.length > 0 && <ModelRequirement ids={requirementIds} reason={t("others.models_reason")} hideWhenReady />}

      <div className="pt-2 flex flex-col gap-2">
        <ServiceButton status={state.status} onStart={() => window.ApiServer.others.start()} onStop={() => window.ApiServer.others.stop()} onPending={() => window.ApiServer.others.stop()} />
        <div role="status" aria-live="polite" className="flex flex-col gap-1 text-xs">
          {listeningTo && <span className="text-base-content/70">{t("others.listening_to", { source: listeningTo })}</span>}
          {state.statusText && <span className={state.statusLevel === "warning" ? "text-warning" : "text-base-content/70"}>{state.statusText}</span>}
          {state.progress !== null && <progress className="progress progress-primary w-full" value={state.progress} max={100} />}
          {stopped && state.error && <span className="text-error break-words">{state.error}</span>}
        </div>
      </div>

      <Inspector.SubHeader>{t("others.section_translation")}</Inspector.SubHeader>
      <InputCheckbox label="others.translate" value={pr.translate} onChange={(v) => up("translate", v)} />
      <Inspector.Description>{t("others.translate_desc")}</Inspector.Description>
      {pr.translate && <>
        <InputSelect
          label="others.translate_to"
          value={pr.translateTo || "mine"}
          options={targetOptions}
          onValueChange={(v) => up("translateTo", v === "mine" ? "" : v)}
        />
        <div role="status" aria-live="polite" className="flex flex-col gap-1 text-xs">
          {tr.statusText && <span className={tr.statusLevel === "warning" ? "text-warning" : "text-base-content/70"}>{tr.statusText}</span>}
          {tr.progress !== null && <progress className="progress progress-primary w-full" value={tr.progress} max={100} />}
          {tr.status === ServiceNetworkState.disconnected && tr.error && !stopped && <span className="text-error break-words">{tr.error}</span>}
          {tr.missing.length > 0 && <ModelRequirement ids={tr.missing as readonly string[]} reason={t("transl.status_missing")} hideWhenReady />}
        </div>
      </>}

      <InputCheckbox label="others.interim" value={pr.interim} onChange={(v) => up("interim", v)} />
      <Inspector.Description>{t("others.interim_desc")}</Inspector.Description>

      <Inspector.SubHeader>{t("others.where_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("others.where_desc")}</Inspector.Description>

      <div className="h-2" />
      <InputCheckbox label="common.field_auto_start" value={pr.autoStart} onChange={(v) => up("autoStart", v)} />
    </Inspector.Content>
  </Inspector.Body>;
};

export default Inspector_Others;
