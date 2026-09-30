import {
  acquireLlmServer,
  consentKey,
  ensureLlmServerHost,
  getLlmServerLogs,
  listLlmModels,
  llmServerConfigFromSettings,
  llmServerSession,
  llmServerUi,
  planLlmServer,
  setOverBudgetConsent,
  stopLlmServer,
  toLlmServerError,
  type LlmModelInfoExt,
} from "@/core/services/llm_server";
import {
  canOverride,
  CUSTOM_MODEL_OPTION,
  fileName,
  formatMb,
  groupModels,
  isModelPath,
  layersText,
  modelSelectValue,
  offersDownloads,
  parseContext,
  parseGpuLayers,
  planReasonKey,
  stateKey,
} from "@/core/services/llm_server/format";
import { formatBytes } from "@/core/services/diagnostics/format";
import { policyKey, type ResSnapshot } from "@/core/services/resources/format";
import type { RustHelperLogLine, RustLlmServerPlan } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import classNames from "classnames";
import { FC, memo, useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { InputSelect, InputText, type InputSelectOption } from "../components/input";

/**
 * Bundled llama-server panel (shown by the LLM panel when the provider is `bundled`): model choice from
 * `listLlmModels()` (automatic, the catalog, opt-in specialists or a GGUF file), runtime variant, context, GPU
 * layers and flash attention (the `llm.server*` fields of the translation settings), the governor's plan with
 * `<ModelRequirement>` for the GGUF and the runtime pack, a session-only "start anyway" for budget refusals,
 * status, test load, stop and the server log.
 *
 * Owned by P4-B. Export name frozen by P4-S0 (the LLM panel, P4-F, renders `<LlmServer />`). The settings sit in
 * the translation inspector's Deactivatable (they apply on the next start); the status and log stay usable
 * while translation runs (`pointer-events-auto`).
 */

const LOG_REFRESH_MS = 2000;
const LOG_LINES = 150;

function llmSlice() {
  return window.ApiServer.state.services.translation.data.llm;
}

const tone = {
  neutral: "border-base-content/10 bg-base-200/40",
  success: "border-success/30 bg-success/5",
  warning: "border-warning/40 bg-warning/5",
  error: "border-error/40 bg-error/5",
} as const;

const Box: FC<{ kind: keyof typeof tone; className?: string; children: ReactNode }> = ({ kind, className, children }) => (
  <div className={classNames("rounded-box border px-3 py-2 text-xs space-y-1", tone[kind], className)}>{children}</div>
);

export const LlmServer: FC = memo(() => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const ui = useSnapshot(llmServerUi);
  const session = useSnapshot(llmServerSession);
  const resources = useSnapshot(window.ApiServer.resources.state);
  const policy = policyKey(resources.snapshot as ResSnapshot | null);

  const [models, setModels] = useState<LlmModelInfoExt[]>([]);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planTick, setPlanTick] = useState(0);
  const [busy, setBusy] = useState<"test" | "stop" | null>(null);
  const [test, setTest] = useState<{ ok: boolean; seconds?: string; message?: string } | null>(null);
  const [showLogs, setShowLogs] = useState(false);
  const [logs, setLogs] = useState<RustHelperLogLine[]>([]);
  const [pickCustom, setPickCustom] = useState(() => isModelPath(pr.serverModel));
  const [customPath, setCustomPath] = useState(() => (isModelPath(pr.serverModel) ? pr.serverModel : ""));

  const config = useMemo(
    () => llmServerConfigFromSettings(pr),
    [pr.serverModel, pr.serverVariant, pr.serverContext, pr.serverGpuLayers, pr.serverFlashAttn],
  );

  useEffect(() => {
    if (tauri) void ensureLlmServerHost();
  }, [tauri]);

  const loadModels = useCallback(() => {
    listLlmModels()
      .then((m) => {
        setModels(m);
        setModelsError(null);
      })
      .catch((e) => setModelsError(toLlmServerError(e).message));
  }, []);

  // The recommended model moves with the voice changer and other Sigil models (the policy key).
  useEffect(() => {
    if (tauri) loadModels();
  }, [tauri, loadModels, policy]);

  useEffect(() => {
    if (!tauri) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      planLlmServer(config)
        .then(() => !cancelled && setPlanError(null))
        .catch((e) => !cancelled && setPlanError(toLlmServerError(e).message));
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tauri, config, policy, ui.status?.state, planTick]);

  useEffect(() => {
    if (!tauri || !showLogs) return;
    let cancelled = false;
    const load = () =>
      getLlmServerLogs(LOG_LINES)
        .then((l) => !cancelled && setLogs(l))
        .catch(() => undefined);
    load();
    const timer = setInterval(load, LOG_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [tauri, showLogs]);

  const plan = ui.plan as RustLlmServerPlan | null;
  const planIsCurrent = !!plan && (config.model === "auto" || plan.model === config.model || isModelPath(config.model ?? ""));
  const groups = useMemo(() => groupModels(models), [models]);
  const recommended = models.find((m) => m.recommended);
  const selectedId = pr.serverModel?.trim() || "auto";
  const shownModel = models.find((m) => m.id === (selectedId === "auto" ? recommended?.id ?? plan?.model : selectedId));

  const modelLabel = (m: LlmModelInfoExt) => {
    const parts = [m.title, formatBytes(m.sizeBytes)];
    if (m.installed) parts.push(t("transl.llm_server_installed"));
    let label = parts.join(" · ");
    if (m.availability === "needs_runtime_update") label += ` (${t("transl.llm_server_needs_runtime_update")})`;
    else if (m.availability === "experimental") label += ` (${t("transl.llm_server_experimental")})`;
    else if (m.role === "ab") label += ` (${t("transl.llm_server_role_ab")})`;
    return label;
  };

  const modelOptions = useMemo(() => {
    const opts: (InputSelectOption | { label: string; options: InputSelectOption[] })[] = [
      {
        value: "auto",
        label: recommended ? t("transl.llm_server_model_auto_named", { model: recommended.title }) : t("transl.llm_server_model_auto"),
      },
    ];
    const general = groups.general.map((m) => ({ value: m.id, label: modelLabel(m) }));
    // A stored id missing from the list (a development catalog override) stays selectable.
    if (selectedId !== "auto" && !isModelPath(selectedId) && !models.some((m) => m.id === selectedId)) {
      general.unshift({ value: selectedId, label: selectedId });
    }
    if (general.length) opts.push({ label: t("transl.llm_server_group_general"), options: general });
    if (groups.specialists.length) {
      opts.push({ label: t("transl.llm_server_group_specialists"), options: groups.specialists.map((m) => ({ value: m.id, label: modelLabel(m) })) });
    }
    opts.push({ label: t("transl.llm_server_group_custom"), options: [{ value: CUSTOM_MODEL_OPTION, label: t("transl.llm_server_model_custom") }] });
    return opts;
    // modelLabel only depends on t and the model rows
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups, models, recommended, selectedId, t]);

  const onModel = (value: string) => {
    if (value === CUSTOM_MODEL_OPTION) {
      setPickCustom(true);
      if (isModelPath(customPath)) llmSlice().serverModel = customPath.trim();
      return;
    }
    setPickCustom(false);
    llmSlice().serverModel = value;
  };

  const commitPath = (path: string) => {
    setCustomPath(path);
    if (isModelPath(path)) llmSlice().serverModel = path.trim();
  };

  const browse = async () => {
    const picked = await open({ filters: [{ name: "GGUF", extensions: ["gguf"] }], multiple: false });
    if (typeof picked === "string") commitPath(picked);
  };

  const testLoad = async () => {
    setBusy("test");
    setTest(null);
    const t0 = performance.now();
    try {
      const lease = await acquireLlmServer("inspector", config);
      const seconds = ((performance.now() - t0) / 1000).toFixed(1);
      lease.release();
      setTest({ ok: true, seconds });
    } catch (e) {
      setTest({ ok: false, message: toLlmServerError(e).message });
    } finally {
      setBusy(null);
      setPlanTick((n) => n + 1);
    }
  };

  const stop = async () => {
    setBusy("stop");
    try {
      await stopLlmServer();
    } catch (e) {
      setTest({ ok: false, message: toLlmServerError(e).message });
    } finally {
      setBusy(null);
    }
  };

  const reasonText = (p: RustLlmServerPlan) => {
    const key = planReasonKey(p.reasonCode);
    return key ? t(key, { free: formatMb(p.availableMb), need: formatMb(p.vramMb) }) : p.reason;
  };

  const status = ui.status;
  const state = status?.state ?? "stopped";
  const consented = !!session.overBudget[consentKey(config)];
  const holderLabel = (h: string) => {
    const key = `transl.llm_server_holder_${h}`;
    const text = t(key);
    return text === key ? h : text;
  };

  if (!tauri) return null;

  return (
    <>
      <Inspector.SubHeader>{t("transl.llm_server_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("transl.llm_server_desc")}</Inspector.Description>

      <InputSelect label="transl.llm_server_model" options={modelOptions} value={pickCustom ? CUSTOM_MODEL_OPTION : modelSelectValue(pr.serverModel)} onValueChange={onModel} />
      {modelsError && <div className="text-xs text-error">{t("transl.llm_server_models_error", { message: modelsError })}</div>}
      {pickCustom && (
        <div className="flex items-end gap-2">
          <div className="flex-grow min-w-0">
            <InputText label="transl.llm_server_model_path" value={customPath} placeholder={"C:\\Models\\model.gguf"} onChange={(e) => setCustomPath(e.target.value)} onBlur={(e) => commitPath(e.target.value)} />
          </div>
          <button type="button" className="btn btn-sm" onClick={() => void browse()}>
            {t("transl.llm_server_browse")}
          </button>
        </div>
      )}
      {!pickCustom && shownModel?.description && (
        <div className="text-xs text-base-content/70">
          {shownModel.description}
          {shownModel.directions && shownModel.directions.length > 0 && (
            <span className="ms-1">{t("transl.llm_server_directions", { list: shownModel.directions.map((d) => `${d.from} → ${d.to}`).join(", ") })}</span>
          )}
        </div>
      )}

      <InputSelect
        label="transl.llm_server_variant"
        options={(["auto", "vulkan", "cuda13", "cpu"] as const).map((v) => ({ value: v, label: t(`transl.llm_server_variant_${v}`) }))}
        value={pr.serverVariant}
        onValueChange={(v) => (llmSlice().serverVariant = v as typeof pr.serverVariant)}
      />
      <InputText
        label="transl.llm_server_context"
        type="number"
        min={512}
        max={131072}
        step={512}
        value={pr.serverContext}
        onChange={(e) => {
          const v = parseContext(e.target.value);
          if (v !== null) llmSlice().serverContext = v;
        }}
      />
      <InputText
        label="transl.llm_server_gpu_layers"
        value={pr.serverGpuLayers < 0 ? "" : String(pr.serverGpuLayers)}
        placeholder={t("transl.llm_server_gpu_layers_auto")}
        onChange={(e) => (llmSlice().serverGpuLayers = parseGpuLayers(e.target.value))}
      />
      <InputSelect
        label="transl.llm_server_flash_attn"
        options={(["auto", "on", "off"] as const).map((v) => ({ value: v, label: t(`transl.llm_server_flash_${v}`) }))}
        value={pr.serverFlashAttn}
        onValueChange={(v) => (llmSlice().serverFlashAttn = v as typeof pr.serverFlashAttn)}
      />

      {/* The governor's plan for these settings. */}
      {pickCustom && !isModelPath(pr.serverModel) ? (
        <Box kind="neutral">{t("transl.llm_server_pick_file")}</Box>
      ) : planError ? (
        <Box kind="error">{planError}</Box>
      ) : !plan || !planIsCurrent ? (
        <Box kind="neutral">{t("transl.llm_server_plan_loading")}</Box>
      ) : (
        <Box kind={plan.allowed ? "success" : consented ? "warning" : "error"}>
          <div className="font-semibold">
            {plan.device === "gpu"
              ? t("transl.llm_server_plan_gpu", {
                  variant: t(`transl.llm_server_variant_${plan.variant}`),
                  layers: layersText(plan.gpuLayers) === "all" ? t("transl.llm_server_plan_all_layers") : t("transl.llm_server_plan_layers", { count: plan.gpuLayers }),
                  need: formatMb(plan.vramMb),
                  free: formatMb(plan.availableMb),
                })
              : t("transl.llm_server_plan_cpu")}
          </div>
          <div>
            {!plan.allowed && <span className="font-semibold me-1">{t("transl.llm_server_plan_refused")}</span>}
            {reasonText(plan)}
          </div>
          {canOverride(plan) &&
            (consented ? (
              <div className="flex items-center gap-2">
                <span>{t("transl.llm_server_override_on")}</span>
                <button type="button" className="btn btn-ghost btn-xs" onClick={() => setOverBudgetConsent(config, false)}>
                  {t("transl.llm_server_override_undo")}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <button type="button" className="btn btn-warning btn-xs" onClick={() => setOverBudgetConsent(config, true)}>
                  {t("transl.llm_server_override")}
                </button>
                <span className="text-base-content/60">{t("transl.llm_server_override_hint")}</span>
              </div>
            ))}
        </Box>
      )}
      {/* Downloads only for a plan that can run (a hard refusal would never use them). */}
      {!(pickCustom && !isModelPath(pr.serverModel)) && planIsCurrent && plan && offersDownloads(plan) && (
        <ModelRequirement
          ids={plan.missing}
          reason={t("transl.llm_server_requires")}
          onReady={() => {
            setPlanTick((n) => n + 1);
            loadModels();
          }}
        />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={classNames("btn btn-sm", busy === "test" && "loading")} disabled={busy !== null || state === "starting"} onClick={() => void testLoad()}>
          {t("transl.llm_server_test")}
        </button>
        <button type="button" className={classNames("btn btn-sm btn-ghost", busy === "stop" && "loading")} disabled={busy !== null || state === "stopped"} onClick={() => void stop()}>
          {t("transl.llm_server_stop")}
        </button>
      </div>
      {test && (
        <div className={classNames("text-xs", test.ok ? "text-success" : "text-error")}>
          {test.ok ? t("transl.llm_server_test_ok", { seconds: test.seconds }) : t("transl.llm_server_test_failed", { message: test.message })}
        </div>
      )}

      {/* Live status and log: usable while translation runs (the settings above are not). */}
      <div className="pointer-events-auto space-y-2">
        <Box kind={state === "failed" ? "error" : state === "ready" ? "success" : "neutral"}>
          <div className="flex items-center gap-2">
            <span className="font-semibold">{t("transl.llm_server_status")}:</span>
            <span>{t(stateKey(state))}</span>
          </div>
          {status && state !== "stopped" && status.model && (
            <div className="text-base-content/70">
              {t("transl.llm_server_status_detail", {
                model: models.find((m) => m.id === status.model)?.title ?? (isModelPath(status.model) ? fileName(status.model) : status.model),
                variant: t(`transl.llm_server_variant_${status.variant ?? "auto"}`),
                ctx: status.ctxSize ?? "?",
                vram: status.vramMb > 0 ? formatMb(status.vramMb) : t("transl.llm_server_no_vram"),
              })}
            </div>
          )}
          {status && status.restarts > 0 && <div className="text-base-content/70">{t("transl.llm_server_restarts", { count: status.restarts })}</div>}
          {ui.holders.length > 0 && <div className="text-base-content/70">{t("transl.llm_server_holders", { list: ui.holders.map(holderLabel).join(", ") })}</div>}
          {status?.error && <div className="text-error break-words">{status.error}</div>}
          {!status?.error && session.lastError && state !== "ready" && session.lastError.message !== test?.message && (
            <div className="text-error break-words">{session.lastError.message}</div>
          )}
        </Box>
        <button type="button" className="btn btn-ghost btn-xs" onClick={() => setShowLogs((v) => !v)}>
          {showLogs ? t("transl.llm_server_logs_hide") : t("transl.llm_server_logs_show")}
        </button>
        {showLogs && (
          <pre dir="ltr" className="text-[10px] leading-snug max-h-56 overflow-auto rounded-box bg-base-200 p-2 whitespace-pre-wrap break-all text-start">
            {logs.length ? logs.map((l) => l.text).join("\n") : t("transl.llm_server_logs_empty")}
          </pre>
        )}
      </div>
    </>
  );
});
