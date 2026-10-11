import { createLlmClient } from "@/core/services/llm/client";
import { resolveLlmEndpoint } from "@/core/services/llm/providers";
import { LLM_PROVIDERS, LLM_REASONING_EFFORTS, LLM_TRANSPORTS, type LlmProviderId } from "@/core/services/llm/types";
import { filterGlossary } from "@/core/services/translation/glossary";
import { canonicalLanguage } from "@/core/services/translation/languages";
import {
  buildTranslationPrompt,
  detectPromptFormat,
  resolvePromptFormat,
  type BuiltPrompt,
  type ResolvedPromptFormat,
} from "@/core/services/translation/prompt";
import { TRANSLATION_PROMPT_FORMATS, Translation_Backends, type Translation_State, type TranslationPromptFormat } from "@/core/services/translation/schema";
import { llmConnection, servedModelInfo } from "@/core/services/translation/services/llm";
import { isTauri } from "@tauri-apps/api/core";
import { FC, memo, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputBaseText, InputCheckbox, InputCode, InputSelect, type InputSelectOption } from "../components/input";
import { Details, Hint, LiveArea, NumberField, sliceSetter, TextField, translationData, useInterimSwitch } from "./common";
import { LanguagePair } from "./LanguagePair";
import { LlmServer } from "./LlmServer";
import {
  customPromptIssues,
  listsModel,
  LLM_PROVIDER_FIELDS,
  LLM_SERVER_PRESETS,
  llmServerPresetOf,
  promptFormatUses,
  type LlmServerPresetId,
} from "./logic";

type LlmSlice = Translation_State["llm"];
type RemoteProvider = keyof typeof LLM_PROVIDER_FIELDS;

/** Model placeholders per remote provider (the schema defaults, shown when the field is empty). */
const MODEL_PLACEHOLDER: Record<RemoteProvider, string> = {
  gemini: "gemini-3.5-flash-lite",
  openrouter: "google/gemini-2.5-flash-lite",
  openai: "gpt-4o-mini",
  custom: "qwen3:8b",
};

const KEY_PLACEHOLDER: Record<RemoteProvider, string> = {
  gemini: "AIza...",
  openrouter: "sk-or-...",
  openai: "sk-...",
  custom: "",
};

/** Providers whose endpoint accepts a raw completion (same rule as the LLM engine; used for the prompt preview). */
function rawAvailable(provider: LlmProviderId): boolean {
  return provider === "bundled" || provider === "custom" || provider === "openrouter";
}

/** Largest model list shown in the picker after a connection check. */
const MAX_LISTED_MODELS = 500;

/**
 * LLM panel (`llm` backend): the provider (Sigil's own model server, Gemini, OpenRouter, OpenAI or a server you run,
 * with presets for Ollama, LM Studio, llama.cpp and vLLM), keys and model with a connection check that lists the
 * server's models, the prompt format with a custom template and a live prompt preview, reasoning, limits, streaming
 * and the connection route. Everything here applies on the next start (the shell locks it while translation runs).
 *
 * Owned by P4-F.
 */
export const Llm: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const up = sliceSetter("llm");
  const provider = pr.provider;
  const interim = useInterimSwitch(Translation_Backends.llm);

  const providerOptions: InputSelectOption[] = LLM_PROVIDERS.map((p) => ({ value: p, label: t(`transl.llm_provider_${p}`) }));

  return (
    <>
      <Inspector.SubHeader>{t("transl.llm_title")}</Inspector.SubHeader>
      <Hint>{t("transl.llm_desc")}</Hint>
      <LanguagePair />

      <InputSelect label="transl.llm_field_provider" options={providerOptions} value={provider} onValueChange={(v) => up("provider", v as LlmProviderId)} />
      {provider === "bundled" ? (
        isTauri() ? <LlmServer /> : <Hint tone="warning">{t("transl.llm_bundled_desktop_only")}</Hint>
      ) : (
        <RemoteProviderFields provider={provider as RemoteProvider} />
      )}

      <PromptSettings />

      <Inspector.SubHeader>{t("transl.llm_reply_subheader")}</Inspector.SubHeader>
      <InputSelect
        label="transl.llm_field_reasoning"
        options={LLM_REASONING_EFFORTS.map((effort) => ({ value: effort, label: t(`transform.reasoning_${effort}`) }))}
        value={pr.reasoningEffort}
        onValueChange={(v) => up("reasoningEffort", v as LlmSlice["reasoningEffort"])}
      />
      <Hint>{t("transl.llm_reasoning_hint")}</Hint>
      <NumberField label="transl.llm_field_max_tokens" value={pr.maxTokens} min={16} max={8192} step={64} onCommit={(v) => up("maxTokens", v)} />
      <NumberField
        label="transl.llm_field_temperature"
        value={pr.temperature}
        min={0}
        max={2}
        integer={false}
        step={0.05}
        onCommit={(v) => up("temperature", v)}
      />
      <NumberField
        label="transl.llm_field_timeout"
        value={pr.timeoutMs}
        min={1000}
        max={120_000}
        scale={1000}
        step={1}
        onCommit={(v) => up("timeoutMs", v)}
      />
      <Hint>{t("transl.llm_timeout_hint")}</Hint>
      <InputCheckbox label="transl.llm_field_stream" value={pr.stream} onChange={(v) => up("stream", v)} />
      <Hint>{t("transl.llm_stream_hint")}</Hint>
      <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={interim.on} />
      <Hint>{t("transl.llm_interim_hint")}</Hint>
      {interim.heldOff && <Hint tone="warning">{t("transl.interim_cpu_hint")}</Hint>}

      <Details title={t("transl.llm_advanced")}>
        <InputSelect
          label="transl.llm_field_transport"
          options={LLM_TRANSPORTS.map((v) => ({ value: v, label: t(`transl.llm_transport_${v}`) }))}
          value={pr.transport}
          onValueChange={(v) => up("transport", v as LlmSlice["transport"])}
        />
        <Hint>{t("transl.llm_transport_hint")}</Hint>
      </Details>
    </>
  );
});

/** Key, model and (custom) server address of a remote provider, plus the connection check. */
const RemoteProviderFields: FC<{ provider: RemoteProvider }> = ({ provider }) => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const up = sliceSetter("llm");
  const fields = LLM_PROVIDER_FIELDS[provider];
  // "Other address" keeps the URL (typed next) and is shown until the URL changes; otherwise the choice follows the URL.
  const [otherFor, setOtherFor] = useState<string | null>(null);
  const preset = otherFor !== null && otherFor === pr.customUrl ? "other" : llmServerPresetOf(pr.customUrl);

  const presetOptions: InputSelectOption[] = [
    ...LLM_SERVER_PRESETS.map((p) => ({ value: p.id, label: t(`transl.llm_preset_${p.id}`) })),
    { value: "other", label: t("transl.llm_preset_other") },
  ];
  const applyPreset = (id: string) => {
    const hit = LLM_SERVER_PRESETS.find((p) => p.id === (id as LlmServerPresetId));
    if (hit) {
      setOtherFor(null);
      up("customUrl", hit.url);
    } else {
      setOtherFor(pr.customUrl);
    }
  };

  return (
    <>
      {provider === "custom" && (
        <>
          <InputSelect label="transl.llm_field_preset" options={presetOptions} value={preset} onValueChange={applyPreset} />
          <TextField
            label="transl.llm_field_url"
            value={pr.customUrl}
            placeholder="http://localhost:11434/v1"
            onChange={(e) => up("customUrl", e.target.value)}
          />
          <Hint>{t("transl.llm_custom_hint")}</Hint>
        </>
      )}
      <TextField
        label={provider === "custom" ? "transl.llm_field_key_optional" : "transl.llm_field_key"}
        type="password"
        autoComplete="new-password"
        value={pr[fields.key]}
        placeholder={KEY_PLACEHOLDER[provider]}
        onChange={(e) => up(fields.key, e.target.value)}
      />
      <TextField
        label="transl.llm_field_model"
        value={pr[fields.model]}
        placeholder={MODEL_PLACEHOLDER[provider]}
        onChange={(e) => up(fields.model, e.target.value)}
      />
      {provider === "gemini" && <Hint>{t("transl.llm_gemini_hint")}</Hint>}
      {provider === "openrouter" && <Hint>{t("transl.llm_openrouter_hint")}</Hint>}
      <ConnectionCheck provider={provider} />
    </>
  );
};

type CheckState = { kind: "idle" } | { kind: "running" } | { kind: "failed"; error: string } | { kind: "done"; models: string[]; total: number };



/** Lists the provider's models with the current settings (proves URL, key and route) and offers them as a pick list. */
const ConnectionCheck: FC<{ provider: RemoteProvider }> = ({ provider }) => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const [state, setState] = useState<CheckState>({ kind: "idle" });
  const controller = useRef<AbortController | null>(null);
  const fields = LLM_PROVIDER_FIELDS[provider];
  const model = pr[fields.model].trim();

  useEffect(() => () => controller.current?.abort(), []);
  // A result describes the settings it was made with.
  const connectionKey = JSON.stringify([provider, pr.transport, pr[fields.key], provider === "custom" ? pr.customUrl : ""]);
  useEffect(() => {
    controller.current?.abort();
    setState({ kind: "idle" });
  }, [connectionKey]);

  const run = async () => {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    setState({ kind: "running" });
    try {
      const cfg = translationData().llm;
      // The model name is not needed to list models: a placeholder keeps an empty field from failing the check.
      const settings = llmConnection(cfg);
      const endpoint = resolveLlmEndpoint({ ...settings, model: settings.model || MODEL_PLACEHOLDER[provider] });
      const models = (await createLlmClient(endpoint).listModels(abort.signal)).slice().sort((a, b) => a.localeCompare(b));
      if (abort.signal.aborted) return;
      setState({ kind: "done", models: models.slice(0, MAX_LISTED_MODELS), total: models.length });
    } catch (e) {
      if (abort.signal.aborted) return;
      setState({ kind: "failed", error: e instanceof Error ? e.message : String(e) });
    }
  };

  // The verdict follows the model field, so picking a listed model updates it.
  let verdict: { ok: boolean; text: string } | null = null;
  if (state.kind === "failed") verdict = { ok: false, text: t("transl.llm_check_failed", { error: state.error }) };
  else if (state.kind === "done") {
    const n = state.total;
    if (n === 0) verdict = { ok: true, text: t("transl.llm_check_ok_empty") };
    else if (!model) verdict = { ok: false, text: t("transl.llm_check_ok_pick", { n }) };
    else if (listsModel(state.models, model)) verdict = { ok: true, text: t("transl.llm_check_ok", { n }) };
    else verdict = { ok: false, text: t("transl.llm_check_model_missing", { model, n }) };
  }

  const pickOptions = useMemo<InputSelectOption[]>(() => {
    if (state.kind !== "done" || state.models.length === 0) return [];
    const opts = state.models.map((m) => ({ value: m, label: m }));
    return model && !state.models.includes(model) ? [{ value: model, label: model }, ...opts] : opts;
  }, [state, model]);

  return (
    <>
      {/* Checking is harmless while translation runs; picking a model is a setting like the others. */}
      <LiveArea className="mt-1">
        <div className="flex items-center gap-2">
          <button type="button" className="btn btn-sm" disabled={state.kind === "running"} onClick={() => void run()}>
            {state.kind === "running" ? t("transl.llm_check_running") : t("transl.llm_check")}
          </button>
        </div>
        {verdict && (
          <div role="status" className={`text-xs px-1 break-words ${verdict.ok ? "text-success" : "text-warning"}`}>
            {verdict.text}
          </div>
        )}
      </LiveArea>
      {pickOptions.length > 0 && (
        <InputSelect
          label="transl.llm_field_listed_models"
          options={pickOptions}
          value={model}
          onValueChange={(v) => (translationData().llm[fields.model] = v)}
        />
      )}
    </>
  );
};

const FORMAT_LABEL_KEY = (f: TranslationPromptFormat) => `transl.llm_format_${f.replace(/-/g, "_")}`;

/** Prompt family, custom template and preview. */
const PromptSettings: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const up = sliceSetter("llm");
  const resolved = useResolvedFormat(pr);
  const uses = promptFormatUses(resolved);
  const usedParts = [
    uses.context && t("transl.llm_uses_context"),
    uses.glossary && t("transl.llm_uses_glossary"),
    uses.background && t("transl.llm_uses_background"),
  ].filter((x): x is string => !!x);
  const issues = customPromptIssues(pr.customPrompt);

  return (
    <>
      <Inspector.SubHeader>{t("transl.llm_prompt_subheader")}</Inspector.SubHeader>
      <InputSelect
        label="transl.llm_field_prompt_format"
        options={TRANSLATION_PROMPT_FORMATS.map((f) => ({ value: f, label: t(FORMAT_LABEL_KEY(f)) }))}
        value={pr.promptFormat}
        onValueChange={(v) => up("promptFormat", v as TranslationPromptFormat)}
      />
      {pr.promptFormat === "auto" && <Hint>{t("transl.llm_format_resolved", { format: t(FORMAT_LABEL_KEY(resolved)) })}</Hint>}
      <Hint>
        {usedParts.length ? t("transl.llm_format_uses", { list: usedParts.join(", ") }) : t("transl.llm_format_uses_none")}
      </Hint>

      {pr.promptFormat === "custom" && (
        <>
          <InputCode
            label="transl.llm_field_custom_prompt"
            language="text"
            value={pr.customPrompt}
            placeholder={t("transl.llm_custom_prompt_example")}
            onChange={(v) => up("customPrompt", v ?? "")}
          />
          <div className="flex items-center gap-2">
            <button type="button" className="btn btn-xs btn-ghost normal-case" onClick={() => up("customPrompt", t("transl.llm_custom_prompt_example"))}>
              {t("transl.llm_custom_prompt_use_example")}
            </button>
          </div>
          <Hint>{t("transl.llm_custom_prompt_hint")}</Hint>
          {!pr.customPrompt.trim() && <Hint tone="warning">{t("transl.llm_custom_prompt_empty")}</Hint>}
          {issues.missingText && <Hint tone="warning">{t("transl.llm_custom_prompt_no_text")}</Hint>}
          {issues.unknown.length > 0 && (
            <Hint tone="warning">{t("transl.llm_custom_prompt_unknown", { list: issues.unknown.map((u) => `{${u}}`).join(", ") })}</Hint>
          )}
        </>
      )}

      <PromptPreview format={resolved} />
    </>
  );
};

/**
 * The prompt family the engine will use: the setting, or for `auto` the catalog's family of the served model (the
 * bundled server) or a guess from the model name (remote providers).
 */
function useResolvedFormat(pr: Readonly<LlmSlice>): ResolvedPromptFormat {
  const [catalogFormat, setCatalogFormat] = useState<{ model: string; format: string | null } | null>(null);
  const bundled = pr.provider === "bundled";
  const serverModel = pr.serverModel;

  useEffect(() => {
    if (pr.promptFormat !== "auto" || !bundled || !isTauri()) return;
    let cancelled = false;
    servedModelInfo(serverModel)
      .then((info) => !cancelled && setCatalogFormat({ model: info?.id ?? serverModel, format: info?.promptFormat ?? null }))
      .catch(() => !cancelled && setCatalogFormat(null));
    return () => {
      cancelled = true;
    };
  }, [pr.promptFormat, bundled, serverModel]);

  if (pr.promptFormat !== "auto") return pr.promptFormat;
  if (bundled) return resolvePromptFormat("auto", { catalogFormat: catalogFormat?.format, model: catalogFormat?.model ?? serverModel });
  const field = LLM_PROVIDER_FIELDS[pr.provider as RemoteProvider];
  return detectPromptFormat(field ? pr[field.model] : "");
}

function renderPrompt(built: BuiltPrompt): string {
  if (built.mode === "completion") return built.prompt;
  return built.messages.map((m) => `[${m.role}]\n${m.content}`).join("\n\n");
}

/** What the model receives for a sample line with the current settings, glossary and background. */
const PromptPreview: FC<{ format: ResolvedPromptFormat }> = ({ format }) => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const [sample, setSample] = useState(() => t("transl.llm_preview_sample"));
  const [open, setOpen] = useState(false);

  const text = useMemo(() => {
    if (!open) return "";
    const line = sample.trim() || t("transl.llm_preview_sample");
    const from = canonicalLanguage(data.sourceLanguage);
    const to = canonicalLanguage(data.targetLanguage) || data.targetLanguage.trim() || "en";
    try {
      const built = buildTranslationPrompt(
        format,
        {
          text: line,
          from,
          to,
          glossary: filterGlossary(data.glossary, line, from, to),
          background: data.context.enabled ? data.context.background : "",
          customPrompt: data.llm.customPrompt,
        },
        { rawAvailable: rawAvailable(data.llm.provider), bundled: data.llm.provider === "bundled" },
      );
      return renderPrompt(built);
    } catch (e) {
      return e instanceof Error ? e.message : String(e);
    }
  }, [open, sample, format, data, t]);

  return (
    <Details title={t("transl.llm_preview_title")} onToggle={setOpen}>
      <div className="pointer-events-auto flex flex-col gap-2">
        <InputBaseText
          fieldWidth={false}
          className="w-full"
          dir="auto"
          aria-label={t("transl.llm_preview_line")}
          placeholder={t("transl.llm_preview_line")}
          value={sample}
          onChange={(e) => setSample(e.target.value)}
        />
        <pre dir="ltr" className="text-[10px] leading-snug max-h-56 overflow-auto rounded-box bg-base-200 p-2 whitespace-pre-wrap break-words text-start">
          {text}
        </pre>
        <Hint>{t("transl.llm_preview_hint")}</Hint>
      </div>
    </Details>
  );
};
