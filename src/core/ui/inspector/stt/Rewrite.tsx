import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import {
    TRANSFORM_GEMINI_MODEL_DEFAULT,
    TRANSFORM_MAX_TOKENS_MAX,
    TRANSFORM_MAX_TOKENS_MIN,
    TRANSFORM_REASONING_EFFORTS,
    TRANSFORM_TIMEOUT_MS_MAX,
    TRANSFORM_TIMEOUT_MS_MIN,
    Transform_Backends,
    Transform_Provider,
    Transform_ReasoningEffort,
    Transform_State,
} from "@/core/services/transform/schema";
import { resolveTransformEndpoint } from "@/core/services/transform/services/llmRequest";
import { createLlmClient, modelListed } from "@/core/services/llm/client";
import { LLM_TRANSPORTS, LlmTransportChoice } from "@/core/services/llm/types";
import { ServiceNetworkState } from "@/types";
import Inspector from "../components";
import ServiceButton from "../../service-button";
import { InputCheckbox, InputSelect, InputText, InputCode } from "../components/input";
import { Advanced } from "./common";
import { FC, useEffect, useRef, useState } from "react";

/** Settings whose change needs a new engine; everything else is read per request. */
const RESTART_KEYS: (keyof Transform_State)[] = ["backend", "provider"];

/** A number field that commits only valid integers in [min, max] (it can be empty or partly typed meanwhile). */
const IntField: FC<{ label: string; value: number; min: number; max: number; step?: number; onCommit: (v: number) => void }> = ({
    label,
    value,
    min,
    max,
    step = 1,
    onCommit,
}) => {
    const [text, setText] = useState(String(value));
    useEffect(() => setText(String(value)), [value]);
    return (
        <InputText
            label={label}
            type="number"
            inputMode="numeric"
            min={min}
            max={max}
            step={step}
            value={text}
            onChange={(e) => {
                setText(e.target.value);
                const n = Number(e.target.value);
                if (e.target.value !== "" && Number.isInteger(n) && n >= min && n <= max) onCommit(n);
            }}
            onBlur={() => setText(String(value))}
        />
    );
};

type CheckState = { kind: "idle" } | { kind: "running" } | { kind: "done"; ok: boolean; text: string };

/** Lists the server's models with the current settings: proves the URL, key and route work before going live. */
const ConnectionCheck: FC<{ data: Transform_State }> = ({ data }) => {
    const { t } = useTranslation();
    const [state, setState] = useState<CheckState>({ kind: "idle" });
    const controller = useRef<AbortController | null>(null);
    useEffect(() => () => controller.current?.abort(), []);
    // Stale results would describe other settings.
    const connectionKey = JSON.stringify([
        data.provider,
        data.transport,
        data.openaiKey,
        data.openaiUrl,
        data.openaiModel,
        data.openRouterKey,
        data.openRouterUrl,
        data.openRouterModel,
        data.geminiKey,
        data.geminiModel,
        data.customKey,
        data.customUrl,
        data.customModel,
    ]);
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
            const endpoint = resolveTransformEndpoint(window.ApiServer.state.services.transform.data);
            const models = await createLlmClient(endpoint).listModels(abort.signal);
            if (abort.signal.aborted) return;
            const model = endpoint.model;
            const known = modelListed(models, model);
            const text =
                models.length === 0
                    ? t("transform.check_ok_empty")
                    : known
                      ? t("transform.check_ok", { n: models.length })
                      : t("transform.check_ok_model_missing", { model, n: models.length });
            setState({ kind: "done", ok: models.length === 0 || known, text });
        } catch (e) {
            if (abort.signal.aborted) return;
            setState({ kind: "done", ok: false, text: t("transform.check_failed", { error: e instanceof Error ? e.message : String(e) }) });
        }
    };

    return (
        <div className="flex flex-col gap-1 mt-1 mb-2">
            <div className="flex items-center gap-2">
                <button type="button" className="btn btn-sm" disabled={state.kind === "running"} onClick={run}>
                    {state.kind === "running" ? t("transform.check_running") : t("transform.btn_check_connection")}
                </button>
            </div>
            {state.kind === "done" && (
                <div role="status" className={`text-xs px-1 ${state.ok ? "text-success" : "text-warning"}`}>
                    {state.text}
                </div>
            )}
        </div>
    );
};

/**
 * Rewrite: the second tab of the Speech to Text inspector (the service is still `transform`; ids and saved settings
 * are unchanged). Rewrites each finished sentence with a language model. Everything but provider, key, model,
 * instructions and Start sits under "More options".
 */
export const Rewrite: FC = () => {
    const { t } = useTranslation();
    const transformState = useSnapshot(window.ApiServer.state.services.transform);
    const { data } = transformState;
    const service = window.ApiServer.transform;

    const { status } = useSnapshot(window.ApiServer.transform.serviceState);

    const handleUpdate = <K extends keyof Transform_State>(key: K, value: Transform_State[K]) => {
        window.ApiServer.patchService("transform", (state) => {
            state.data[key] = value;
        });
        if (status === ServiceNetworkState.connected && RESTART_KEYS.includes(key)) {
            service.start();
        }
    };

    const llm = data.backend === Transform_Backends.openai;

    // The form is `display: contents`, so the groups below are what the inspector spaces: the fields and Start (spaced),
    // the toggles (stacked tight, like the Speech tab's), and the fold.
    return (
        <form onSubmit={e => e.preventDefault()} className="contents">
            <input type="text" name="username" autoComplete="username" className="hidden" readOnly placeholder="username" />

            <div className="flex flex-col space-y-2 pt-2">
                <Inspector.Description>{t("transform.intro")}</Inspector.Description>

                {/* The debug engine is a development aid: shown in dev builds, and to a profile still on it so it can leave. */}
                {(!llm || import.meta.env.DEV) && (
                    <InputSelect
                        label="transform.field_engine"
                        value={data.backend}
                        options={[
                            { label: t("transform.engine_llm"), value: Transform_Backends.openai },
                            { label: t("transform.engine_debug"), value: Transform_Backends.dummy },
                        ]}
                        onValueChange={(v) => handleUpdate("backend", v as Transform_Backends)}
                    />
                )}

                {llm && (
                    <>
                        <InputSelect
                            label="transform.field_provider"
                            value={data.provider}
                            options={[
                                { label: t("transform.provider_openrouter"), value: "openrouter" },
                                { label: t("transform.provider_gemini"), value: "gemini" },
                                { label: t("transform.provider_openai"), value: "openai" },
                                { label: t("transform.provider_custom"), value: "custom" },
                            ]}
                            onValueChange={(v) => handleUpdate("provider", v as Transform_Provider)}
                        />

                        {data.provider === "openai" && (
                            <>
                                <InputText
                                    label="transform.field_api_key"
                                    type="password"
                                    autoComplete="new-password"
                                    value={data.openaiKey}
                                    placeholder="sk-..."
                                    onChange={(e) => handleUpdate("openaiKey", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_api_url"
                                    value={data.openaiUrl}
                                    placeholder="https://api.openai.com/v1"
                                    onChange={(e) => handleUpdate("openaiUrl", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_model"
                                    value={data.openaiModel}
                                    placeholder="gpt-4o-mini"
                                    onChange={(e) => handleUpdate("openaiModel", e.target.value)}
                                />
                            </>
                        )}

                        {data.provider === "openrouter" && (
                            <>
                                <InputText
                                    label="transform.field_api_key"
                                    type="password"
                                    autoComplete="new-password"
                                    value={data.openRouterKey}
                                    placeholder="sk-or-..."
                                    onChange={(e) => handleUpdate("openRouterKey", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_api_url"
                                    value={data.openRouterUrl}
                                    placeholder="https://openrouter.ai/api/v1"
                                    onChange={(e) => handleUpdate("openRouterUrl", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_model"
                                    value={data.openRouterModel}
                                    placeholder="google/gemini-2.5-flash-lite"
                                    onChange={(e) => handleUpdate("openRouterModel", e.target.value)}
                                />
                            </>
                        )}

                        {data.provider === "gemini" && (
                            <>
                                <InputText
                                    label="transform.field_api_key"
                                    type="password"
                                    autoComplete="new-password"
                                    value={data.geminiKey}
                                    placeholder="AIza..."
                                    onChange={(e) => handleUpdate("geminiKey", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_model"
                                    value={data.geminiModel}
                                    placeholder={TRANSFORM_GEMINI_MODEL_DEFAULT}
                                    onChange={(e) => handleUpdate("geminiModel", e.target.value)}
                                />
                                <div className="text-xs text-base-content/70 px-1">{t("transform.gemini_hint")}</div>
                            </>
                        )}

                        {data.provider === "custom" && (
                            <>
                                <InputText
                                    label="transform.field_api_key_optional"
                                    type="password"
                                    autoComplete="new-password"
                                    value={data.customKey}
                                    placeholder="sk-..."
                                    onChange={(e) => handleUpdate("customKey", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_api_url"
                                    value={data.customUrl}
                                    placeholder="http://localhost:11434/v1"
                                    onChange={(e) => handleUpdate("customUrl", e.target.value)}
                                />
                                <InputText
                                    label="transform.field_model"
                                    value={data.customModel}
                                    placeholder="llama3.2"
                                    onChange={(e) => handleUpdate("customModel", e.target.value)}
                                />
                            </>
                        )}

                        <ConnectionCheck data={data as Transform_State} />

                        <InputCode
                            label="transform.field_system_prompt"
                            language="text"
                            value={data.systemPrompt}
                            onChange={(v) => handleUpdate("systemPrompt", v || "")}
                            placeholder={t("transform.system_prompt_placeholder")}
                        />
                    </>
                )}

                <div className="pt-2 flex flex-col">
                    <ServiceButton
                        status={status}
                        startLabel="transform.btn_start"
                        stopLabel="transform.btn_stop"
                        onStart={() => service.start()}
                        onStop={() => service.stop()}
                    />
                </div>
            </div>

            <div className="flex flex-col mt-4">
                <InputCheckbox
                    label="common.field_action_bar"
                    value={transformState.showActionButton}
                    onChange={(v) => {
                        window.ApiServer.state.services.transform.showActionButton = v;
                    }}
                />
                <InputCheckbox
                    label="common.field_auto_start"
                    value={data.autoStart}
                    onChange={(v) => handleUpdate("autoStart", v)}
                />
                <InputCheckbox
                    label="stt.field_stop_with_stream"
                    value={data.stopWithStream}
                    onChange={(v) => handleUpdate("stopWithStream", v)}
                />
                <InputCheckbox
                    label="transform.field_context_history"
                    value={data.contextHistory}
                    onChange={(v) => handleUpdate("contextHistory", v)}
                />
                <div className="text-xs text-base-content/70 px-1 mt-1">{t("transform.context_history_hint")}</div>
            </div>

            {llm && (
                <div className="mt-4">
                    <Advanced title={t("transform.advanced_title")}>
                        <InputCheckbox
                            label="transform.field_fence_input"
                            value={data.fenceInput}
                            onChange={(v) => handleUpdate("fenceInput", v)}
                        />
                        <div className="text-xs text-base-content/70 mb-2 px-1">{t("transform.fence_input_hint")}</div>

                        <IntField
                            label="transform.field_max_tokens"
                            value={data.maxTokens}
                            min={TRANSFORM_MAX_TOKENS_MIN}
                            max={TRANSFORM_MAX_TOKENS_MAX}
                            step={64}
                            onCommit={(v) => handleUpdate("maxTokens", v)}
                        />
                        <InputSelect
                            label="transform.field_reasoning_effort"
                            value={data.reasoningEffort}
                            options={TRANSFORM_REASONING_EFFORTS.map((effort) => ({
                                label: t(`transform.reasoning_${effort}`),
                                value: effort,
                            }))}
                            onValueChange={(v) => handleUpdate("reasoningEffort", v as Transform_ReasoningEffort)}
                        />
                        <div className="text-xs text-base-content/70 mt-1 mb-2 px-1">{t("transform.tokens_reasoning_hint")}</div>

                        <InputCheckbox
                            label="transform.field_stream"
                            value={data.stream}
                            onChange={(v) => handleUpdate("stream", v)}
                        />
                        <div className="text-xs text-base-content/70 mb-2 px-1">{t("transform.stream_hint_until_next")}</div>
                        <IntField
                            label="transform.field_timeout"
                            value={Math.round(data.timeoutMs / 1000)}
                            min={TRANSFORM_TIMEOUT_MS_MIN / 1000}
                            max={TRANSFORM_TIMEOUT_MS_MAX / 1000}
                            onCommit={(v) => handleUpdate("timeoutMs", v * 1000)}
                        />
                        <InputSelect
                            label="transform.field_transport"
                            value={data.transport}
                            options={LLM_TRANSPORTS.map((transport) => ({
                                label: t(`transform.transport_${transport}`),
                                value: transport,
                            }))}
                            onValueChange={(v) => handleUpdate("transport", v as LlmTransportChoice)}
                        />
                        <div className="text-xs text-base-content/70 mt-1 mb-2 px-1">{t("transform.transport_hint")}</div>
                        <div className="text-xs text-base-content/70 mb-2 px-1">{t("transform.live_settings_hint")}</div>
                    </Advanced>
                </div>
            )}
        </form>
    );
};

export default Rewrite;
