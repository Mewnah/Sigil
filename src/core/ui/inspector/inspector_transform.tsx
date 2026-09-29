import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import {
    TRANSFORM_MAX_TOKENS_MAX,
    TRANSFORM_MAX_TOKENS_MIN,
    TRANSFORM_REASONING_EFFORTS,
    Transform_Backends,
    Transform_Provider,
    Transform_ReasoningEffort,
    Transform_State,
} from "@/core/services/transform/schema";
import { ServiceNetworkState } from "@/types";
import Inspector from "./components";
import ServiceButton from "../service-button";
import { InputCheckbox, InputSelect, InputText, InputCode } from "./components/input";
import { FC, useEffect, useState } from "react";

/** Settings whose change needs a new engine; everything else is read per request. */
const RESTART_KEYS: (keyof Transform_State)[] = ["backend", "provider"];

const MaxTokensField: FC<{ value: number; onCommit: (v: number) => void }> = ({ value, onCommit }) => {
    // Local text so the field can be empty or partly typed; only valid numbers are committed.
    const [text, setText] = useState(String(value));
    useEffect(() => setText(String(value)), [value]);
    return (
        <InputText
            label="transform.field_max_tokens"
            type="number"
            inputMode="numeric"
            min={TRANSFORM_MAX_TOKENS_MIN}
            max={TRANSFORM_MAX_TOKENS_MAX}
            step={64}
            value={text}
            onChange={(e) => {
                setText(e.target.value);
                const n = Number(e.target.value);
                if (e.target.value !== "" && Number.isInteger(n) && n >= TRANSFORM_MAX_TOKENS_MIN && n <= TRANSFORM_MAX_TOKENS_MAX)
                    onCommit(n);
            }}
            onBlur={() => setText(String(value))}
        />
    );
};

const Inspector_Transform = () => {
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

    return (
        <Inspector.Body>
            <Inspector.Header>{t("transform.title")}</Inspector.Header>
            <Inspector.Content>
                <form onSubmit={e => e.preventDefault()} className="contents">
                    <input type="text" name="username" autoComplete="username" className="hidden" readOnly placeholder="username" />

                    <Inspector.SubHeader>{t("transform.engine_subheader")}</Inspector.SubHeader>
                    <InputSelect
                        label="transform.field_engine"
                        value={data.backend}
                        options={[
                            { label: t("transform.engine_llm"), value: Transform_Backends.openai },
                            { label: t("transform.engine_debug"), value: Transform_Backends.dummy },
                        ]}
                        onValueChange={(v) => handleUpdate("backend", v as Transform_Backends)}
                    />

                    {data.backend === Transform_Backends.openai && (
                        <>
                            <InputSelect
                                label="transform.field_provider"
                                value={data.provider}
                                options={[
                                    { label: t("transform.provider_openrouter"), value: "openrouter" },
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

                            <div className="h-4" />
                            <InputCode
                                label="transform.field_system_prompt"
                                language="text"
                                value={data.systemPrompt}
                                onChange={(v) => handleUpdate("systemPrompt", v || "")}
                                placeholder={t("transform.system_prompt_placeholder")}
                            />
                            <InputCheckbox
                                label="transform.field_fence_input"
                                value={data.fenceInput}
                                onChange={(v) => handleUpdate("fenceInput", v)}
                            />
                            <div className="text-xs text-base-content/70 mb-2 px-1">{t("transform.fence_input_hint")}</div>

                            <MaxTokensField value={data.maxTokens} onCommit={(v) => handleUpdate("maxTokens", v)} />
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
                            <div className="text-xs text-base-content/70 mb-2 px-1">{t("transform.live_settings_hint")}</div>
                        </>
                    )}

                    <div className="h-4" />

                    <ServiceButton
                        status={status}
                        onStart={() => service.start()}
                        onStop={() => service.stop()}
                    />
                    <div className="h-2" />
                    <InputCheckbox
                        label="transform.field_context_history"
                        value={data.contextHistory}
                        onChange={(v) => handleUpdate("contextHistory", v)}
                    />
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
                    <div className="text-xs font-medium text-base-content/80 mt-2 px-1">
                        {t("transform.context_history_hint")}
                    </div>
                </form>
            </Inspector.Content>
        </Inspector.Body>
    );
};

export default Inspector_Transform;
