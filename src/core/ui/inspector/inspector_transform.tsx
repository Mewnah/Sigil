import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { Service_Transform_Schema, Transform_Backends } from "@/core/services/transform/schema";
import { ServiceNetworkState } from "@/types";
import Inspector from "./components";
import ServiceButton from "../service-button";
import { InputCheckbox, InputSelect, InputText, InputCode } from "./components/input";
import { useState } from "react";

const Inspector_Transform = () => {
    const { t } = useTranslation();
    const transformState = useSnapshot(window.ApiServer.state.services.transform);
    const { data } = transformState;
    const service = window.ApiServer.transform;

    const { status } = useSnapshot(window.ApiServer.transform.serviceState);

    const handleUpdate = <K extends keyof typeof data>(key: K, value: typeof data[K]) => {
        window.ApiServer.patchService("transform", (state) => {
            state.data[key] = value;
        });
        // Restart service if critical config changes AND it is currently running
        if (status === ServiceNetworkState.connected && ["backend", "provider", "model", "customModel", "openaiModel", "openRouterModel"].includes(key)) {
            service.start();
        }
    };

    return (
        <Inspector.Body>
            <Inspector.Header>AI Transform</Inspector.Header>
            <Inspector.Content>
                <form onSubmit={e => e.preventDefault()} className="contents">
                    <input type="text" name="username" autoComplete="username" className="hidden" readOnly placeholder="username" />



                    <Inspector.SubHeader>Transformation Engine</Inspector.SubHeader>
                    <InputSelect
                        label="Engine"
                        value={data.backend}
                        options={[
                            { label: "LLM", value: Transform_Backends.openai },
                            { label: "Debug (Uppercase)", value: Transform_Backends.dummy },
                        ]}
                        onValueChange={(v) => handleUpdate("backend", v as Transform_Backends)}
                    />

                    {data.backend === Transform_Backends.openai && (
                        <>
                            <InputSelect
                                label="AI Provider"
                                value={data.provider}
                                options={[
                                    { label: "OpenAI (Cloud)", value: "openai" },
                                    { label: "OpenRouter", value: "openrouter" },
                                    { label: "Custom", value: "custom" },
                                ]}
                                onValueChange={(v) => handleUpdate("provider", v as any)}
                            />

                            {/* OpenAI Config */}
                            {data.provider === "openai" && (
                                <>
                                    <InputText
                                        label="API Key"
                                        type="password"
                                        autoComplete="new-password"
                                        value={data.openaiKey}
                                        placeholder="sk-..."
                                        onChange={(e) => handleUpdate("openaiKey", e.target.value)}
                                    />

                                    <InputText
                                        label="API URL"
                                        value={data.openaiUrl}
                                        placeholder="https://api.openai.com/v1"
                                        onChange={(e) => handleUpdate("openaiUrl", e.target.value)}
                                    />
                                    <InputText
                                        label="Model Name"
                                        value={data.openaiModel}
                                        placeholder="gpt-3.5-turbo"
                                        onChange={(e) => handleUpdate("openaiModel", e.target.value)}
                                    />
                                </>
                            )}

                            {/* OpenRouter Config */}
                            {data.provider === "openrouter" && (
                                <>
                                    <InputText
                                        label="API Key"
                                        type="password"
                                        autoComplete="new-password"
                                        value={data.openRouterKey}
                                        placeholder="sk-or-..."
                                        onChange={(e) => handleUpdate("openRouterKey", e.target.value)}
                                    />

                                    <InputText
                                        label="API URL"
                                        value={data.openRouterUrl}
                                        placeholder="https://openrouter.ai/api/v1"
                                        onChange={(e) => handleUpdate("openRouterUrl", e.target.value)}
                                    />
                                    <InputText
                                        label="Model Name"
                                        value={data.openRouterModel}
                                        placeholder="mistralai/mistral-7b-instruct:free"
                                        onChange={(e) => handleUpdate("openRouterModel", e.target.value)}
                                    />
                                </>
                            )}

                            {/* Custom Config */}
                            {data.provider === "custom" && (
                                <>
                                    <InputText
                                        label="API Key"
                                        type="password"
                                        autoComplete="new-password"
                                        value={data.customKey}
                                        placeholder="sk-..."
                                        onChange={(e) => handleUpdate("customKey", e.target.value)}
                                    />

                                    <InputText
                                        label="API URL"
                                        value={data.customUrl}
                                        placeholder="http://localhost:11434/v1"
                                        onChange={(e) => handleUpdate("customUrl", e.target.value)}
                                    />
                                    <InputText
                                        label="Model Name"
                                        value={data.customModel}
                                        placeholder="llama3"
                                        onChange={(e) => handleUpdate("customModel", e.target.value)}
                                    />
                                </>
                            )}

                            <div className="h-4" />
                            <InputCode
                                label="System Prompt"
                                language="text"
                                value={data.systemPrompt}
                                onChange={(v) => handleUpdate("systemPrompt", v || "")}
                                placeholder="Example: Rewrite the input text to be more professional, correcting any grammar mistakes."
                            />
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
                </form>
            </Inspector.Content>
        </Inspector.Body>
    );
};

export default Inspector_Transform;
