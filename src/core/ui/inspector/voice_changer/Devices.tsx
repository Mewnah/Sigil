// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Input / output / monitor device selects and the devices section.
import { useEffect, useMemo, useState, FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { invoke } from "@tauri-apps/api/core";
import { InputContainer, InputSelect } from "../components/input";
import { VcDetails, VcSubHeader } from "./common";

interface AudioDevice {
    id: string;
    name: string;
}

/** Device list of one direction; `null` until the first reply. */
export function useAudioDevices(kind: "input" | "output"): AudioDevice[] | null {
    const [devices, setDevices] = useState<AudioDevice[] | null>(null);
    useEffect(() => {
        let cancelled = false;
        invoke<AudioDevice[]>(kind === "input" ? "plugin:audio|list_input_devices" : "plugin:audio|list_output_devices")
            .then((d) => {
                if (!cancelled) setDevices(Array.isArray(d) ? d : []);
            })
            .catch((e) => {
                console.error(e);
                if (!cancelled) setDevices([]);
            });
        return () => {
            cancelled = true;
        };
    }, [kind]);
    return devices;
}

/**
 * Options for the listed devices, plus the saved one when it is not connected right now (so a restored
 * setting is shown by name instead of a blank select, and is kept until the user picks another).
 */
export function useDeviceOptions(devices: AudioDevice[] | null, saved: string, builtIn: readonly string[]) {
    const { t } = useTranslation();
    return useMemo(() => {
        const listed = (devices ?? []).map((d) => ({ label: d.name, value: d.id }));
        const missing = devices !== null && saved !== "" && !builtIn.includes(saved) && !listed.some((o) => o.value === saved);
        return missing ? [...listed, { label: t("voice_changer.device_not_connected", { name: saved }), value: saved }] : listed;
    }, [devices, saved, builtIn, t]);
}

/** Select value of "system default" (Radix Select shows its placeholder for an empty value). */
const INPUT_DEVICE_DEFAULT = "__vc_input_default__";
const NO_BUILT_IN: readonly string[] = [];

export const VoiceInputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
    labelKey: string;
    defaultLabelKey: string;
}> = ({ value, onChange, labelKey, defaultLabelKey }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("input");
    const options = useDeviceOptions(devices, value, NO_BUILT_IN);

    return (
        <InputSelect
            label={labelKey}
            value={value || INPUT_DEVICE_DEFAULT}
            onValueChange={(v) => onChange(v === INPUT_DEVICE_DEFAULT ? "" : v)}
            options={[{ label: t(defaultLabelKey), value: INPUT_DEVICE_DEFAULT }, ...options]}
        />
    );
};

const MONITOR_OUTPUT_OFF = "__vc_monitor_off__";
const OUTPUT_BUILT_IN: readonly string[] = ["default"];

export const VoiceMonitorOutputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
}> = ({ value, onChange }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("output");
    const options = useDeviceOptions(devices, value, OUTPUT_BUILT_IN);

    const selectValue = value === "" ? MONITOR_OUTPUT_OFF : value || "default";

    return (
        <InputSelect
            label="voice_changer.monitor_device"
            value={selectValue}
            onValueChange={(v) => {
                if (v === MONITOR_OUTPUT_OFF) onChange("");
                else onChange(v);
            }}
            options={[
                { label: t("voice_changer.monitor_off"), value: MONITOR_OUTPUT_OFF },
                { label: t("voice_changer.device_system_default"), value: "default" },
                ...options,
            ]}
        />
    );
};

export const VoiceOutputDeviceSelect: FC<{
    value: string;
    onChange: (device: string) => void;
    labelKey: string;
    defaultLabelKey: string;
}> = ({ value, onChange, labelKey, defaultLabelKey }) => {
    const { t } = useTranslation();
    const devices = useAudioDevices("output");
    const options = useDeviceOptions(devices, value || "default", OUTPUT_BUILT_IN);

    return (
        <InputSelect
            label={labelKey}
            value={value || "default"}
            onValueChange={onChange}
            options={[{ label: t(defaultLabelKey), value: "default" }, ...options]}
        />
    );
};

/** Devices section: input, output and the optional monitor output. */
export const DevicesSection: FC = () => {
    const voiceChanger = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    return (
        <>
            <VcSubHeader titleKey="voice_changer.section_devices" infoKey="voice_changer.section_devices_info" />
            <VoiceInputDeviceSelect
                value={voiceChanger.inputDevice}
                onChange={(device) => void vcApi.setInputDevice(device)}
                labelKey="voice_changer.device_input"
                defaultLabelKey="voice_changer.device_system_default"
            />
            <VoiceOutputDeviceSelect
                value={voiceChanger.outputDevice}
                onChange={async (device) => {
                    window.ApiServer.voiceChanger.state.voiceChanger.outputDevice = device;
                    await window.ApiServer.voiceChanger.syncParamsAndRestartIfRunning();
                }}
                labelKey="voice_changer.device_output"
                defaultLabelKey="voice_changer.device_system_default"
            />
            <VcDetails titleKey="voice_changer.monitor_section">
                <VoiceMonitorOutputDeviceSelect
                    value={voiceChanger.monitorOutputDevice}
                    onChange={async (device) => {
                        vcApi.state.voiceChanger.monitorOutputDevice = device;
                        await vcApi.syncParamsAndRestartIfRunning();
                    }}
                />
                <InputContainer label="voice_changer.monitor_gain">
                    <input
                        type="number"
                        className="input input-bordered input-sm w-full"
                        min={0}
                        max={4}
                        step={0.1}
                        value={voiceChanger.monitorGain}
                        onChange={(e) => {
                            const x = parseFloat(e.target.value);
                            if (!Number.isNaN(x)) {
                                vcApi.state.voiceChanger.monitorGain = Math.min(4, Math.max(0, x));
                            }
                        }}
                        onBlur={() => void vcApi.syncParams()}
                    />
                </InputContainer>
            </VcDetails>
        </>
    );
};
