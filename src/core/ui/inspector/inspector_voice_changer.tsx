import { FC, useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import { invoke } from "@tauri-apps/api/core";
import { InputSelect, InputContainer } from "./components/input";
import { RiMagicFill, RiPlayFill, RiStopFill } from "react-icons/ri";
import Tooltip from "../dropdown/Tooltip";

interface AudioDevice {
    id: string;
    name: string;
}

const VoiceInputDeviceSelect: FC<{ value: string; onChange: (device: string) => void }> = ({ value, onChange }) => {
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        invoke<AudioDevice[]>("plugin:audio|list_input_devices")
            .then(setDevices)
            .catch(console.error);
    }, []);

    return (
        <InputSelect
            label="Input Device"
            value={value || ""}
            onValueChange={onChange}
            options={[
                { label: "System Default", value: "" },
                ...devices.map(d => ({ label: d.name, value: d.id }))
            ]}
        />
    );
};

const VoiceOutputDeviceSelect: FC<{ value: string; onChange: (device: string) => void }> = ({ value, onChange }) => {
    const [devices, setDevices] = useState<AudioDevice[]>([]);

    useEffect(() => {
        invoke<AudioDevice[]>("plugin:audio|list_output_devices")
            .then(setDevices)
            .catch(console.error);
    }, []);

    return (
        <InputSelect
            label="Output Device"
            value={value || "default"}
            onValueChange={onChange}
            options={[
                { label: "System Default", value: "default" },
                ...devices.map(d => ({ label: d.name, value: d.id }))
            ]}
        />
    );
};

export const Inspector_VoiceChanger: FC = () => {
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const voiceChanger = snap.voiceChanger;
    const isRunning = snap.isRunning;
    const presets = snap.presets;

    const handleToggle = () => {
        window.ApiServer.voiceChanger.toggle();
    };

    const handlePitchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        window.ApiServer.voiceChanger.setPitch(value);
    };

    const handleFormantChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseFloat(e.target.value);
        window.ApiServer.voiceChanger.setFormant(value);
    };

    const handlePresetChange = (presetId: string) => {
        window.ApiServer.voiceChanger.applyPreset(presetId);
    };

    const handleVocoderWindowChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const value = parseInt(e.target.value, 10);
        window.ApiServer.voiceChanger.setVocoderWindowMs(value);
    };

    const handleVocoderOversampleChange = (value: string) => {
        const n = parseInt(value, 10) as 4 | 8 | 16 | 32;
        if (n === 4 || n === 8 || n === 16 || n === 32) {
            window.ApiServer.voiceChanger.setVocoderOversample(n);
        }
    };

    return (
        <div className="flex flex-col gap-4 p-4 overflow-y-auto h-full">
            {/* Header with Start/Stop Button */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <RiMagicFill className="text-primary text-xl" />
                    <h2 className="text-lg font-semibold">Voice Changer</h2>
                </div>
                <Tooltip content={isRunning ? "Stop Voice Changer" : "Start Voice Changer"}>
                    <button
                        onClick={handleToggle}
                        className={`btn btn-sm ${isRunning ? "btn-error" : "btn-primary"}`}
                    >
                        {isRunning ? (
                            <>
                                <RiStopFill className="mr-1" /> Stop
                            </>
                        ) : (
                            <>
                                <RiPlayFill className="mr-1" /> Start
                            </>
                        )}
                    </button>
                </Tooltip>
            </div>

            {/* Status Indicator */}
            <div className={`rounded-lg p-3 ${isRunning ? "bg-success/20" : "bg-base-200"}`}>
                <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isRunning ? "bg-success animate-pulse" : "bg-base-content/30"}`} />
                    <span className="text-sm">
                        {isRunning ? "Voice changer is active" : "Voice changer is stopped"}
                    </span>
                </div>
            </div>

            {/* Preset Selection */}
            <InputSelect
                label="Preset"
                value={voiceChanger.preset}
                onValueChange={handlePresetChange}
                options={presets.map(p => ({ label: p.name, value: p.id }))}
            />

            {/* Pitch Slider */}
            <InputContainer label="Pitch">
                <div className="flex flex-col w-full gap-1">
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-base-content/60">
                            {voiceChanger.pitch > 0 ? "+" : ""}{voiceChanger.pitch.toFixed(1)} semitones
                        </span>
                    </div>
                    <input
                        type="range"
                        min={-12}
                        max={12}
                        step={0.5}
                        value={voiceChanger.pitch}
                        onChange={handlePitchChange}
                        className="range range-sm range-primary"
                    />
                    <div className="flex justify-between text-xs text-base-content/40">
                        <span>Deeper</span>
                        <span>Higher</span>
                    </div>
                </div>
            </InputContainer>

            {/* Formant Slider */}
            <InputContainer label="Formant Shift">
                <div className="flex flex-col w-full gap-1">
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-base-content/60">
                            {voiceChanger.formant > 0 ? "+" : ""}{voiceChanger.formant.toFixed(2)}
                        </span>
                    </div>
                    <input
                        type="range"
                        min={-1}
                        max={1}
                        step={0.05}
                        value={voiceChanger.formant}
                        onChange={handleFormantChange}
                        className="range range-sm range-primary"
                    />
                    <div className="flex justify-between text-xs text-base-content/40">
                        <span>Darker tone</span>
                        <span>Bright tone</span>
                    </div>
                </div>
            </InputContainer>

            <div className="divider text-xs text-base-content/40">Phase vocoder (CPU vs quality)</div>

            <InputContainer label="Analysis window">
                <div className="flex flex-col w-full gap-1">
                    <div className="flex justify-between items-center">
                        <span className="text-xs text-base-content/60">{voiceChanger.vocoderWindowMs} ms</span>
                    </div>
                    <input
                        type="range"
                        min={30}
                        max={60}
                        step={1}
                        value={voiceChanger.vocoderWindowMs}
                        onChange={handleVocoderWindowChange}
                        className="range range-sm range-primary"
                    />
                    <div className="flex justify-between text-xs text-base-content/40">
                        <span>Lower latency</span>
                        <span>Often smoother</span>
                    </div>
                </div>
            </InputContainer>

            <InputSelect
                label="Oversampling"
                value={String(voiceChanger.vocoderOversample)}
                onValueChange={handleVocoderOversampleChange}
                options={[
                    { label: "4 (lowest CPU)", value: "4" },
                    { label: "8 (balanced)", value: "8" },
                    { label: "16 (higher quality)", value: "16" },
                    { label: "32 (heaviest)", value: "32" },
                ]}
            />

            {/* Device Selection */}
            <div className="divider text-xs text-base-content/40">Audio Devices</div>

            <VoiceInputDeviceSelect
                value={voiceChanger.inputDevice}
                onChange={(device) => {
                    window.ApiServer.voiceChanger.state.voiceChanger.inputDevice = device;
                }}
            />

            <VoiceOutputDeviceSelect
                value={voiceChanger.outputDevice}
                onChange={async (device) => {
                    window.ApiServer.voiceChanger.state.voiceChanger.outputDevice = device;
                    await window.ApiServer.voiceChanger.syncParams();
                }}
            />

            {/* Info */}
            <div className="text-xs text-base-content/50 bg-base-200 rounded-lg p-3 mt-2">
                <p className="font-medium mb-1">Tips</p>
                <ul className="list-disc list-inside space-y-1">
                    <li>Start runs pitch processing; Stop turns it off. Adjust pitch while running as needed.</li>
                    <li>Pitch uses a phase vocoder (constant duration), then the app matches your output device sample rate.</li>
                    <li>Formant adjusts low/high balance (brightness), not true spectral formants.</li>
                    <li>
                        Window length and oversampling control CPU load and quality. If the app stutters, try a shorter window
                        or oversampling 4 or 8.
                    </li>
                    <li>
                        This engine only exposes pitch, formant tilt, and these vocoder settings. Other effects (input/output
                        gain, dry/wet mix, noise gate, EQ, or true spectral formant shifting) are not implemented yet; a future
                        upgrade path could add a higher-end stretcher (for example Signalsmith or Rubber Band).
                    </li>
                </ul>
            </div>
        </div>
    );
};
