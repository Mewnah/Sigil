import { FC, useState, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { useSnapshot } from "valtio";
import {
    RiUserVoiceFill,
    RiMicFill,
    RiCheckLine,
    RiArrowRightLine,
    RiArrowLeftLine,
    RiRocketFill,
    RiVolumeUpFill,
} from "react-icons/ri";
import { SiTwitch, SiDiscord } from "react-icons/si";
import classNames from "classnames";

// Kick logo
const KickIcon: FC = () => (
    <svg width={24} height={24} viewBox="0 0 512 512" fill="currentColor">
        <path d="M115.2 0h76.8v204.8L345.6 0H448L268.8 230.4 460.8 512H345.6L192 281.6V512h-76.8V0z" />
    </svg>
);

interface StepProps {
    onNext: () => void;
    onBack?: () => void;
    isFirst?: boolean;
    isLast?: boolean;
}

// Step 1: Welcome
const WelcomeStep: FC<StepProps> = ({ onNext }) => (
    <div className="flex flex-col items-center text-center space-y-6">
        <div className="text-6xl">🚀</div>
        <h2 className="text-3xl font-bold font-header">
            Welcome to <span className="text-primary">Sigil</span>
        </h2>
        <p className="text-base-content/70 max-w-md">
            The ultimate streaming toolkit for VRChat creators. Let's get you set up in just a few steps.
        </p>
        <button onClick={onNext} className="btn btn-primary btn-lg gap-2">
            Get Started <RiArrowRightLine />
        </button>
    </div>
);

// Step 2: Audio Setup
const AudioStep: FC<StepProps> = ({ onNext, onBack }) => {
    const [inputDevices, setInputDevices] = useState<{ id: string; name: string }[]>([]);
    const [outputDevices, setOutputDevices] = useState<{ id: string; name: string }[]>([]);
    const [selectedInput, setSelectedInput] = useState("");
    const [selectedOutput, setSelectedOutput] = useState("");
    const [loading, setLoading] = useState(false);

    const loadDevices = async () => {
        setLoading(true);
        try {
            const inputs = await invoke<{ id: string; name: string }[]>("plugin:audio|list_input_devices");
            const outputs = await invoke<{ id: string; name: string }[]>("plugin:audio|list_output_devices");
            setInputDevices(inputs);
            setOutputDevices(outputs);
        } catch (e) {
            console.error("Failed to load devices", e);
        }
        setLoading(false);
    };

    useState(() => {
        loadDevices();
    });

    const saveAndContinue = () => {
        window.ApiServer.state.audioInputDevice = selectedInput;
        window.ApiServer.state.audioOutputDevice = selectedOutput;
        onNext();
    };

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary"><RiMicFill /></div>
            <h2 className="text-2xl font-bold font-header">Audio Setup</h2>
            <p className="text-base-content/70 text-center">
                Select your microphone and speakers for speech recognition and voice output.
            </p>

            <div className="w-full space-y-4">
                <div className="form-control w-full">
                    <label className="label"><span className="label-text font-semibold">Microphone</span></label>
                    <select
                        className="select select-bordered w-full"
                        value={selectedInput}
                        onChange={(e) => setSelectedInput(e.target.value)}
                        aria-label="Select microphone"
                    >
                        <option value="">System Default</option>
                        {inputDevices.map((d) => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </select>
                </div>

                <div className="form-control w-full">
                    <label className="label"><span className="label-text font-semibold">Speakers</span></label>
                    <select
                        className="select select-bordered w-full"
                        value={selectedOutput}
                        onChange={(e) => setSelectedOutput(e.target.value)}
                        aria-label="Select speakers"
                    >
                        <option value="">System Default</option>
                        {outputDevices.map((d) => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="flex gap-3 w-full">
                <button onClick={onBack} className="btn btn-ghost flex-1 gap-2">
                    <RiArrowLeftLine /> Back
                </button>
                <button onClick={saveAndContinue} className="btn btn-primary flex-1 gap-2">
                    Continue <RiArrowRightLine />
                </button>
            </div>
        </div>
    );
};

// Step 3: STT Selection
const SttStep: FC<StepProps> = ({ onNext, onBack }) => {
    const [selected, setSelected] = useState<"whisper" | "vosk" | "">("");

    const options = [
        { id: "whisper", name: "Whisper", desc: "Highly accurate, runs locally via Rust", icon: <RiUserVoiceFill /> },
        { id: "vosk", name: "Vosk", desc: "Fast offline recognition, lightweight", icon: <RiVolumeUpFill /> },
    ];

    const saveAndContinue = () => {
        // STT backend will be configured in the settings panel
        // This step just helps users understand their options
        onNext();
    };

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary"><RiUserVoiceFill /></div>
            <h2 className="text-2xl font-bold font-header">Speech Recognition</h2>
            <p className="text-base-content/70 text-center">
                Choose your speech-to-text engine. Both run locally for privacy.
            </p>

            <div className="w-full space-y-3">
                {options.map((opt) => (
                    <button
                        key={opt.id}
                        onClick={() => setSelected(opt.id as "whisper" | "vosk")}
                        className={classNames(
                            "w-full p-4 rounded-xl border-2 flex items-center gap-4 transition-all",
                            selected === opt.id
                                ? "border-primary bg-primary/10"
                                : "border-base-content/10 hover:border-primary/30"
                        )}
                    >
                        <div className="text-2xl text-primary">{opt.icon}</div>
                        <div className="text-left">
                            <div className="font-semibold">{opt.name}</div>
                            <div className="text-xs text-base-content/60">{opt.desc}</div>
                        </div>
                        {selected === opt.id && <RiCheckLine className="ms-auto text-xl text-primary" />}
                    </button>
                ))}
            </div>

            <div className="flex gap-3 w-full">
                <button onClick={onBack} className="btn btn-ghost flex-1 gap-2">
                    <RiArrowLeftLine /> Back
                </button>
                <button onClick={saveAndContinue} className="btn btn-primary flex-1 gap-2">
                    Continue <RiArrowRightLine />
                </button>
            </div>
        </div>
    );
};

// Step 4: Integrations
const IntegrationsStep: FC<StepProps> = ({ onNext, onBack }) => {
    const integrations = [
        { id: "twitch", name: "Twitch", icon: <SiTwitch />, desc: "Connect your Twitch chat" },
        { id: "kick", name: "Kick", icon: <KickIcon />, desc: "Connect your Kick chat" },
        { id: "discord", name: "Discord", icon: <SiDiscord />, desc: "Voice channel integration" },
    ];

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary"><SiTwitch /></div>
            <h2 className="text-2xl font-bold font-header">Connect Integrations</h2>
            <p className="text-base-content/70 text-center">
                You can set these up now or later from the dashboard.
            </p>

            <div className="w-full space-y-3">
                {integrations.map((int) => (
                    <button
                        key={int.id}
                        onClick={() => {
                            window.ApiServer.changeTab({ tab: int.id as any });
                            window.ApiServer.state.onboardingComplete = true;
                        }}
                        className="w-full p-4 rounded-xl border border-base-content/10 flex items-center gap-4 hover:border-primary/30 transition-all"
                    >
                        <div className="text-2xl">{int.icon}</div>
                        <div className="text-left">
                            <div className="font-semibold">{int.name}</div>
                            <div className="text-xs text-base-content/60">{int.desc}</div>
                        </div>
                        <RiArrowRightLine className="ms-auto text-base-content/30" />
                    </button>
                ))}
            </div>

            <div className="flex gap-3 w-full">
                <button onClick={onBack} className="btn btn-ghost flex-1 gap-2">
                    <RiArrowLeftLine /> Back
                </button>
                <button onClick={onNext} className="btn btn-primary flex-1 gap-2">
                    Finish Setup <RiRocketFill />
                </button>
            </div>
        </div>
    );
};

// Step 5: Accessibility
const AccessibilityStep: FC<StepProps> = ({ onNext, onBack }) => {
    const state = useSnapshot(window.ApiServer.state);
    const clientTheme = state.clientTheme;
    const isHighContrast = clientTheme === "streamer";

    const handleToggleHighContrast = () => {
        if (isHighContrast) {
            window.ApiServer.state.clientTheme = "sigil-dark";
        } else {
            window.ApiServer.state.clientTheme = "streamer";
        }
    };

    const ariaCheckedValue = isHighContrast ? "true" : "false";

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl">♿</div>
            <h2 className="text-2xl font-bold font-header">Accessibility</h2>
            <p className="text-base-content/70 text-center">
                Sigil is designed for everyone. Customize your experience.
            </p>

            <div className="w-full space-y-4">
                {/* High Contrast Toggle */}
                <button
                    onClick={handleToggleHighContrast}
                    className={classNames(
                        "w-full p-4 rounded-xl border-2 flex items-center gap-4 transition-all",
                        isHighContrast
                            ? "border-primary bg-primary/10"
                            : "border-base-content/10 hover:border-primary/30"
                    )}
                    role="switch"
                    aria-checked={ariaCheckedValue}
                    aria-label="Toggle high contrast mode"
                >
                    <div className="text-2xl">🔆</div>
                    <div className="text-left flex-1">
                        <div className="font-semibold">High Contrast Mode</div>
                        <div className="text-xs text-base-content/60">
                            Better visibility for streaming & low vision
                        </div>
                    </div>
                    <div className={classNames(
                        "w-12 h-6 rounded-full transition-colors relative",
                        isHighContrast ? "bg-primary" : "bg-base-content/20"
                    )}>
                        <div className={classNames(
                            "absolute top-1 w-4 h-4 rounded-full bg-white transition-all",
                            isHighContrast ? "left-7" : "left-1"
                        )} />
                    </div>
                </button>

                {/* Info Cards */}
                <div className="p-4 rounded-xl bg-base-100/50 border border-base-content/10">
                    <div className="text-sm font-medium mb-2">✨ Built-in Features</div>
                    <ul className="text-xs text-base-content/60 space-y-1">
                        <li>• Keyboard navigation support</li>
                        <li>• Screen reader friendly labels</li>
                        <li>• Large, clear buttons</li>
                        <li>• Customizable text sizes in Settings</li>
                    </ul>
                </div>
            </div>

            <div className="flex gap-3 w-full">
                <button onClick={onBack} className="btn btn-ghost flex-1 gap-2">
                    <RiArrowLeftLine /> Back
                </button>
                <button onClick={onNext} className="btn btn-primary flex-1 gap-2">
                    Continue <RiArrowRightLine />
                </button>
            </div>
        </div>
    );
};

// Complete Step
const CompleteStep: FC<{ onComplete: () => void }> = ({ onComplete }) => (
    <div className="flex flex-col items-center text-center space-y-6">
        <div className="text-6xl">🎉</div>
        <h2 className="text-3xl font-bold font-header">You're All Set!</h2>
        <p className="text-base-content/70 max-w-md">
            Sigil is ready to use. Start streaming with speech recognition, TTS, and more.
        </p>
        <button onClick={onComplete} className="btn btn-primary btn-lg gap-2">
            Launch Dashboard <RiArrowRightLine />
        </button>
    </div>
);

// Main Onboarding Component
const Onboarding: FC = memo(() => {
    const [step, setStep] = useState(0);

    const handleComplete = () => {
        window.ApiServer.state.onboardingComplete = true;
    };

    const steps = [
        <WelcomeStep key="welcome" onNext={() => setStep(1)} isFirst />,
        <AudioStep key="audio" onNext={() => setStep(2)} onBack={() => setStep(0)} />,
        <SttStep key="stt" onNext={() => setStep(3)} onBack={() => setStep(1)} />,
        <IntegrationsStep key="integrations" onNext={() => setStep(4)} onBack={() => setStep(2)} />,
        <AccessibilityStep key="accessibility" onNext={() => setStep(5)} onBack={() => setStep(3)} />,
        <CompleteStep key="complete" onComplete={handleComplete} />,
    ];

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-base-300 flex items-center justify-center p-8"
            role="dialog"
            aria-label="Sigil Setup Wizard"
        >
            {/* Progress indicator */}
            <div
                className="absolute top-8 left-1/2 -translate-x-1/2 flex gap-2"
                role="status"
                aria-label={`Setup progress: step ${step + 1} of 6`}
            >
                {[0, 1, 2, 3, 4, 5].map((i) => (
                    <div
                        key={i}
                        className={classNames(
                            "w-3 h-3 rounded-full transition-colors",
                            i <= step ? "bg-primary" : "bg-base-content/20"
                        )}
                        aria-hidden="true"
                    />
                ))}
            </div>

            {/* Step content */}
            <AnimatePresence mode="wait">
                <motion.div
                    key={step}
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -50 }}
                    transition={{ duration: 0.2 }}
                    className="w-full flex justify-center"
                >
                    {steps[step]}
                </motion.div>
            </AnimatePresence>

            {/* Skip button */}
            {step < 4 && (
                <button
                    onClick={handleComplete}
                    className="absolute bottom-8 text-base-content/50 hover:text-base-content transition-colors"
                >
                    Skip setup
                </button>
            )}
        </motion.div>
    );
});

export default Onboarding;
