import { FC, ReactNode, useEffect, useMemo, useRef, useState, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import {
    RiArrowLeftLine,
    RiArrowRightLine,
    RiCheckLine,
    RiCheckboxCircleFill,
    RiMessage2Fill,
    RiMicFill,
    RiRocketFill,
    RiUserVoiceFill,
} from "react-icons/ri";
import { SiDiscord, SiObsstudio, SiTwitch } from "react-icons/si";
import classNames from "classnames";
import { useTranslation } from "react-i18next";
import {
    ONBOARDING_AUTO,
    type OnboardingSttSave,
    lockableLanguages,
    onboardingLanguageDefault,
    planOnboardingSttSave,
    systemLanguageTags,
} from "@/core/services/stt/languageSuggestion";
import type { STT_State } from "@/core/services/stt/schema";
import { useLanguageName } from "@/core/ui/inspector/stt/common";
import { Services } from "@/services-registry";
import { KickIcon } from "./icons/KickIcon";

/**
 * The first-run setup wizard: five short steps (welcome, microphone and speakers, speech to text, where captions go,
 * done). It shows only for a new install (`state/firstRun.ts`: no settings file), never to an existing profile.
 * Skipping at any step before the last, or finishing, sets `onboardingComplete`; quitting in the middle shows it again.
 */

/** Steps, in order. The progress dots and the Skip button follow this list. */
export const ONBOARDING_STEPS = ["welcome", "audio", "stt", "send", "done"] as const;
export type OnboardingStep = (typeof ONBOARDING_STEPS)[number];
/**
 * The wizard covers the app but not its title bar: the window has no system decorations (`decorations: false`), so the
 * drag area and the minimise, maximise and close buttons are in `SigilHeader` (`h-10`, which is `top-10`) and stay usable.
 * `SigilLayout` makes everything below the header inert while the wizard shows.
 */
export const WIZARD_ROOT_CLASS = "fixed inset-x-0 bottom-0 top-10 z-[90] bg-base-300 flex items-center justify-center p-8";

/** Skip is offered on every step but the last. */
const LAST_SKIPPABLE_STEP = ONBOARDING_STEPS.length - 2;

/** Skipping only marks the wizard as done: nothing else is touched. */
export function skipOnboarding(): void {
    window.ApiServer.state.onboardingComplete = true;
}

/**
 * Ends the wizard. The Speech to Text card goes to the bottom bar (so "press Start" has something to press), and when
 * the user picked a destination its panel opens.
 */
export function finishOnboarding(tab?: Services): void {
    window.ApiServer.patchService("stt", s => { s.showActionButton = true; });
    window.ApiServer.state.onboardingComplete = true;
    if (tab) window.ApiServer.changeTab({ tab });
}

/** The devices the wizard offers, as `plugin:audio|list_input_devices` / `list_output_devices` return them (id == name). */
interface AudioDevice {
    id: string;
    name: string;
}

/**
 * What the Audio step writes. A device is written only when the user changed it: opening the wizard and pressing
 * Continue (or a device list that never loaded) must not replace what the profile already has. Choosing
 * "System default" is a choice too.
 */
export function planOnboardingAudioSave(
    saved: { input: string; output: string },
    picked: { input: string; output: string },
): { input?: string; output?: string } {
    const plan: { input?: string; output?: string } = {};
    if (picked.input !== saved.input) plan.input = picked.input;
    if (picked.output !== saved.output) plan.output = picked.output;
    return plan;
}

/**
 * What the speech step writes to the profile (`stt.data`). Marking the desktop default migration done stops it from
 * overriding this pick, and the built-in engine is a choice made here: a later switch to Chrome or Edge is not offered
 * the built-in engine again (`localOfferShown`).
 */
export function applyOnboardingSttSave(
    data: Pick<STT_State, "backend" | "desktopDefaultMigrated" | "localOfferShown"> & { local: Pick<STT_State["local"], "language" | "languageDecision"> },
    save: OnboardingSttSave,
): void {
    data.backend = save.backend;
    data.desktopDefaultMigrated = true;
    data.localOfferShown = true;
    // The language is only ever set by this answer: "Detect automatically" is stored as an explicit choice.
    if (save.localLanguage !== undefined) data.local.language = save.localLanguage;
    if (save.decision) data.local.languageDecision = save.decision;
}

/** The profile's current devices ("" = system default). Safe before the host state exists. */
function savedDevices(): { input: string; output: string } {
    const state = typeof window === "undefined" ? undefined : window.ApiServer?.state;
    return { input: state?.audioInputDeviceNative ?? "", output: state?.audioOutputDevice ?? "" };
}

/** The saved device stays selectable when it is not plugged in right now, so the list never silently shows another one. */
function withCurrent(devices: AudioDevice[], current: string): AudioDevice[] {
    return current && !devices.some((d) => d.id === current) ? [...devices, { id: current, name: current }] : devices;
}

/**
 * The heading of a step. Moving to a step moves focus here (it is not a tab stop of its own), so a keyboard or screen
 * reader user lands on the new step instead of staying on a button that has just left the screen.
 */
const StepHeading: FC<{ className: string; children: ReactNode }> = ({ className, children }) => {
    const ref = useRef<HTMLHeadingElement>(null);
    useEffect(() => { ref.current?.focus({ preventScroll: true }); }, []);
    return <h2 ref={ref} tabIndex={-1} className={classNames(className, "outline-none")}>{children}</h2>;
};

interface StepProps {
    onNext: () => void;
    onBack?: () => void;
}

const StepButtons: FC<{ onBack?: () => void; onNext: () => void; nextLabel: string; nextIcon?: ReactNode }> = ({ onBack, onNext, nextLabel, nextIcon }) => {
    const { t } = useTranslation();
    return (
        <div className="flex gap-3 w-full">
            <button type="button" onClick={onBack} className="btn btn-ghost flex-1 gap-2">
                <RiArrowLeftLine className="rtl:-scale-x-100" aria-hidden="true" /> {t("onboarding.back")}
            </button>
            <button type="button" onClick={onNext} className="btn btn-primary flex-1 gap-2">
                {nextLabel} {nextIcon ?? <RiArrowRightLine className="rtl:-scale-x-100" aria-hidden="true" />}
            </button>
        </div>
    );
};

// Step 1: Welcome
export const WelcomeStep: FC<StepProps> = ({ onNext }) => {
    const { t } = useTranslation();
    return (
        <div className="flex flex-col items-center text-center space-y-6">
            <div className="text-6xl text-primary" aria-hidden="true"><RiRocketFill /></div>
            <StepHeading className="text-3xl font-bold font-header">{t("onboarding.welcome_title")}</StepHeading>
            <p className="text-base-content/70 max-w-md">{t("onboarding.welcome_body")}</p>
            <button type="button" onClick={onNext} className="btn btn-primary btn-lg gap-2">
                {t("onboarding.get_started")} <RiArrowRightLine className="rtl:-scale-x-100" aria-hidden="true" />
            </button>
        </div>
    );
};

// Step 2: Microphone and speakers
export const AudioStep: FC<StepProps> = ({ onNext, onBack }) => {
    const { t } = useTranslation();
    const [saved] = useState(savedDevices);
    const [inputDevices, setInputDevices] = useState<AudioDevice[]>([]);
    const [outputDevices, setOutputDevices] = useState<AudioDevice[]>([]);
    const [selectedInput, setSelectedInput] = useState(saved.input);
    const [selectedOutput, setSelectedOutput] = useState(saved.output);

    // The lists load once the step is on screen; an unmounted step drops a late answer. A list that fails to load
    // leaves "System default" as the only choice, and then nothing is written.
    useEffect(() => {
        let cancelled = false;
        void Promise.allSettled([
            invoke<AudioDevice[]>("plugin:audio|list_input_devices"),
            invoke<AudioDevice[]>("plugin:audio|list_output_devices"),
        ]).then(([inputs, outputs]) => {
            if (cancelled) return;
            if (inputs.status === "fulfilled") setInputDevices(inputs.value);
            else console.error("Failed to load input devices", inputs.reason);
            if (outputs.status === "fulfilled") setOutputDevices(outputs.value);
            else console.error("Failed to load output devices", outputs.reason);
        });
        return () => { cancelled = true; };
    }, []);

    const saveAndContinue = () => {
        const plan = planOnboardingAudioSave(saved, { input: selectedInput, output: selectedOutput });
        const state = window.ApiServer.state;
        if (plan.input !== undefined) {
            // cpal device name from `plugin:audio|list_input_devices` (native STT default), not a Web Audio deviceId.
            // A deliberate pick (including "system default") ends the one-time legacy device migration.
            state.audioInputDeviceNative = plan.input;
            state.audioInputDeviceNativeMigrated = true;
        }
        if (plan.output !== undefined) state.audioOutputDevice = plan.output;
        onNext();
    };

    const picker = (id: string, label: string, value: string, devices: AudioDevice[], onChange: (v: string) => void) => (
        <div className="form-control w-full">
            <label className="label" htmlFor={id}><span className="label-text font-semibold">{label}</span></label>
            <select id={id} className="select select-bordered w-full" value={value} onChange={(e) => onChange(e.target.value)}>
                <option value="">{t("onboarding.audio_default")}</option>
                {withCurrent(devices, value).map((d) => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                ))}
            </select>
        </div>
    );

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary" aria-hidden="true"><RiMicFill /></div>
            <StepHeading className="text-2xl font-bold font-header">{t("onboarding.audio_title")}</StepHeading>
            <p className="text-base-content/70 text-center">{t("onboarding.audio_body")}</p>

            <div className="w-full space-y-4">
                {picker("onboarding-microphone", t("onboarding.audio_microphone"), selectedInput, inputDevices, setSelectedInput)}
                {picker("onboarding-speakers", t("onboarding.audio_speakers"), selectedOutput, outputDevices, setSelectedOutput)}
            </div>

            <StepButtons onBack={onBack} onNext={saveAndContinue} nextLabel={t("onboarding.continue")} />
        </div>
    );
};

/**
 * "What language do you speak?" for the built-in engine. Starts on the Windows language when Sigil can route it
 * (so most people only press Continue) and otherwise on "Detect automatically"; both stay choices in the list, and
 * nothing is stored until Continue.
 */
const SttLanguageQuestion: FC<{ value: string; onChange: (code: string) => void; fromWindows: boolean }> = ({ value, onChange, fromWindows }) => {
    const { t } = useTranslation();
    const languageName = useLanguageName();
    const options = useMemo(
        () => [...lockableLanguages()].map((code) => ({ code, label: languageName(code) })).sort((a, b) => a.label.localeCompare(b.label)),
        [languageName],
    );
    return (
        <div className="form-control w-full space-y-1" data-testid="onboarding-language">
            <label className="label" htmlFor="onboarding-stt-language">
                <span className="label-text font-semibold">{t("stt.onboarding_language_title")}</span>
            </label>
            <select
                id="onboarding-stt-language"
                className="select select-bordered w-full"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                aria-label={t("stt.onboarding_language_label")}
            >
                <option value={ONBOARDING_AUTO}>{t("stt.local_language_auto")}</option>
                {options.map((o) => (
                    <option key={o.code} value={o.code}>{o.label}</option>
                ))}
            </select>
            <p className="text-xs text-base-content/60">
                {t("stt.onboarding_language_hint")}
                {fromWindows && value !== ONBOARDING_AUTO ? ` ${t("stt.onboarding_language_system")}` : ""}
            </p>
        </div>
    );
};

// Step 3: Speech to Text. One engine, chosen for the user; the others stay in the engine list of the Speech panel.
export const SttStep: FC<StepProps> = ({ onNext, onBack }) => {
    const { t } = useTranslation();
    const windowsTags = useMemo(() => systemLanguageTags(), []);
    // The wizard preselects the Windows language, English included; "Detect automatically" when none is routable.
    const windowsDefault = useMemo(() => onboardingLanguageDefault(windowsTags), [windowsTags]);
    const windowsLanguage = windowsDefault === ONBOARDING_AUTO ? null : windowsDefault;
    const [language, setLanguage] = useState<string>(windowsDefault);

    const saveAndContinue = () => {
        const save = planOnboardingSttSave("local", language, Date.now());
        window.ApiServer.patchService("stt", s => applyOnboardingSttSave(s.data, save));
        onNext();
    };

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary" aria-hidden="true"><RiUserVoiceFill /></div>
            <StepHeading className="text-2xl font-bold font-header">{t("onboarding.stt_title")}</StepHeading>
            <p className="text-base-content/70 text-center">{t("onboarding.stt_body")}</p>

            <div className="w-full p-4 rounded-xl border-2 border-primary bg-primary/10 flex items-center gap-4" data-testid="onboarding-engine">
                <div className="text-2xl text-primary" aria-hidden="true"><RiRocketFill /></div>
                <div className="text-start">
                    <div className="font-semibold">{t("stt.provider_local")}</div>
                    <div className="text-xs text-base-content/60">{t("stt.local_onboarding_desc")}</div>
                </div>
                <RiCheckLine className="ms-auto text-xl text-primary" aria-hidden="true" />
            </div>

            <SttLanguageQuestion value={language} onChange={setLanguage} fromWindows={windowsLanguage !== null && language === windowsLanguage} />
            <p className="text-xs text-base-content/60 text-center">{t("onboarding.stt_other")}</p>

            <StepButtons onBack={onBack} onNext={saveAndContinue} nextLabel={t("onboarding.continue")} />
        </div>
    );
};

// Step 4: Where should captions go?
export const SendStep: FC<StepProps & { onPick: (tab: Services) => void }> = ({ onNext, onBack, onPick }) => {
    const { t } = useTranslation();
    const cards: { tab: Services; name: string; desc: string; icon: ReactNode }[] = [
        { tab: Services.vrc, name: "VRChat", desc: t("onboarding.send_vrc_desc"), icon: <RiMessage2Fill /> },
        { tab: Services.obs, name: "OBS Studio", desc: t("onboarding.send_obs_desc"), icon: <SiObsstudio /> },
    ];
    const others: { tab: Services; name: string; icon: ReactNode }[] = [
        { tab: Services.twitch, name: "Twitch", icon: <SiTwitch /> },
        { tab: Services.kick, name: "Kick", icon: <KickIcon /> },
        { tab: Services.discord, name: "Discord", icon: <SiDiscord /> },
    ];

    return (
        <div className="flex flex-col items-center space-y-6 w-full max-w-md">
            <div className="text-5xl text-primary" aria-hidden="true"><RiMessage2Fill /></div>
            <StepHeading className="text-2xl font-bold font-header">{t("onboarding.send_title")}</StepHeading>
            <p className="text-base-content/70 text-center">{t("onboarding.send_body")}</p>

            <div className="w-full space-y-3">
                {cards.map((card) => (
                    <button
                        type="button"
                        key={card.tab}
                        onClick={() => onPick(card.tab)}
                        className="w-full p-4 rounded-xl border border-base-content/10 flex items-center gap-4 hover:border-primary/30 transition-all"
                    >
                        <div className="text-2xl text-primary" aria-hidden="true">{card.icon}</div>
                        <div className="text-start">
                            <div className="font-semibold">{card.name}</div>
                            <div className="text-xs text-base-content/60">{card.desc}</div>
                        </div>
                        <RiArrowRightLine className="ms-auto text-base-content/30 rtl:-scale-x-100" aria-hidden="true" />
                    </button>
                ))}
            </div>

            <div className="flex flex-wrap justify-center gap-2">
                {others.map((o) => (
                    <button type="button" key={o.tab} onClick={() => onPick(o.tab)} className="btn btn-sm btn-ghost gap-2 border border-base-content/10">
                        <span aria-hidden="true">{o.icon}</span> {o.name}
                    </button>
                ))}
            </div>

            <StepButtons onBack={onBack} onNext={onNext} nextLabel={t("onboarding.finish")} nextIcon={<RiRocketFill />} />
        </div>
    );
};

// Step 5: Done
export const DoneStep: FC<{ onOpen: () => void }> = ({ onOpen }) => {
    const { t } = useTranslation();
    return (
        <div className="flex flex-col items-center text-center space-y-6">
            <div className="text-6xl text-success" aria-hidden="true"><RiCheckboxCircleFill /></div>
            <StepHeading className="text-3xl font-bold font-header">{t("onboarding.done_title")}</StepHeading>
            <p className="text-base-content/70 max-w-md">{t("onboarding.done_body")}</p>
            <button type="button" onClick={onOpen} className="btn btn-primary btn-lg gap-2">
                {t("onboarding.done_open")} <RiArrowRightLine className="rtl:-scale-x-100" aria-hidden="true" />
            </button>
        </div>
    );
};

// Main Onboarding Component
const Onboarding: FC<{ initialStep?: number }> = memo(({ initialStep = 0 }) => {
    const { t } = useTranslation();
    const [step, setStep] = useState(Math.min(Math.max(initialStep, 0), ONBOARDING_STEPS.length - 1));
    const total = ONBOARDING_STEPS.length;

    const steps = [
        <WelcomeStep key="welcome" onNext={() => setStep(1)} />,
        <AudioStep key="audio" onNext={() => setStep(2)} onBack={() => setStep(0)} />,
        <SttStep key="stt" onNext={() => setStep(3)} onBack={() => setStep(1)} />,
        <SendStep key="send" onNext={() => setStep(4)} onBack={() => setStep(2)} onPick={finishOnboarding} />,
        <DoneStep key="done" onOpen={() => finishOnboarding()} />,
    ];

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className={WIZARD_ROOT_CLASS}
            role="dialog"
            aria-modal="true"
            aria-label={t("onboarding.dialog_label")}
        >
            {/* Progress indicator: a bar to assistive technology (its label says "step n of 5"), dots to the eye */}
            <div
                className="absolute top-8 left-1/2 -translate-x-1/2 flex gap-2"
                role="progressbar"
                aria-label={t("onboarding.progress", { step: step + 1, total })}
                aria-valuemin={1}
                aria-valuemax={total}
                aria-valuenow={step + 1}
                aria-valuetext={t("onboarding.progress", { step: step + 1, total })}
            >
                {ONBOARDING_STEPS.map((id, i) => (
                    <div
                        key={id}
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
            {step <= LAST_SKIPPABLE_STEP && (
                <button
                    type="button"
                    onClick={skipOnboarding}
                    className="absolute bottom-8 text-base-content/50 hover:text-base-content transition-colors"
                >
                    {t("onboarding.skip")}
                </button>
            )}
        </motion.div>
    );
});

export default Onboarding;
