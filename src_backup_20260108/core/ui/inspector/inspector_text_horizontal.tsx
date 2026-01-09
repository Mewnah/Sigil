import { FC, memo, useState, PropsWithChildren, ReactNode } from "react";
import { useGetState } from "@/client";
import { Element_TextState, FlexAlign, FontCase } from "@/client/elements/text/schema";
import { useUpdateElement } from "@/utils";
import { useSnapshot } from "valtio";
import { useTranslation } from "react-i18next";
import classNames from "classnames";
import { RgbaColorPicker } from "react-colorful";
import { GrTextAlignCenter, GrTextAlignLeft, GrTextAlignRight } from "react-icons/gr";
import { RiAlignBottom, RiAlignTop, RiAlignVertically, RiArrowDownSLine, RiArrowRightSLine } from "react-icons/ri";
import { TextEventSource } from "@/types";
import Dropdown from "../dropdown/Dropdown";
import TransformInput from "./components/transform-input";
import { InputTextSource, InputFile, InputCode, InputFont } from "./components/input";

// ============================================================================
// Compact Input Components for Horizontal Layout
// ============================================================================

interface CompactFieldProps {
    label: string;
    children: ReactNode;
    className?: string;
    span?: number;
}

const CompactField: FC<CompactFieldProps> = ({ label, children, className, span = 1 }) => {
    const { t } = useTranslation();
    return (
        <div className={classNames("flex flex-col gap-1", className)} style={span > 1 ? { gridColumn: `span ${span}` } : undefined}>
            <label className="text-[10px] uppercase tracking-wide text-white/40 font-medium truncate">
                {t(label)}
            </label>
            {children}
        </div>
    );
};

const CompactText: FC<{
    label: string;
    value: string | number;
    onChange: (value: string) => void;
    type?: string;
    min?: string;
    max?: string;
    step?: string;
    span?: number;
}> = memo(({ label, value, onChange, type = "text", span, ...props }) => (
    <CompactField label={label} span={span}>
        <input
            type={type}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="h-7 px-2 text-xs bg-white/5 border border-white/10 rounded focus:border-primary/50 outline-none font-mono w-full [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            {...props}
        />
    </CompactField>
));

const CompactColor: FC<{
    label: string;
    value: string;
    onChange: (value: string) => void;
}> = memo(({ label, value, onChange }) => {
    const parseRgba = (v: string) => {
        const match = v.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/);
        if (match) return { r: +match[1], g: +match[2], b: +match[3], a: +(match[4] ?? 1) };
        return { r: 255, g: 255, b: 255, a: 1 };
    };

    return (
        <CompactField label={label}>
            <Dropdown
                placement="top"
                content={
                    <div className="p-2">
                        <RgbaColorPicker
                            color={parseRgba(value)}
                            onChange={(c) => onChange(`rgba(${c.r}, ${c.g}, ${c.b}, ${c.a})`)}
                        />
                    </div>
                }
            >
                <button
                    className="h-7 w-full rounded border border-white/10 flex items-center gap-2 px-2"
                    style={{ backgroundColor: value }}
                >
                    <span className="text-[10px] font-mono text-white/70 truncate mix-blend-difference">
                        {value.slice(0, 16)}
                    </span>
                </button>
            </Dropdown>
        </CompactField>
    );
});

const CompactToggle: FC<{
    label: string;
    value: boolean;
    onChange: (value: boolean) => void;
}> = memo(({ label, value, onChange }) => {
    const { t } = useTranslation();
    return (
        <label className="flex items-center gap-2 cursor-pointer h-7">
            <input
                type="checkbox"
                checked={value}
                onChange={(e) => onChange(e.target.checked)}
                className="w-4 h-4 rounded bg-white/5 border border-white/20 accent-primary"
            />
            <span className="text-xs text-white/70 truncate">{t(label)}</span>
        </label>
    );
});

const CompactChips: FC<{
    label: string;
    value: string;
    onChange: (value: string) => void;
    options: { value: string; label: ReactNode }[];
}> = memo(({ label, value, onChange, options }) => (
    <CompactField label={label}>
        <div className="flex gap-0.5">
            {options.map((opt) => (
                <button
                    key={opt.value}
                    onClick={() => onChange(opt.value)}
                    className={classNames(
                        "flex-1 h-7 flex items-center justify-center rounded text-xs transition-colors",
                        value === opt.value
                            ? "bg-primary text-black"
                            : "bg-white/5 text-white/60 hover:bg-white/10"
                    )}
                >
                    {opt.label}
                </button>
            ))}
        </div>
    </CompactField>
));

const CompactRange: FC<{
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number;
}> = memo(({ label, value, onChange, min = 0, max = 100, step = 1 }) => (
    <CompactField label={label}>
        <div className="flex items-center gap-2">
            <input
                type="range"
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value))}
                min={min}
                max={max}
                step={step}
                title={label}
                className="flex-1 h-1 accent-primary"
            />
            <span className="text-xs text-white/50 font-mono w-8 text-right">{value}</span>
        </div>
    </CompactField>
));

// ============================================================================
// Horizontal Inspector Layout Components
// ============================================================================

const SectionGrid: FC<PropsWithChildren<{ columns?: number }>> = ({ children, columns = 4 }) => (
    <div
        className="grid gap-x-3 gap-y-2 p-3"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
    >
        {children}
    </div>
);

const AdvancedSection: FC<PropsWithChildren<{ title?: string; defaultOpen?: boolean }>> = ({ children, title = "Advanced Settings", defaultOpen = false }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="border-t border-white/10">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={classNames(
                    "w-full px-3 py-2 flex items-center gap-2 text-xs font-medium transition-all",
                    isOpen
                        ? "bg-primary/10 text-primary border-l-2 border-primary"
                        : "text-white/50 hover:text-white hover:bg-white/5"
                )}
            >
                {isOpen ? <RiArrowDownSLine className="text-sm" /> : <RiArrowRightSLine className="text-sm" />}
                <span>{title}</span>
            </button>
            {isOpen && <div className="bg-white/[0.02]">{children}</div>}
        </div>
    );
};

// ============================================================================
// Main Horizontal Text Inspector
// ============================================================================

const Inspector_TextHorizontal: FC<{ id: string }> = memo(({ id }) => {
    const { t } = useTranslation();
    const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
    const data: Element_TextState = useGetState(
        (state) => state.elements[id]?.scenes[activeScene]?.data as Element_TextState
    );
    const up = useUpdateElement<Element_TextState>(id);

    const [category, setCategory] = useState<"source" | "text" | "box" | "effects" | "css">("text");

    if (!data) return <div className="p-4 text-white/30 text-xs">Element not found</div>;

    // Check if synced with AI Transform
    const isSyncedWithAI = data.sourceMain === TextEventSource.transform_raw;
    const showSyncOption = data.sourceMain === TextEventSource.stt || isSyncedWithAI;

    const categories = ["source", "text", "box", "effects", "css"] as const;

    return (
        <div className="flex flex-col h-full text-white">
            {/* Category Tabs */}
            <div className="flex gap-1 px-3 pt-2 border-b border-white/5 flex-shrink-0 overflow-x-auto">
                {categories.map((cat) => (
                    <button
                        key={cat}
                        onClick={() => setCategory(cat)}
                        className={classNames(
                            "px-3 py-1.5 text-xs font-medium rounded-t transition-colors capitalize whitespace-nowrap",
                            category === cat
                                ? "bg-white/10 text-white border-b-2 border-primary"
                                : "text-white/50 hover:text-white/70"
                        )}
                    >
                        {cat}
                    </button>
                ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto">
                {/* SOURCE TAB */}
                {category === "source" && (
                    <>
                        {/* Primary Source Selection */}
                        <div className="p-3 border-b border-white/5">
                            <div className="text-[10px] uppercase tracking-wide text-white/40 font-medium mb-2">Text Source</div>
                            <div className="grid grid-cols-2 gap-3">
                                <InputTextSource
                                    label=""
                                    value={isSyncedWithAI ? TextEventSource.stt : data.sourceMain}
                                    onChange={(e) => up("sourceMain", e)}
                                />
                                <div className="flex flex-col gap-2 justify-center">
                                    {showSyncOption && (
                                        <CompactToggle
                                            label="Sync with AI Transform"
                                            value={isSyncedWithAI}
                                            onChange={(v) => up("sourceMain", v ? TextEventSource.transform_raw : TextEventSource.stt)}
                                        />
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Source Options Grid */}
                        <div className="p-3 border-b border-white/5">
                            <div className="text-[10px] uppercase tracking-wide text-white/40 font-medium mb-2">Options</div>
                            <div className="grid grid-cols-3 gap-x-4 gap-y-2">
                                <CompactToggle label="Preview Mode" value={data.previewMode} onChange={(v) => up("previewMode", v)} />
                                <CompactToggle label="Interim Results" value={data.sourceInterim} onChange={(v) => up("sourceInterim", v)} />
                                <CompactToggle label="Keyboard Input" value={data.sourceInputField} onChange={(v) => up("sourceInputField", v)} />
                            </div>
                        </div>

                        {/* Profanity Settings */}
                        <div className="p-3 border-b border-white/5">
                            <div className="text-[10px] uppercase tracking-wide text-white/40 font-medium mb-2">Profanity Filter</div>
                            <div className="max-w-xs">
                                <CompactText label="Mask Character" value={data.textProfanityMask} onChange={(v) => up("textProfanityMask", v)} />
                            </div>
                        </div>

                        {/* Transform Section */}
                        <AdvancedSection title="Transform (Position & Size)" defaultOpen={true}>
                            <div className="p-3">
                                <TransformInput id={id} />
                            </div>
                        </AdvancedSection>
                    </>
                )}

                {/* TEXT TAB */}
                {category === "text" && (
                    <>
                        <SectionGrid columns={4}>
                            <CompactText label="Font Size" value={data.textFontSize} onChange={(v) => up("textFontSize", v)} type="number" />
                            <CompactColor label="Text Color" value={data.textColor} onChange={(v) => up("textColor", v)} />
                            <CompactChips
                                label="H Align"
                                value={data.textAlignH}
                                onChange={(v) => up("textAlignH", v as FlexAlign)}
                                options={[
                                    { value: FlexAlign.start, label: <GrTextAlignLeft /> },
                                    { value: FlexAlign.center, label: <GrTextAlignCenter /> },
                                    { value: FlexAlign.end, label: <GrTextAlignRight /> },
                                ]}
                            />
                            <CompactChips
                                label="V Align"
                                value={data.textAlignV}
                                onChange={(v) => up("textAlignV", v as FlexAlign)}
                                options={[
                                    { value: FlexAlign.start, label: <RiAlignTop /> },
                                    { value: FlexAlign.center, label: <RiAlignVertically /> },
                                    { value: FlexAlign.end, label: <RiAlignBottom /> },
                                ]}
                            />
                        </SectionGrid>
                        <AdvancedSection title="Font & Style">
                            <SectionGrid columns={4}>
                                <div className="col-span-2">
                                    <InputFont label="Font Family" value={data.textFontFamily} onChange={(v) => up("textFontFamily", v)} />
                                </div>
                                <CompactRange label="Weight" value={parseInt(data.textFontWeight) || 400} onChange={(v) => up("textFontWeight", v.toString())} min={100} max={900} step={100} />
                                <CompactChips
                                    label="Case"
                                    value={data.textCase}
                                    onChange={(v) => up("textCase", v as FontCase)}
                                    options={[
                                        { value: FontCase.lowercase, label: "aa" },
                                        { value: FontCase.uppercase, label: "AA" },
                                        { value: FontCase.inherit, label: "-" },
                                    ]}
                                />
                                <CompactText label="Line Height" value={data.textLineHeight} onChange={(v) => up("textLineHeight", v)} type="number" step="0.1" />
                                <CompactColor label="Interim Color" value={data.textColorInterim} onChange={(v) => up("textColorInterim", v)} />
                                <CompactColor label="Profanity Color" value={data.textProfanityColor} onChange={(v) => up("textProfanityColor", v)} />
                                <CompactColor label="Profanity Interim" value={data.textProfanityInterimColor} onChange={(v) => up("textProfanityInterimColor", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Text Shadow">
                            <SectionGrid columns={4}>
                                <CompactText label="Shadow X" value={data.textShadowX} onChange={(v) => up("textShadowX", v)} type="number" />
                                <CompactText label="Shadow Y" value={data.textShadowY} onChange={(v) => up("textShadowY", v)} type="number" />
                                <CompactText label="Blur" value={data.textShadowZ} onChange={(v) => up("textShadowZ", v)} type="number" />
                                <CompactColor label="Shadow Color" value={data.textShadowColor} onChange={(v) => up("textShadowColor", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Text Outline">
                            <SectionGrid columns={4}>
                                <CompactText label="Outline Size" value={data.textStroke} onChange={(v) => up("textStroke", v)} type="number" step="0.1" />
                                <CompactColor label="Outline Color" value={data.textStrokeColor} onChange={(v) => up("textStrokeColor", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                    </>
                )}

                {/* BOX TAB */}
                {category === "box" && (
                    <>
                        <SectionGrid columns={4}>
                            <CompactToggle label="Auto Width" value={data.boxAutoWidth} onChange={(v) => up("boxAutoWidth", v)} />
                            <CompactToggle label="Auto Height" value={data.boxAutoHeight} onChange={(v) => up("boxAutoHeight", v)} />
                            <CompactText label="Padding" value={data.boxPadding} onChange={(v) => up("boxPadding", v)} type="number" />
                            <CompactColor label="Background" value={data.boxColor} onChange={(v) => up("boxColor", v)} />
                        </SectionGrid>
                        <AdvancedSection title="Box Alignment">
                            <SectionGrid columns={4}>
                                <CompactChips
                                    label="Box H Align"
                                    value={data.boxAlignH}
                                    onChange={(v) => up("boxAlignH", v as FlexAlign)}
                                    options={[
                                        { value: FlexAlign.start, label: <GrTextAlignLeft /> },
                                        { value: FlexAlign.center, label: <GrTextAlignCenter /> },
                                        { value: FlexAlign.end, label: <GrTextAlignRight /> },
                                    ]}
                                />
                                <CompactChips
                                    label="Box V Align"
                                    value={data.boxAlignV}
                                    onChange={(v) => up("boxAlignV", v as FlexAlign)}
                                    options={[
                                        { value: FlexAlign.start, label: <RiAlignTop /> },
                                        { value: FlexAlign.center, label: <RiAlignVertically /> },
                                        { value: FlexAlign.end, label: <RiAlignBottom /> },
                                    ]}
                                />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Border & Radius">
                            <SectionGrid columns={4}>
                                <CompactText label="Border Radius" value={data.boxBorderRadius} onChange={(v) => up("boxBorderRadius", v)} type="number" />
                                <CompactText label="Border Width" value={data.boxBorderWidth} onChange={(v) => up("boxBorderWidth", v)} type="number" />
                                <CompactColor label="Border Color" value={data.boxBorderColor} onChange={(v) => up("boxBorderColor", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Box Inner Padding">
                            <SectionGrid columns={4}>
                                <CompactText label="Top" value={data.boxScrollPaddingTop} onChange={(v) => up("boxScrollPaddingTop", v)} type="number" />
                                <CompactText label="Right" value={data.boxScrollPaddingRight} onChange={(v) => up("boxScrollPaddingRight", v)} type="number" />
                                <CompactText label="Bottom" value={data.boxScrollPaddingBottom} onChange={(v) => up("boxScrollPaddingBottom", v)} type="number" />
                                <CompactText label="Left" value={data.boxScrollPaddingLeft} onChange={(v) => up("boxScrollPaddingLeft", v)} type="number" />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Box Shadow">
                            <SectionGrid columns={4}>
                                <CompactText label="Shadow X" value={data.boxShadowX} onChange={(v) => up("boxShadowX", v)} type="number" />
                                <CompactText label="Shadow Y" value={data.boxShadowY} onChange={(v) => up("boxShadowY", v)} type="number" />
                                <CompactText label="Blur" value={data.boxShadowZ} onChange={(v) => up("boxShadowZ", v)} type="number" />
                                <CompactText label="Spread" value={data.boxShadowSpread} onChange={(v) => up("boxShadowSpread", v)} type="number" />
                                <CompactColor label="Shadow Color" value={data.boxShadowColor} onChange={(v) => up("boxShadowColor", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Background Image">
                            <SectionGrid columns={2}>
                                <InputFile type="image" label="Background Image" value={data.boxImageFileId} onChange={(v) => up("boxImageFileId", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                    </>
                )}

                {/* EFFECTS TAB */}
                {category === "effects" && (
                    <>
                        <SectionGrid columns={4}>
                            <CompactToggle label="Animation" value={data.animateEnable} onChange={(v) => up("animateEnable", v)} />
                            <CompactText label="Clear After (s)" value={data.behaviorClearTimer} onChange={(v) => up("behaviorClearTimer", parseFloat(v) || 0)} type="number" />
                            <CompactToggle label="Audio" value={data.soundEnable} onChange={(v) => up("soundEnable", v)} />
                            <CompactRange label="Volume" value={data.soundVolume} onChange={(v) => up("soundVolume", v)} min={0} max={1} step={0.01} />
                        </SectionGrid>
                        <AdvancedSection title="Animation Timing">
                            <SectionGrid columns={4}>
                                <CompactText label="Char Delay (ms)" value={data.animateDelayChar} onChange={(v) => up("animateDelayChar", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Word Delay (ms)" value={data.animateDelayWord} onChange={(v) => up("animateDelayWord", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Sentence Delay (ms)" value={data.animateDelaySentence} onChange={(v) => up("animateDelaySentence", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Clear Delay (ms)" value={data.behaviorClearDelay} onChange={(v) => up("behaviorClearDelay", parseFloat(v) || 0)} type="number" />
                                <CompactToggle label="Emit CSS Event" value={data.animateEvent} onChange={(v) => up("animateEvent", v)} />
                                <CompactToggle label="Last Sentence Only" value={data.behaviorLastSentence} onChange={(v) => up("behaviorLastSentence", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Audio Files">
                            <SectionGrid columns={2}>
                                <InputFile type="audio" label="Typing Sound" value={data.soundFile} onChange={(v) => up("soundFile", v)} />
                                <InputFile type="audio" label="New Sentence Sound" value={data.soundFileNewSentence} onChange={(v) => up("soundFileNewSentence", v)} />
                                <InputFile type="audio" label="Show Sound" value={data.soundFileOnShow} onChange={(v) => up("soundFileOnShow", v)} />
                                <InputFile type="audio" label="Hide Sound" value={data.soundFileOnHide} onChange={(v) => up("soundFileOnHide", v)} />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Audio Params">
                            <SectionGrid columns={4}>
                                <CompactText label="Detune Min" value={data.soundDetuneMin} onChange={(v) => up("soundDetuneMin", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Detune Max" value={data.soundDetuneMax} onChange={(v) => up("soundDetuneMax", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Playback Min" value={data.soundPlaybackMin} onChange={(v) => up("soundPlaybackMin", parseFloat(v) || 0)} type="number" />
                                <CompactText label="Playback Max" value={data.soundPlaybackMax} onChange={(v) => up("soundPlaybackMax", parseFloat(v) || 0)} type="number" />
                            </SectionGrid>
                        </AdvancedSection>
                        <AdvancedSection title="Particles">
                            <SectionGrid columns={4}>
                                <CompactToggle label="Enable Particles" value={data.particlesEnable} onChange={(v) => up("particlesEnable", v)} />
                            </SectionGrid>
                            <SectionGrid columns={3}>
                                <InputFile type="image" label="Particle 1" value={data.particlesSpriteFileIdFirst} onChange={(v) => up("particlesSpriteFileIdFirst", v)} />
                                <InputFile type="image" label="Particle 2" value={data.particlesSpriteFileIdSecond} onChange={(v) => up("particlesSpriteFileIdSecond", v)} />
                                <InputFile type="image" label="Particle 3" value={data.particlesSpriteFileIdThird} onChange={(v) => up("particlesSpriteFileIdThird", v)} />
                            </SectionGrid>
                            <SectionGrid columns={4}>
                                <CompactText label="Count Min" value={data.particlesCountMin} onChange={(v) => up("particlesCountMin", v)} type="number" />
                                <CompactText label="Count Max" value={data.particlesCountMax} onChange={(v) => up("particlesCountMax", v)} type="number" />
                                <CompactText label="Duration Min" value={data.particlesDurationMin} onChange={(v) => up("particlesDurationMin", v)} type="number" />
                                <CompactText label="Duration Max" value={data.particlesDurationMax} onChange={(v) => up("particlesDurationMax", v)} type="number" />
                                <CompactText label="Dir X Min" value={data.particlesDirectionXMin} onChange={(v) => up("particlesDirectionXMin", v)} type="number" />
                                <CompactText label="Dir X Max" value={data.particlesDirectionXMax} onChange={(v) => up("particlesDirectionXMax", v)} type="number" />
                                <CompactText label="Dir Y Min" value={data.particlesDirectionYMin} onChange={(v) => up("particlesDirectionYMin", v)} type="number" />
                                <CompactText label="Dir Y Max" value={data.particlesDirectionYMax} onChange={(v) => up("particlesDirectionYMax", v)} type="number" />
                                <CompactText label="Scale Min" value={data.particlesScaleMin} onChange={(v) => up("particlesScaleMin", v)} type="number" />
                                <CompactText label="Scale Max" value={data.particlesScaleMax} onChange={(v) => up("particlesScaleMax", v)} type="number" />
                                <CompactText label="Rotation Min" value={data.particlesRotationMin} onChange={(v) => up("particlesRotationMin", v)} type="number" />
                                <CompactText label="Rotation Max" value={data.particlesRotationMax} onChange={(v) => up("particlesRotationMax", v)} type="number" />
                            </SectionGrid>
                        </AdvancedSection>
                    </>
                )}

                {/* CSS TAB */}
                {category === "css" && (
                    <div className="p-3">
                        <InputCode language="css" label="Custom CSS" value={data.css} onChange={(v) => up("css", v || "")} />
                    </div>
                )}
            </div>
        </div>
    );
});

Inspector_TextHorizontal.displayName = "Inspector_TextHorizontal";
export default Inspector_TextHorizontal;
