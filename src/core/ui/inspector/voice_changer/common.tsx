// Voice-changer inspector (P3-S0 split of inspector_voice_changer.tsx; owned by P3-G).
// Shared building blocks: info popover, slider row, section headers, collapsible groups, draft number input,
// path field, and the structural-change helper.
import type { VoiceChangerState } from "@/core/services/voice_changer/schema";
import classNames from "classnames";
import { FC, ReactNode, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { RiInformationLine } from "react-icons/ri";
import Inspector from "../components";

/** Set a field that needs an engine restart, then sync (and restart when running). */
export const patchVcStructural = async <K extends keyof VoiceChangerState>(key: K, v: VoiceChangerState[K]) => {
    window.ApiServer.voiceChanger.state.voiceChanger[key] = v;
    await window.ApiServer.voiceChanger.syncParamsAndRestartIfRunning();
};

/** Scroll/clipping region for the service inspector panel (SimpleBar + inspector content width). */
export function getInspectorPopoverClip(el: HTMLElement): DOMRect {
    const scroll = el.closest(".simplebar-content-wrapper");
    if (scroll instanceof HTMLElement) return scroll.getBoundingClientRect();
    let p: HTMLElement | null = el.parentElement;
    while (p) {
        const cn = typeof p.className === "string" ? p.className : "";
        if (cn.includes("22rem") && cn.includes("flex-col")) {
            return p.getBoundingClientRect();
        }
        p = p.parentElement;
    }
    const bar = el.closest(".simplebar-wrapper") ?? el.closest(".overflow-y-auto");
    if (bar instanceof HTMLElement) return bar.getBoundingClientRect();
    return new DOMRect(0, 0, window.innerWidth, window.innerHeight);
}

export const InspectorInfoButton: FC<{ text: string }> = ({ text }) => {
    const { t } = useTranslation();
    const [show, setShow] = useState(false);
    const wrapRef = useRef<HTMLDivElement>(null);
    const [box, setBox] = useState<{ top: number; left: number; width: number; maxHeight: number } | null>(null);

    useLayoutEffect(() => {
        if (!show) {
            setBox(null);
            return;
        }
        const el = wrapRef.current;
        if (!el) return;
        const place = () => {
            const br = el.getBoundingClientRect();
            const clip = getInspectorPopoverClip(el);
            const margin = 8;
            const width = Math.max(160, Math.min(280, clip.width - margin * 2));
            let left = br.right - width;
            left = Math.max(clip.left + margin, Math.min(left, clip.right - margin - width));

            const capH = Math.min(320, Math.floor(window.innerHeight * 0.55));
            let top = br.bottom + margin;
            let maxHeight = Math.min(capH, clip.bottom - margin - top);
            if (maxHeight < 100) {
                maxHeight = Math.min(capH, br.top - clip.top - margin * 2);
                top = Math.max(clip.top + margin, br.top - margin - maxHeight);
                maxHeight = Math.min(maxHeight, clip.bottom - margin - top);
            }
            maxHeight = Math.max(100, maxHeight);
            setBox({ top, left, width, maxHeight });
        };
        place();
        window.addEventListener("resize", place);
        window.addEventListener("scroll", place, true);
        return () => {
            window.removeEventListener("resize", place);
            window.removeEventListener("scroll", place, true);
        };
    }, [show]);

    return (
        <div ref={wrapRef} className="relative shrink-0">
            <button
                type="button"
                className="btn btn-ghost btn-xs btn-circle min-h-7 w-7 p-0"
                aria-label={t("voice_changer.info_learn_more")}
                aria-expanded={show}
                onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setShow((s) => !s);
                }}
            >
                <RiInformationLine className="text-lg text-base-content/70" />
            </button>
            {show && box
                ? createPortal(
                      <>
                          <button
                              type="button"
                              className="fixed inset-0 z-[10000] cursor-default bg-transparent"
                              aria-label={t("voice_changer.info_close_overlay")}
                              onClick={() => setShow(false)}
                          />
                          <div
                              className="fixed z-[10001] overflow-y-auto whitespace-pre-line rounded-lg border border-base-content/15 bg-base-100 p-3 text-xs font-normal text-base-content/80 shadow-lg"
                              style={{
                                  top: box.top,
                                  left: box.left,
                                  width: box.width,
                                  maxHeight: box.maxHeight,
                              }}
                          >
                              {text}
                          </div>
                      </>,
                      document.body,
                  )
                : null}
        </div>
    );
};

export const VC_RANGE =
    "range range-sm range-primary h-6 w-full min-w-0 shrink-0 cursor-pointer [width:100%]";

/** Label on its own row; uniform full-width slider and end caps (matches inspector tone controls). */
export const VcSliderField: FC<{
    labelKey: string;
    /** Optional (i) next to the label. */
    infoKey?: string;
    inactive?: boolean;
    controlsRow: ReactNode;
    rangeMin: number;
    rangeMax: number;
    rangeStep: number;
    rangeValue: number;
    onRangeChange: (v: number) => void;
    /** Fired when a drag ends or the slider loses focus (for values that need an engine restart). */
    onRangeCommit?: () => void;
    lowLabelKey: string;
    highLabelKey: string;
}> = ({
    labelKey,
    infoKey,
    inactive,
    controlsRow,
    rangeMin,
    rangeMax,
    rangeStep,
    rangeValue,
    onRangeChange,
    onRangeCommit,
    lowLabelKey,
    highLabelKey,
}) => {
    const { t } = useTranslation();
    return (
        <div className="flex w-full min-w-0 flex-col gap-2">
            {infoKey ? (
                <div className="flex min-w-0 items-center gap-1">
                    <div className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</div>
                    <InspectorInfoButton text={t(infoKey)} />
                </div>
            ) : (
                <div className="text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</div>
            )}
            <div
                className={classNames(
                    "flex w-full min-w-0 flex-col gap-2",
                    inactive && "pointer-events-none opacity-50",
                )}
            >
                <div className="flex min-h-8 w-full min-w-0 flex-wrap items-center gap-2">{controlsRow}</div>
                <input
                    type="range"
                    min={rangeMin}
                    max={rangeMax}
                    step={rangeStep}
                    value={rangeValue}
                    onChange={(e) => onRangeChange(parseFloat(e.target.value))}
                    onPointerUp={onRangeCommit}
                    onBlur={onRangeCommit}
                    className={VC_RANGE}
                />
                <div className="flex w-full min-w-0 justify-between gap-2 text-xs text-base-content/40">
                    <span className="min-w-0 truncate">{t(lowLabelKey)}</span>
                    <span className="min-w-0 truncate text-end">{t(highLabelKey)}</span>
                </div>
            </div>
        </div>
    );
};

/** Section title matching other service inspectors (`Inspector.SubHeader` + optional (i)). */
export const VcSubHeader: FC<{ titleKey: string; infoKey?: string }> = ({ titleKey, infoKey }) => {
    const { t } = useTranslation();
    return (
        <Inspector.SubHeader>
            <span className="min-w-0 flex-1 truncate">{t(titleKey)}</span>
            {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
        </Inspector.SubHeader>
    );
};

/** Collapsible group (same pattern as OBS inspector) so dense RVC/advanced options stay scannable. */
export const VcDetails: FC<{
    titleKey: string;
    infoKey?: string;
    defaultOpen?: boolean;
    children: ReactNode;
}> = ({ titleKey, infoKey, defaultOpen = false, children }) => {
    const { t } = useTranslation();
    const [open, setOpen] = useState(defaultOpen);
    return (
        <details
            className="rounded-lg border border-base-content/10 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden"
            open={open}
            onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
        >
            <summary className="flex min-h-9 items-center gap-2 px-2.5 py-1.5 text-xs font-semibold uppercase tracking-wide text-base-content/55 hover:bg-base-content/5">
                <span className="min-w-0 flex-1 truncate">{t(titleKey)}</span>
                {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
            </summary>
            <div className="flex flex-col gap-3 border-t border-base-content/10 px-2.5 py-2.5">{children}</div>
        </details>
    );
};

/**
 * Number input that keeps what the user types until blur / Enter and only then commits the clamped value
 * (a controlled input that clamps on every keystroke makes e.g. "250" impossible to type when min is 100).
 */
export const VcDraftNumberInput: FC<{
    value: number;
    min: number;
    max: number;
    step: number;
    ariaLabel: string;
    onCommit: (v: number) => void;
}> = ({ value, min, max, step, ariaLabel, onCommit }) => {
    const [draft, setDraft] = useState<string | null>(null);
    const commit = () => {
        if (draft === null) return;
        const x = parseFloat(draft);
        setDraft(null);
        if (!Number.isNaN(x)) onCommit(Math.min(max, Math.max(min, x)));
    };
    return (
        <input
            type="number"
            className="input input-bordered input-sm w-24 shrink-0 font-mono"
            min={min}
            max={max}
            step={step}
            aria-label={ariaLabel}
            value={draft ?? String(value)}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
                if (e.key === "Enter") commit();
                if (e.key === "Escape") setDraft(null);
            }}
        />
    );
};

/** Path row with optional file/folder browse (Tauri dialog) and optional (i) next to the label. */
export const VcPathField: FC<{
    labelKey: string;
    infoKey?: string;
    value: string;
    onChange: (path: string) => void;
    onCommit: () => void;
    directory?: boolean;
    extensions?: string[];
    filterName?: string;
}> = ({ labelKey, infoKey, value, onChange, onCommit, directory, extensions, filterName }) => {
    const { t } = useTranslation();
    const label = (
        <label className="label py-0">
            <span className="label-text">{t(labelKey)}</span>
        </label>
    );
    return (
        <div className="flex flex-col gap-1">
            {infoKey ? (
                <div className="flex items-center gap-1">
                    {label}
                    <InspectorInfoButton text={t(infoKey)} />
                </div>
            ) : (
                label
            )}
            <div className="flex items-center gap-2">
                <input
                    type="text"
                    className="input input-bordered input-sm min-w-0 flex-1 font-mono text-xs"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    onBlur={() => onCommit()}
                />
                {isTauri() ? (
                    <button
                        type="button"
                        className="btn btn-sm h-8 min-h-8 shrink-0 border border-base-content/20 bg-base-100 font-semibold normal-case text-base-content hover:bg-base-200"
                        onClick={async () => {
                            const path = await open(
                                directory
                                    ? { directory: true, multiple: false }
                                    : {
                                          filters: [{ name: filterName || "ONNX", extensions: extensions || ["onnx"] }],
                                          multiple: false,
                                      },
                            );
                            if (path === null || Array.isArray(path)) return;
                            onChange(path);
                            onCommit();
                        }}
                    >
                        {t("voice_changer.onnx_browse")}
                    </button>
                ) : (
                    <span className="max-w-[10rem] shrink-0 text-xs text-base-content/50">
                        {t("voice_changer.onnx_browse_unavailable")}
                    </span>
                )}
            </div>
        </div>
    );
};

/** One option of a `VcChoiceSelect`. */
export interface VcChoice {
    value: string;
    label: string;
    disabled?: boolean;
}

/**
 * Native select (supports disabled options, unlike `InputSelect`), label on its own row with an
 * optional (i), and an optional note under it (e.g. what runs instead of the stored choice).
 */
export const VcChoiceSelect: FC<{
    labelKey: string;
    infoKey?: string;
    value: string;
    options: readonly VcChoice[];
    onChange: (value: string) => void;
    disabled?: boolean;
    note?: ReactNode;
    noteTone?: "muted" | "warn";
}> = ({ labelKey, infoKey, value, options, onChange, disabled, note, noteTone = "muted" }) => {
    const { t } = useTranslation();
    return (
        <div className="flex w-full min-w-0 flex-col gap-1">
            <div className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</span>
                {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
            </div>
            <select
                className="select select-bordered select-sm w-full"
                aria-label={t(labelKey)}
                value={value}
                disabled={disabled}
                onChange={(e) => onChange(e.target.value)}
            >
                {options.map((o) => (
                    <option key={o.value} value={o.value} disabled={o.disabled} className={o.disabled ? "opacity-40" : undefined}>
                        {o.label}
                    </option>
                ))}
            </select>
            {note ? (
                <p className={classNames("text-xs leading-snug", noteTone === "warn" ? "text-warning" : "text-base-content/55")}>{note}</p>
            ) : null}
        </div>
    );
};

/** A small caption line (muted, or a warning). */
export const VcNote: FC<{ tone?: "muted" | "warn" | "error"; children: ReactNode; className?: string }> = ({
    tone = "muted",
    children,
    className,
}) => (
    <p
        className={classNames(
            "text-xs leading-snug break-words",
            tone === "warn" ? "text-warning" : tone === "error" ? "text-error" : "text-base-content/55",
            className,
        )}
    >
        {children}
    </p>
);

/** A slider with a number field (committed on blur / Enter), for hot values synced as they change. */
export const VcNumberSlider: FC<{
    labelKey: string;
    infoKey?: string;
    value: number;
    min: number;
    max: number;
    step: number;
    /** Digits shown in the number field. */
    digits?: number;
    unitKey?: string;
    lowLabelKey: string;
    highLabelKey: string;
    inactive?: boolean;
    /** Every change (slider move, committed field). */
    onChange: (v: number) => void;
    /** Slider released / field committed (for values that restart the engine). */
    onCommit?: () => void;
    resetTo?: number;
}> = ({ labelKey, infoKey, value, min, max, step, digits = 2, unitKey, lowLabelKey, highLabelKey, inactive, onChange, onCommit, resetTo }) => {
    const { t } = useTranslation();
    const shown = Number.isFinite(value) ? Number(value.toFixed(digits)) : min;
    return (
        <VcSliderField
            labelKey={labelKey}
            infoKey={infoKey}
            inactive={inactive}
            controlsRow={
                <>
                    <VcDraftNumberInput
                        value={shown}
                        min={min}
                        max={max}
                        step={step}
                        ariaLabel={t(labelKey)}
                        onCommit={(v) => {
                            onChange(v);
                            onCommit?.();
                        }}
                    />
                    {unitKey ? <span className="shrink-0 text-xs text-base-content/60">{t(unitKey)}</span> : null}
                    {resetTo !== undefined && Math.abs(value - resetTo) > step / 2 ? (
                        <button
                            type="button"
                            className="btn btn-ghost btn-xs shrink-0"
                            onClick={() => {
                                onChange(resetTo);
                                onCommit?.();
                            }}
                        >
                            {t("voice_changer.pitch_reset")}
                        </button>
                    ) : null}
                </>
            }
            rangeMin={min}
            rangeMax={max}
            rangeStep={step}
            rangeValue={shown}
            onRangeChange={onChange}
            onRangeCommit={onCommit}
            lowLabelKey={lowLabelKey}
            highLabelKey={highLabelKey}
        />
    );
};
