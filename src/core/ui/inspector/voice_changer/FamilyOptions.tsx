// Voice-changer inspector (P5-G): the options form of each voice family (`manifest.family`), controlled by the
// caller (the "new voice" form and the selected voice's options). Values follow `families.ts` (`FamilyForms`): an
// empty number field means "the preset's value" (`null`). Rust normalizes and validates them again.
import {
    DDSP_SVC_PRESETS,
    FAMILY_OPTION_RANGES,
    familyOptionsProblem,
    MEANVC_STEPS,
    MEANVC_TICK_MS,
    parseSidecarArgs,
    SIDECAR_RECIPES,
    sidecarProblemField,
    sidecarRecipeStartsHelper,
    SIGIL_ZS_PRESETS,
    switchSidecarRecipe,
    type DdspSvcForm,
    type FamilyFormOf,
    type FamilyOptionsProblem,
    type MeanVcForm,
    type SidecarRecipe,
    type SidecarV2Form,
    type SigilZsForm,
    type SovitsForm,
} from "@/core/services/voice_changer/families";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { InspectorInfoButton, VcChoiceSelect, VcDetails, VcNote, VcNumberSlider, VcPathField } from "./common";

const R = FAMILY_OPTION_RANGES;

/**
 * A number field where empty means "default" (`null`): keeps what is typed until blur / Enter, then commits the
 * clamped value (rounded when `integer`), or `null` for an empty field.
 */
const OptionalNumberField: FC<{
    labelKey: string;
    infoKey?: string;
    value: number | null;
    min: number;
    max: number;
    step: number;
    integer?: boolean;
    placeholder: string;
    disabled?: boolean;
    onCommit: (v: number | null) => void;
}> = ({ labelKey, infoKey, value, min, max, step, integer, placeholder, disabled, onCommit }) => {
    const { t } = useTranslation();
    const [draft, setDraft] = useState<string | null>(null);
    const commit = () => {
        if (draft === null) return;
        const text = draft.trim();
        setDraft(null);
        if (text === "") {
            onCommit(null);
            return;
        }
        const x = parseFloat(text);
        if (Number.isNaN(x)) return;
        const c = Math.min(max, Math.max(min, x));
        onCommit(integer ? Math.round(c) : c);
    };
    return (
        <label className="flex min-w-0 flex-col gap-1">
            <span className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t(labelKey)}</span>
                {infoKey ? <InspectorInfoButton text={t(infoKey)} /> : null}
            </span>
            <input
                type="number"
                className="input input-bordered input-sm w-full font-mono"
                min={min}
                max={max}
                step={step}
                placeholder={placeholder}
                disabled={disabled}
                value={draft ?? (value === null ? "" : String(value))}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commit}
                onKeyDown={(e) => {
                    if (e.key === "Enter") commit();
                    if (e.key === "Escape") setDraft(null);
                }}
            />
        </label>
    );
};

const Check: FC<{ labelKey: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }> = ({ labelKey, checked, disabled, onChange }) => {
    const { t } = useTranslation();
    return (
        <label className="label cursor-pointer justify-start gap-2 py-0">
            <input
                type="checkbox"
                className="checkbox checkbox-sm checkbox-primary"
                checked={checked}
                disabled={disabled}
                onChange={(e) => onChange(e.target.checked)}
            />
            <span className="label-text text-sm">{t(labelKey)}</span>
        </label>
    );
};

const DdspFields: FC<{ form: DdspSvcForm; onChange: (f: DdspSvcForm) => void; disabled?: boolean }> = ({ form, onChange, disabled }) => {
    const { t } = useTranslation();
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.family_preset"
                infoKey="voice_changer.family_ddsp_preset_info"
                value={form.preset}
                disabled={disabled}
                options={DDSP_SVC_PRESETS.map((p) => ({ value: p, label: t(`voice_changer.family_ddsp_preset_${p}`) }))}
                onChange={(v) => onChange({ ...form, preset: v as DdspSvcForm["preset"] })}
            />
            <VcDetails titleKey="voice_changer.family_options_advanced">
                <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
                    <OptionalNumberField
                        labelKey="voice_changer.family_steps"
                        infoKey="voice_changer.family_ddsp_steps_info"
                        value={form.steps}
                        min={R.ddspSteps.min}
                        max={R.ddspSteps.max}
                        step={1}
                        integer
                        placeholder={t("voice_changer.family_from_preset")}
                        disabled={disabled}
                        onCommit={(steps) => onChange({ ...form, steps })}
                    />
                    <OptionalNumberField
                        labelKey="voice_changer.family_ddsp_t_start"
                        infoKey="voice_changer.family_ddsp_t_start_info"
                        value={form.t_start}
                        min={R.ddspTStart.min}
                        max={R.ddspTStart.max}
                        step={0.05}
                        placeholder={String(R.ddspTStart.default)}
                        disabled={disabled}
                        onCommit={(t_start) => onChange({ ...form, t_start })}
                    />
                </div>
            </VcDetails>
        </>
    );
};

const ZsFields: FC<{ form: SigilZsForm; onChange: (f: SigilZsForm) => void; disabled?: boolean }> = ({ form, onChange, disabled }) => {
    const { t } = useTranslation();
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.family_preset"
                infoKey="voice_changer.family_zs_preset_info"
                value={form.preset}
                disabled={disabled}
                options={SIGIL_ZS_PRESETS.map((p) => ({ value: p, label: t(`voice_changer.family_zs_preset_${p}`) }))}
                onChange={(v) => onChange({ ...form, preset: v as SigilZsForm["preset"] })}
            />
            <VcDetails titleKey="voice_changer.family_options_advanced">
                <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-3">
                    <OptionalNumberField
                        labelKey="voice_changer.family_steps"
                        infoKey="voice_changer.family_zs_steps_info"
                        value={form.steps}
                        min={R.zsSteps.min}
                        max={R.zsSteps.max}
                        step={1}
                        integer
                        placeholder={t("voice_changer.family_from_preset")}
                        disabled={disabled}
                        onCommit={(steps) => onChange({ ...form, steps })}
                    />
                    <OptionalNumberField
                        labelKey="voice_changer.family_zs_cfg_rate"
                        infoKey="voice_changer.family_zs_cfg_rate_info"
                        value={form.cfg_rate}
                        min={R.zsCfgRate.min}
                        max={R.zsCfgRate.max}
                        step={0.05}
                        placeholder={t("voice_changer.family_from_preset")}
                        disabled={disabled}
                        onCommit={(cfg_rate) => onChange({ ...form, cfg_rate })}
                    />
                    <OptionalNumberField
                        labelKey="voice_changer.family_zs_prompt_s"
                        infoKey="voice_changer.family_zs_prompt_s_info"
                        value={form.prompt_s}
                        min={R.zsPromptS.min}
                        max={R.zsPromptS.max}
                        step={0.5}
                        placeholder={t("voice_changer.family_from_preset")}
                        disabled={disabled}
                        onCommit={(prompt_s) => onChange({ ...form, prompt_s })}
                    />
                </div>
                <VcChoiceSelect
                    labelKey="voice_changer.family_zs_fp16_vocoder"
                    infoKey="voice_changer.family_zs_fp16_vocoder_info"
                    value={form.fp16_vocoder === null ? "auto" : form.fp16_vocoder ? "on" : "off"}
                    disabled={disabled}
                    options={(["auto", "on", "off"] as const).map((m) => ({ value: m, label: t(`voice_changer.lever_mode_${m}`) }))}
                    onChange={(v) => onChange({ ...form, fp16_vocoder: v === "auto" ? null : v === "on" })}
                />
                <Check
                    labelKey="voice_changer.family_zs_fp16_xlsr"
                    checked={form.fp16_xlsr}
                    disabled={disabled}
                    onChange={(fp16_xlsr) => onChange({ ...form, fp16_xlsr })}
                />
                <Check
                    labelKey="voice_changer.family_zs_fp16_dit"
                    checked={form.fp16_dit}
                    disabled={disabled}
                    onChange={(fp16_dit) => onChange({ ...form, fp16_dit })}
                />
                <VcNote tone={form.fp16_dit ? "warn" : "muted"}>{t("voice_changer.family_zs_fp16_note")}</VcNote>
            </VcDetails>
        </>
    );
};

const MeanVcFields: FC<{ form: MeanVcForm; onChange: (f: MeanVcForm) => void; disabled?: boolean }> = ({ form, onChange, disabled }) => {
    const { t } = useTranslation();
    return (
        <div className="grid w-full min-w-0 grid-cols-1 gap-2 sm:grid-cols-2">
            <VcChoiceSelect
                labelKey="voice_changer.family_steps"
                infoKey="voice_changer.family_meanvc_steps_info"
                value={String(form.steps)}
                disabled={disabled}
                options={MEANVC_STEPS.map((s) => ({ value: String(s), label: t(`voice_changer.family_meanvc_steps_${s}`) }))}
                onChange={(v) => onChange({ ...form, steps: Number(v) === 1 ? 1 : 2 })}
            />
            <VcChoiceSelect
                labelKey="voice_changer.family_meanvc_tick"
                infoKey="voice_changer.family_meanvc_tick_info"
                value={String(form.tick_ms)}
                disabled={disabled}
                options={MEANVC_TICK_MS.map((ms) => ({ value: String(ms), label: t(`voice_changer.family_meanvc_tick_${ms}`) }))}
                onChange={(v) => onChange({ ...form, tick_ms: Number(v) >= 120 ? 160 : 80 })}
            />
        </div>
    );
};

const SovitsFields: FC<{ form: SovitsForm; onChange: (f: SovitsForm) => void; disabled?: boolean }> = ({ form, onChange, disabled }) => (
    <VcNumberSlider
        labelKey="voice_changer.family_sovits_cluster_ratio"
        infoKey="voice_changer.family_sovits_cluster_ratio_info"
        value={form.cluster_ratio}
        min={R.sovitsClusterRatio.min}
        max={R.sovitsClusterRatio.max}
        step={0.01}
        inactive={disabled}
        lowLabelKey="voice_changer.label_own_timbre"
        highLabelKey="voice_changer.label_voice_timbre"
        resetTo={0}
        onChange={(cluster_ratio) => onChange({ ...form, cluster_ratio })}
    />
);

/** Sidecar arguments, one per line (kept as typed until the field loses focus). */
const ArgsField: FC<{ args: readonly string[]; disabled?: boolean; onCommit: (args: string[]) => void }> = ({ args, disabled, onCommit }) => {
    const { t } = useTranslation();
    const joined = args.join("\n");
    const [draft, setDraft] = useState<string | null>(null);
    useEffect(() => setDraft(null), [joined]);
    return (
        <label className="flex min-w-0 flex-col gap-1">
            <span className="flex min-w-0 items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-medium leading-snug text-base-content/80">{t("voice_changer.family_sidecar_args")}</span>
                <InspectorInfoButton text={t("voice_changer.family_sidecar_args_info")} />
            </span>
            <textarea
                className="textarea textarea-bordered textarea-sm w-full font-mono text-xs"
                rows={3}
                spellCheck={false}
                disabled={disabled}
                value={draft ?? joined}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={() => {
                    if (draft === null) return;
                    onCommit(parseSidecarArgs(draft));
                    setDraft(null);
                }}
            />
        </label>
    );
};

/**
 * The sidecar's address: required for a custom server; for a recipe an optional server of that engine the user
 * already started (Sigil then attaches instead of starting the helper). Always shown when it holds a value, so a
 * stored address is never sent unseen.
 */
const SidecarUrlField: FC<{ form: SidecarV2Form; problem: FamilyOptionsProblem | null; disabled?: boolean; onChange: (url: string) => void }> = ({
    form,
    problem,
    disabled,
    onChange,
}) => {
    const { t } = useTranslation();
    const helper = sidecarRecipeStartsHelper(form.recipe);
    const urlProblem = sidecarProblemField(problem) === "url" ? problem : null;
    return (
        <label className="flex min-w-0 flex-col gap-1">
            <span className="text-xs font-medium leading-snug text-base-content/80">
                {t(helper ? "voice_changer.family_sidecar_attach_url" : "voice_changer.family_sidecar_url")}
            </span>
            <input
                type="text"
                className="input input-bordered input-sm w-full font-mono text-xs"
                placeholder="ws://127.0.0.1:8765/vc"
                spellCheck={false}
                disabled={disabled}
                value={form.url}
                onChange={(e) => onChange(e.target.value)}
            />
            <VcNote tone={urlProblem ? "warn" : "muted"}>
                {urlProblem
                    ? t(`voice_changer.family_problem_${urlProblem}`)
                    : helper
                      ? t(form.url.trim() !== "" ? "voice_changer.family_sidecar_attach_url_active" : "voice_changer.family_sidecar_attach_url_hint")
                      : t("voice_changer.family_sidecar_url_hint")}
            </VcNote>
        </label>
    );
};

const SidecarFields: FC<{ form: SidecarV2Form; onChange: (f: SidecarV2Form) => void; disabled?: boolean }> = ({ form, onChange, disabled }) => {
    const { t } = useTranslation();
    const problem = familyOptionsProblem("sidecar_v2", form);
    const field = sidecarProblemField(problem);
    const helper = sidecarRecipeStartsHelper(form.recipe);
    const problemNote = (f: typeof field) => (problem && field === f ? <VcNote tone="warn">{t(`voice_changer.family_problem_${problem}`)}</VcNote> : null);
    return (
        <>
            <VcChoiceSelect
                labelKey="voice_changer.family_sidecar_recipe"
                infoKey="voice_changer.family_sidecar_recipe_info"
                value={form.recipe}
                disabled={disabled}
                options={SIDECAR_RECIPES.map((r) => ({ value: r, label: t(`voice_changer.family_sidecar_recipe_${r}`) }))}
                onChange={(v) => onChange(switchSidecarRecipe(form, v as SidecarRecipe))}
                note={t(`voice_changer.family_sidecar_recipe_${form.recipe}_about`)}
            />
            {!helper ? <SidecarUrlField form={form} problem={problem} disabled={disabled} onChange={(url) => onChange({ ...form, url })} /> : null}
            {helper ? (
                <>
                    <VcPathField
                        labelKey="voice_changer.family_sidecar_python"
                        infoKey="voice_changer.family_sidecar_python_info"
                        value={form.python}
                        extensions={["exe"]}
                        filterName="Python"
                        onChange={(python) => onChange({ ...form, python })}
                        onCommit={() => {}}
                    />
                    {problemNote("python")}
                    <VcPathField
                        labelKey="voice_changer.family_sidecar_model_dir"
                        infoKey="voice_changer.family_sidecar_model_dir_info"
                        value={form.model_dir}
                        directory
                        onChange={(model_dir) => onChange({ ...form, model_dir })}
                        onCommit={() => {}}
                    />
                    {problemNote("model_dir")}
                    <SidecarUrlField form={form} problem={problem} disabled={disabled} onChange={(url) => onChange({ ...form, url })} />
                </>
            ) : null}
            {/* A problem of the arguments shows outside the collapsed section too, so a disabled Save / Create is explained. */}
            {problemNote("args")}
            <VcDetails titleKey="voice_changer.family_options_advanced">
                {helper ? <ArgsField args={form.args} disabled={disabled} onCommit={(args) => onChange({ ...form, args })} /> : null}
                <OptionalNumberField
                    labelKey="voice_changer.family_sidecar_timeout"
                    infoKey="voice_changer.family_sidecar_timeout_info"
                    value={form.timeout_ms}
                    min={R.sidecarTimeoutMs.min}
                    max={R.sidecarTimeoutMs.max}
                    step={10}
                    integer
                    placeholder={t("voice_changer.family_sidecar_timeout_budget")}
                    disabled={disabled}
                    onCommit={(timeout_ms) => onChange({ ...form, timeout_ms })}
                />
            </VcDetails>
        </>
    );
};

/** The options form of `value.family`. */
export const FamilyOptionsFields: FC<{ value: FamilyFormOf; onChange: (v: FamilyFormOf) => void; disabled?: boolean }> = ({
    value,
    onChange,
    disabled,
}) => {
    switch (value.family) {
        case "ddsp_svc":
            return <DdspFields form={value.form} disabled={disabled} onChange={(form) => onChange({ family: "ddsp_svc", form })} />;
        case "sigil_zs":
            return <ZsFields form={value.form} disabled={disabled} onChange={(form) => onChange({ family: "sigil_zs", form })} />;
        case "meanvc":
            return <MeanVcFields form={value.form} disabled={disabled} onChange={(form) => onChange({ family: "meanvc", form })} />;
        case "sovits":
            return <SovitsFields form={value.form} disabled={disabled} onChange={(form) => onChange({ family: "sovits", form })} />;
        case "sidecar_v2":
            return <SidecarFields form={value.form} disabled={disabled} onChange={(form) => onChange({ family: "sidecar_v2", form })} />;
        default:
            return null;
    }
};
