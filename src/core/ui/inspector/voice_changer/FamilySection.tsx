// Voice-changer inspector (P5-G; stub scaffolded by P5-S0): the selected voice's P5 family.
//
// For a family voice (`VcCapabilities.families`, `families.ts`) this shows what the family is (experimental,
// zero-shot, singing), where it runs (with a one-click switch to a rung it runs on), the zero-shot reference
// picker (the per-voice `referenceVoiceId`, from the shared voice library, with `<ReferenceVoiceRecorder>`),
// singing mode for singing families, and the family's options (`VC_COMMANDS.setFamilyOptions`; a loaded voice
// uses them after it reloads, so a running engine offers a restart). Rendered under the voice library in RVC
// mode; renders nothing for RVC voices. Making a family voice is `FamilyCreator.tsx`.
import { rvcBaseEp } from "@/core/services/voice_changer/contract";
import {
    familyEpChoice,
    familyOptionsForm,
    familyOptionsPayload,
    familyOptionsProblem,
    familyRunsOnEp,
    sameFamilyOptions,
    singingRecommended,
    type FamilyFormOf,
} from "@/core/services/voice_changer/families";
import type { VcFamilyCapability, VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { epDisplayName } from "@/core/services/voice_changer/metrics";
import { FC, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import { InspectorInfoButton, VcNote } from "./common";
import { FamilyBadges, familyEpList, familyName } from "./FamilyLabels";
import { FamilyOptionsFields } from "./FamilyOptions";
import { useVoiceFamily } from "./hooks";
import { ReferenceVoicePicker } from "./ReferenceVoicePicker";
import { importErrorText } from "./errors";

/** The stored options of `voice` as a form. */
function storedForm(fam: VcFamilyCapability, voice: VcSlotInfo): FamilyFormOf {
    return { family: fam.family, form: familyOptionsForm(fam.family, voice.family_options ?? {}) } as FamilyFormOf;
}

/** The family's options, saved with the voice (`manifest.family`). */
const FamilyOptionsEditor: FC<{ fam: VcFamilyCapability; voice: VcSlotInfo }> = ({ fam, voice }) => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const lib = useSnapshot(vcApi.library.state);
    const running = useSnapshot(vcApi.state).isRunning;
    const stored = useMemo(() => storedForm(fam, voice), [fam, voice]);
    const storedKey = JSON.stringify(familyOptionsPayload(stored.family, stored.form as never));
    const [draft, setDraft] = useState<FamilyFormOf>(stored);
    /**
     * Saved while the engine ran: the loaded voice (also one kept in the hot-voice cache while another voice
     * plays) uses the new options only after a restart. Kept per voice by the library until the engine stops.
     */
    const reloadPending = running && lib.optionsReloadPending?.[voice.id] === true;

    // A new voice, or its stored options changed (saved here or elsewhere): show them.
    useEffect(() => {
        setDraft(stored);
    }, [voice.id, storedKey]); // `stored` changes with them

    const dirty = !sameFamilyOptions(draft.family, draft.form as never, stored.form as never);
    const problem = familyOptionsProblem(draft.family, draft.form as never);
    const busy = lib.busy[voice.id] === "options";

    const save = async () => {
        try {
            await vcApi.library.setFamilyOptions(voice.id, familyOptionsPayload(draft.family, draft.form as never));
            if (!vcApi.state.isRunning) toast.success(t("voice_changer.family_options_saved"));
        } catch (e) {
            toast.error(importErrorText(t, e));
        }
    };

    return (
        <div className="flex flex-col gap-2">
            <div className="flex items-center gap-1">
                <span className="min-w-0 flex-1 text-xs font-semibold uppercase tracking-wide text-base-content/55">{t("voice_changer.family_options")}</span>
                <InspectorInfoButton text={t("voice_changer.family_options_info")} />
            </div>
            <FamilyOptionsFields value={draft} onChange={setDraft} disabled={busy} />
            {problem ? <VcNote tone="warn">{t(`voice_changer.family_problem_${problem}`)}</VcNote> : null}
            {dirty ? (
                <div className="flex justify-end gap-1">
                    <button type="button" className="btn btn-ghost btn-xs" disabled={busy} onClick={() => setDraft(stored)}>
                        {t("voice_changer.family_options_revert")}
                    </button>
                    <button type="button" className="btn btn-primary btn-xs" disabled={busy || !!problem} onClick={() => void save()}>
                        {busy ? <span className="loading loading-spinner loading-xs" /> : null}
                        {t("voice_changer.voice_save")}
                    </button>
                </div>
            ) : null}
            {reloadPending && !dirty ? (
                <div className="flex flex-wrap items-center gap-2">
                    <VcNote className="min-w-0 flex-1">{t("voice_changer.family_options_reload_note")}</VcNote>
                    <button type="button" className="btn btn-xs" onClick={() => void vcApi.restartIfRunning()}>
                        {t("voice_changer.family_options_restart")}
                    </button>
                </div>
            ) : null}
        </div>
    );
};

/** The selected voice's family settings (nothing for RVC voices). */
export const FamilySection: FC<{ voice: VcSlotInfo | undefined }> = ({ voice }) => {
    const { t } = useTranslation();
    const snap = useSnapshot(window.ApiServer.voiceChanger.state);
    const vc = snap.voiceChanger;
    const vcApi = window.ApiServer.voiceChanger;
    const { family: fam } = useVoiceFamily(voice);
    if (!voice || !fam) return null;

    const caps = snap.capabilities as typeof vcApi.state.capabilities;
    const runsHere = familyRunsOnEp(fam, vc.rvcExecutionProvider);
    const epChoice = runsHere ? null : familyEpChoice(fam, caps);
    const running = snap.isRunning && vc.mode === "rvc";

    return (
        <div className="flex flex-col gap-2.5 rounded-lg border border-base-content/10 p-2.5" aria-label={familyName(t, fam.family)}>
            <div className="flex min-w-0 flex-wrap items-center gap-1.5">
                <span className="min-w-0 flex-1 truncate text-sm font-semibold">{familyName(t, fam.family)}</span>
                <FamilyBadges fam={fam} />
                <InspectorInfoButton text={t(`voice_changer.family_info_${fam.family}`)} />
            </div>
            <VcNote>{t(`voice_changer.family_about_${fam.family}`)}</VcNote>
            {fam.experimental ? <VcNote tone="warn">{t("voice_changer.family_experimental_note")}</VcNote> : null}
            {!fam.available ? <VcNote tone="warn">{t("voice_changer.family_not_in_build", { name: familyName(t, fam.family) })}</VcNote> : null}

            {!runsHere ? (
                <div className="flex flex-wrap items-center gap-2">
                    <VcNote tone="warn" className="min-w-0 flex-1">
                        {t("voice_changer.family_runs_on_other", { eps: familyEpList(fam), ep: epDisplayName(rvcBaseEp(vc.rvcExecutionProvider)) })}
                    </VcNote>
                    {epChoice ? (
                        <button type="button" className="btn btn-xs" onClick={() => void vcApi.update({ rvcExecutionProvider: epChoice })}>
                            {t("voice_changer.family_use_ep", { ep: t(`voice_changer.ep_name_${epChoice}`) })}
                        </button>
                    ) : (
                        <VcNote tone="warn">{t("voice_changer.family_no_ep")}</VcNote>
                    )}
                </div>
            ) : null}

            {fam.zero_shot ? (
                <div className="flex flex-col gap-1">
                    <ReferenceVoicePicker
                        value={vc.referenceVoiceId}
                        required={fam.creation === "reference"}
                        labelKey={fam.creation === "reference" ? "voice_changer.family_reference" : "voice_changer.family_reference_optional"}
                        onChange={(id) => void vcApi.setReferenceVoice(id)}
                    />
                    {fam.creation === "reference" && vc.referenceVoiceId.trim() === "" ? (
                        <VcNote tone="warn">{t("voice_changer.family_problem_reference_required")}</VcNote>
                    ) : null}
                    {running ? <VcNote>{t("voice_changer.family_reference_switch_note")}</VcNote> : null}
                </div>
            ) : null}

            {singingRecommended(fam) && vc.f0Range !== "singing" ? (
                <div className="flex flex-wrap items-center gap-2">
                    <VcNote className="min-w-0 flex-1">{t("voice_changer.family_singing_suggest")}</VcNote>
                    <button type="button" className="btn btn-xs" onClick={() => void vcApi.update({ f0Range: "singing" })}>
                        {t("voice_changer.family_singing_use")}
                    </button>
                </div>
            ) : null}

            {fam.available ? <FamilyOptionsEditor fam={fam} voice={voice} /> : null}
        </div>
    );
};
