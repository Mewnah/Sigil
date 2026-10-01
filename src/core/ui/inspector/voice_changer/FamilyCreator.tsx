// Voice-changer inspector (P5-G): make a new voice of a P5 family (`voice_changer_create_family_slot`). The
// families this build runs are listed with what they are (zero-shot, singing, experimental) and where they run;
// the form asks for what the family needs: a voice-library reference (zero-shot), a file to import (DDSP-SVC,
// so-vits-svc), or a sidecar recipe / URL, plus the family's options. The voice is made by the voice library's
// job queue (progress, cancel) and selected when it is ready.
import { rvcBaseEp } from "@/core/services/voice_changer/contract";
import {
    availableFamilies,
    familyDraftFromRequest,
    familyDraftProblem,
    familyEpChoice,
    familyPickerExtensions,
    familyRunsOnEp,
    familySlotRequest,
    newFamilyDraft,
    switchDraftFamily,
    type FamilyDraft,
    type FamilyFormOf,
} from "@/core/services/voice_changer/families";
import type { VcFamilyCapability, VcFamilyId, VcFamilySlotRequest } from "@/core/services/voice_changer/ipc";
import { epDisplayName } from "@/core/services/voice_changer/metrics";
import { EMBEDDER_CATALOG_IDS } from "@/core/services/voice_changer/slots";
import { open } from "@tauri-apps/plugin-dialog";
import classNames from "classnames";
import { FC, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiCloseLine, RiFileLine, RiFolderOpenLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { ModelRequirement } from "../components/ModelRequirement";
import { VcNote } from "./common";
import { FamilyBadges, familyEpList, familyName } from "./FamilyLabels";
import { FamilyOptionsFields } from "./FamilyOptions";
import { useVcCapabilities } from "./hooks";
import { ReferenceVoicePicker } from "./ReferenceVoicePicker";

/** Model-store entries a family's voices need before they can run (the family's own, plus its content encoder). */
function familyRequirementIds(fam: VcFamilyCapability): string[] {
    const ids: string[] = [];
    if (fam.needs_content_encoder) ids.push(EMBEDDER_CATALOG_IDS.contentvec);
    for (const id of fam.requires) if (!ids.includes(id)) ids.push(id);
    return ids;
}

/**
 * The file (or folder) to import, for import families. `isDir`: the path came from the folder picker (a typed
 * path is not known to be a folder; one whose name has a dot is still accepted, see `importSourceProblem`).
 */
const SourceField: FC<{ fam: VcFamilyCapability; value: string; onChange: (path: string, isDir: boolean) => void }> = ({ fam, value, onChange }) => {
    const { t } = useTranslation();
    const exts = familyPickerExtensions(fam);
    const pick = async (directory: boolean) => {
        const picked = await open(
            directory
                ? { directory: true, multiple: false, title: t("voice_changer.family_source_pick_title") }
                : {
                      directory: false,
                      multiple: false,
                      title: t("voice_changer.family_source_pick_title"),
                      filters: exts.length > 0 ? [{ name: familyName(t, fam.family), extensions: exts }] : undefined,
                  },
        );
        if (typeof picked === "string" && picked) onChange(picked, directory);
    };
    return (
        <div className="flex min-w-0 flex-col gap-1">
            <span className="text-xs font-medium leading-snug text-base-content/80">{t("voice_changer.family_source")}</span>
            <input
                type="text"
                className="input input-bordered input-sm w-full font-mono text-xs"
                spellCheck={false}
                value={value}
                placeholder={t("voice_changer.family_source_placeholder")}
                onChange={(e) => onChange(e.target.value, false)}
            />
            <div className="flex flex-wrap gap-1">
                <button type="button" className="btn btn-xs gap-1" onClick={() => void pick(false)}>
                    <RiFileLine aria-hidden /> {t("voice_changer.family_source_file")}
                </button>
                <button type="button" className="btn btn-xs gap-1" onClick={() => void pick(true)}>
                    <RiFolderOpenLine aria-hidden /> {t("voice_changer.family_source_folder")}
                </button>
            </div>
            <VcNote>
                {fam.family === "ddsp_svc" || fam.family === "sovits"
                    ? t(`voice_changer.family_source_hint_${fam.family}`)
                    : t("voice_changer.family_source_hint", { types: fam.import_extensions.join(", ") })}
            </VcNote>
        </div>
    );
};

/** One family to choose (radio row). */
const FamilyChoice: FC<{ fam: VcFamilyCapability; selected: boolean; onSelect: () => void }> = ({ fam, selected, onSelect }) => {
    const { t } = useTranslation();
    const ep = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger.rvcExecutionProvider;
    const runsHere = familyRunsOnEp(fam, ep);
    return (
        <label
            className={classNames(
                "flex cursor-pointer items-start gap-2 rounded-lg border p-2",
                selected ? "border-primary/60 bg-primary/5" : "border-base-content/10 hover:bg-base-content/5",
            )}
        >
            <input type="radio" name="vc-family" className="radio radio-sm radio-primary mt-0.5 shrink-0" checked={selected} onChange={onSelect} />
            <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="flex min-w-0 flex-wrap items-center gap-1.5">
                    <span className="text-sm font-semibold">{familyName(t, fam.family)}</span>
                    <FamilyBadges fam={fam} />
                </span>
                <span className="text-xs leading-snug text-base-content/65">{t(`voice_changer.family_about_${fam.family}`)}</span>
                <span className={classNames("text-xs leading-snug", runsHere ? "text-base-content/50" : "text-warning")}>
                    {runsHere
                        ? t("voice_changer.family_runs_on", { eps: familyEpList(fam) })
                        : t("voice_changer.family_runs_on_other", { eps: familyEpList(fam), ep: epDisplayName(rvcBaseEp(ep)) })}
                </span>
            </span>
        </label>
    );
};

/** What the "new voice" panel starts with: a family (and file, e.g. a dropped one), or a request to edit and retry. */
export type FamilyCreatorInitial = { family: VcFamilyId; source?: string; request?: VcFamilySlotRequest | null };

/** The draft the panel opens with (`null` when this build runs no family). */
function initialDraft(initial: FamilyCreatorInitial | null | undefined, families: readonly VcFamilyCapability[]): FamilyDraft | null {
    const start = families.find((f) => f.family === initial?.family)?.family ?? families[0]?.family ?? null;
    if (!start) return null;
    if (initial?.request && initial.request.family === start) return familyDraftFromRequest(initial.request);
    return newFamilyDraft(start, start === initial?.family ? initial?.source ?? "" : "");
}

/**
 * The "new voice" panel. `initial`: a family (and file) to start with, e.g. from a dropped file, or a failed
 * family voice's request to edit and try again. `onClose` after the voice was queued or the user cancelled.
 */
export const FamilyCreator: FC<{ initial?: FamilyCreatorInitial | null; onClose: () => void }> = ({ initial, onClose }) => {
    const { t } = useTranslation();
    const caps = useVcCapabilities();
    const vc = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const families = useMemo(() => availableFamilies(caps), [caps]);
    const unavailable = caps.families.filter((f) => !f.available);
    const [draft, setDraft] = useState<FamilyDraft | null>(() => initialDraft(initial, families));
    const fam = draft ? families.find((f) => f.family === draft.family) ?? null : null;
    const problem = draft ? familyDraftProblem(draft, fam) : "family_unavailable";
    const wrongEp = !!fam && !familyRunsOnEp(fam, vc.rvcExecutionProvider);
    /** This build offers none of the rungs the family runs on: its voices could never load. */
    const noEp = wrongEp && !!fam && familyEpChoice(fam, caps) === null;

    // The capabilities arrived (or changed) after the panel opened: start on a family this build runs.
    useEffect(() => {
        if (families.length === 0) return;
        if (!draft || !families.some((f) => f.family === draft.family)) setDraft(initialDraft(initial, families));
    }, [families, draft, initial]);

    const pickFamily = (family: VcFamilyId) => {
        if (draft?.family === family) return;
        const target = families.find((f) => f.family === family);
        if (!target) return;
        // Keep what applies to both (name, reference, a file or folder the target imports too).
        setDraft((d) => (d ? switchDraftFamily(d, target) : newFamilyDraft(family)));
    };

    const create = () => {
        if (!draft || !fam || problem || noEp) return;
        window.ApiServer.voiceChanger.library.createFamilyVoice(familySlotRequest(draft, fam), { select: true });
        onClose();
    };

    return (
        <div className="flex flex-col gap-2.5 rounded-lg border border-primary/30 bg-primary/5 p-2.5">
            <div className="flex items-center gap-1">
                <span className="min-w-0 flex-1 text-sm font-semibold">{t("voice_changer.family_new_title")}</span>
                <button type="button" className="btn btn-ghost btn-xs btn-square" aria-label={t("voice_changer.voice_cancel")} onClick={onClose}>
                    <RiCloseLine />
                </button>
            </div>
            {families.length === 0 ? (
                <VcNote>{t("voice_changer.family_none_available")}</VcNote>
            ) : (
                <div className="flex flex-col gap-1.5" role="radiogroup" aria-label={t("voice_changer.family_new_title")}>
                    {families.map((f) => (
                        <FamilyChoice key={f.family} fam={f} selected={draft?.family === f.family} onSelect={() => pickFamily(f.family)} />
                    ))}
                </div>
            )}
            {unavailable.length > 0 ? (
                <VcNote>{t("voice_changer.family_unavailable_list", { names: unavailable.map((f) => familyName(t, f.family)).join(", ") })}</VcNote>
            ) : null}

            {draft && fam ? (
                <div className="flex flex-col gap-2.5 border-t border-base-content/10 pt-2.5">
                    {fam.creation === "import" ? (
                        <SourceField fam={fam} value={draft.source} onChange={(source, sourceIsDir) => setDraft({ ...draft, source, sourceIsDir })} />
                    ) : null}
                    {fam.zero_shot ? (
                        <ReferenceVoicePicker
                            value={draft.referenceVoiceId}
                            required={fam.creation === "reference"}
                            labelKey={fam.creation === "reference" ? "voice_changer.family_reference" : "voice_changer.family_reference_optional"}
                            onChange={(referenceVoiceId) => setDraft({ ...draft, referenceVoiceId })}
                        />
                    ) : null}
                    <label className="flex min-w-0 flex-col gap-1">
                        <span className="text-xs font-medium leading-snug text-base-content/80">{t("voice_changer.family_name_field")}</span>
                        <input
                            className="input input-bordered input-sm w-full"
                            value={draft.name}
                            maxLength={120}
                            placeholder={t(`voice_changer.family_name_placeholder_${fam.creation}`)}
                            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                        />
                    </label>
                    <FamilyOptionsFields
                        value={{ family: draft.family, form: draft.form } as FamilyFormOf}
                        onChange={(v) => setDraft({ ...draft, form: v.form })}
                    />
                    {familyRequirementIds(fam).length > 0 ? (
                        <ModelRequirement ids={familyRequirementIds(fam)} reason={t("voice_changer.family_needs")} hideWhenReady />
                    ) : null}
                    {wrongEp ? (
                        <VcNote tone="warn">
                            {noEp ? t("voice_changer.family_create_no_ep", { eps: familyEpList(fam) }) : t("voice_changer.family_create_ep_note", { eps: familyEpList(fam) })}
                        </VcNote>
                    ) : null}
                    {problem && problem !== "family_unavailable" ? <VcNote tone="warn">{t(`voice_changer.family_problem_${problem}`)}</VcNote> : null}
                    <div className="flex justify-end gap-1">
                        <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>
                            {t("voice_changer.voice_cancel")}
                        </button>
                        <button type="button" className="btn btn-primary btn-sm" disabled={!!problem || noEp} onClick={create}>
                            {t("voice_changer.family_create")}
                        </button>
                    </div>
                </div>
            ) : null}
        </div>
    );
};
