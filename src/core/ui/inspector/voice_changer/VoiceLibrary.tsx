// Voice-changer inspector (owned by P3-G): the voice library (slot manager).
//
// - The voices of the slot store: pick one (a hot switch while RVC runs), rename, duplicate, merge two
//   voices of one template, delete. Each row shows the voice's facts read-only (engine, version,
//   sample rate, pitch, content encoder, vocoder, index, speakers).
// - Import `.pth` / `.zip` voices by dropping them on the drop zone (Tauri's native drag and drop:
//   the webview gets file paths) or with the file picker; imports queue, show progress, can be
//   cancelled, and explain failures by the importer's error code.
// - First run: one card installs the base models and the default voice (the download is the
//   user's click; the default voice is imported as soon as it is installed).
// - "Other folder…" keeps bring-your-own ONNX voices anywhere on disk working.
// - P5-G: voices of the P5 families (zero-shot, singing, so-vits-svc, sidecars) are listed with their family,
//   experimental label, reference voice and where they run; "New voice…" makes one (`FamilyCreator.tsx`), and a
//   dropped file that only a family imports (e.g. a DDSP-SVC `.pt`) opens that family's form.
import { familyForImportFile, familyRunsOnEp, referenceName, voiceFamilyCapability } from "@/core/services/voice_changer/families";
import type { VcFamilyId, VcFamilySlotRequest, VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { isFinished, type VcImportJob } from "@/core/services/voice_changer/library";
import { DEFAULT_VOICE_ID, starterModelIds } from "@/core/services/voice_changer/requirements";
import {
    embedderDisplayName,
    INDEX_FILE_EXTENSIONS,
    isTemplateSlot,
    mergePartners,
    sameSlotPath,
    sampleRateLabel,
    slotProblem,
    vocoderDisplayName,
    VOICE_FILE_EXTENSIONS,
} from "@/core/services/voice_changer/slots";
import { voiceLibraryUi } from "@/core/services/voice_library";
import { isTauri } from "@tauri-apps/api/core";
import { appDataDir, join } from "@tauri-apps/api/path";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import classNames from "classnames";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
    RiAddLine,
    RiCheckboxCircleFill,
    RiCloseLine,
    RiDeleteBin6Line,
    RiDragDropLine,
    RiEdit2Line,
    RiFileCopyLine,
    RiGitMergeLine,
    RiMoreFill,
    RiRefreshLine,
} from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Tooltip from "../../dropdown/Tooltip";
import { ModelRequirement } from "../components/ModelRequirement";
import { VcDetails, VcNote, VcPathField } from "./common";
import { importErrorText } from "./errors";
import { FamilyCreator } from "./FamilyCreator";
import { FamilyBadges, familyEpList, familyName, familyShortName } from "./FamilyLabels";
import { useVcCapabilities, useVoiceLibraryHost } from "./hooks";
import { VoiceHotkeys } from "./VoiceHotkeys";

export { importErrorText } from "./errors";

/** Read-only facts of a voice as small badges. */
const VoiceFacts: FC<{ slot: VcSlotInfo }> = ({ slot }) => {
    const { t } = useTranslation();
    const caps = useVcCapabilities();
    const fam = voiceFamilyCapability(caps, slot);
    if (slot.family) return <FamilyVoiceFacts slot={slot} family={slot.family} fam={fam} />;
    const template = isTemplateSlot(slot);
    const facts: string[] = [];
    facts.push(template ? t("voice_changer.voice_engine_sigil") : t("voice_changer.voice_engine_onnx"));
    if (slot.version) facts.push(slot.version);
    const sr = sampleRateLabel(slot.sample_rate);
    if (sr) facts.push(sr);
    if (slot.f0 === false) facts.push(t("voice_changer.voice_fact_no_pitch"));
    // A slot with a problem may have no readable manifest: do not guess its encoder and vocoder.
    if (template && !slot.problem) {
        facts.push(embedderDisplayName(slot.embedder, slot.version));
        facts.push(vocoderDisplayName(slot.vocoder));
    }
    if (slot.has_index) facts.push(t("voice_changer.voice_fact_index"));
    if (slot.speaker_count != null && slot.speaker_count > 1) facts.push(t("voice_changer.voice_fact_speakers", { count: slot.speaker_count }));
    return (
        <span className="flex min-w-0 flex-wrap gap-1">
            {facts.map((f, i) => (
                <span key={i} className="badge badge-ghost badge-xs whitespace-nowrap">
                    {f}
                </span>
            ))}
        </span>
    );
};

/** Facts of a P5 family voice: its family, labels, reference voice and sample rate. */
const FamilyVoiceFacts: FC<{ slot: VcSlotInfo; family: VcFamilyId; fam: ReturnType<typeof voiceFamilyCapability> }> = ({ slot, family, fam }) => {
    const { t } = useTranslation();
    const lib = useSnapshot(voiceLibraryUi);
    const vc = useSnapshot(window.ApiServer.voiceChanger.state).voiceChanger;
    const selected = sameSlotPath(slot.path, vc.rvcSlotDir);
    // The selected voice's reference is the live one (Rust saves it to its slot.json); the others' come from the listing.
    const reference = (selected ? vc.referenceVoiceId : slot.reference_voice_id ?? "").trim();
    const facts: string[] = [familyShortName(t, family)];
    const sr = sampleRateLabel(slot.sample_rate);
    if (sr) facts.push(sr);
    if (slot.has_index) facts.push(t("voice_changer.voice_fact_index"));
    return (
        <span className="flex min-w-0 flex-wrap gap-1">
            {facts.map((f, i) => (
                <span key={i} className="badge badge-ghost badge-xs whitespace-nowrap">
                    {f}
                </span>
            ))}
            <FamilyBadges fam={{ experimental: slot.experimental === true || !!fam?.experimental, zero_shot: !!fam?.zero_shot, singing: !!fam?.singing }} />
            {reference && fam?.zero_shot ? (
                <span className="badge badge-ghost badge-xs max-w-full truncate whitespace-nowrap">
                    {t("voice_changer.family_fact_reference", { name: referenceName(reference, lib.voices) })}
                </span>
            ) : null}
        </span>
    );
};

type Panel = "actions" | "rename" | "merge" | "delete" | null;

const VoiceRow: FC<{
    slot: VcSlotInfo;
    slots: readonly VcSlotInfo[];
    selected: boolean;
    inUse: boolean;
    busy: string | undefined;
    manage: boolean;
    slotFormats: readonly string[];
    hotkey: number | null;
}> = ({ slot, slots, selected, inUse, busy, manage, slotFormats, hotkey }) => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const lib = vcApi.library;
    const [panel, setPanel] = useState<Panel>(null);
    const [name, setName] = useState(slot.name);
    const partners = useMemo(() => mergePartners(slots, slot), [slots, slot]);
    const [partner, setPartner] = useState<string>("");
    const [weight, setWeight] = useState(0.5);
    const [mergeName, setMergeName] = useState("");
    const problem = slotProblem(slot, slotFormats);
    const partnerSlot = partners.find((p) => p.id === partner) ?? partners[0];
    const caps = useVcCapabilities();
    const ep = useSnapshot(vcApi.state).voiceChanger.rvcExecutionProvider;
    const fam = voiceFamilyCapability(caps, slot);
    const wrongEp = !problem && !!fam && !familyRunsOnEp(fam, ep);

    const fail = (e: unknown) => toast.error(importErrorText(t, e));

    const select = () => {
        if (selected || problem) return;
        void vcApi.selectVoice(slot);
    };

    return (
        <li
            className={classNames(
                "rounded-lg border",
                selected ? "border-primary/60 bg-primary/5" : "border-base-content/10",
                problem && "opacity-80",
            )}
        >
            <div className="flex min-w-0 items-start gap-1.5 p-1.5">
                <button
                    type="button"
                    className="flex min-w-0 flex-1 flex-col items-start gap-1 rounded-md px-1 py-0.5 text-start hover:bg-base-content/5 disabled:cursor-default"
                    aria-pressed={selected}
                    disabled={!!problem && !selected}
                    onClick={select}
                    title={slot.path}
                >
                    <span className="flex w-full min-w-0 items-center gap-1.5">
                        {selected ? <RiCheckboxCircleFill className="shrink-0 text-primary" aria-hidden /> : null}
                        <span className="min-w-0 flex-1 truncate text-sm font-semibold">{slot.name}</span>
                        {hotkey !== null ? (
                            <span className="badge badge-outline badge-xs shrink-0">{t("voice_changer.voice_hotkey_badge", { n: hotkey + 1 })}</span>
                        ) : null}
                        {busy ? <span className="loading loading-spinner loading-xs shrink-0" aria-label={t(`voice_changer.voice_busy_${busy}`)} /> : null}
                    </span>
                    <VoiceFacts slot={slot} />
                    {problem ? <VcNote tone="warn">{t("voice_changer.voice_problem", { problem })}</VcNote> : null}
                    {wrongEp && fam ? <VcNote tone="warn">{t("voice_changer.family_row_runs_on", { eps: familyEpList(fam) })}</VcNote> : null}
                </button>
                {manage ? (
                    <Tooltip content={t("voice_changer.voice_actions")}>
                        <button
                            type="button"
                            className="btn btn-ghost btn-xs btn-square"
                            aria-label={t("voice_changer.voice_actions")}
                            aria-expanded={panel !== null}
                            onClick={() => setPanel((p) => (p === null ? "actions" : null))}
                        >
                            <RiMoreFill />
                        </button>
                    </Tooltip>
                ) : null}
            </div>
            {panel !== null ? (
                <div className="flex flex-col gap-2 border-t border-base-content/10 p-2">
                    {panel === "actions" ? (
                        <div className="flex flex-wrap gap-1">
                            <button type="button" className="btn btn-xs gap-1" onClick={() => { setName(slot.name); setPanel("rename"); }}>
                                <RiEdit2Line aria-hidden /> {t("voice_changer.voice_rename")}
                            </button>
                            <button
                                type="button"
                                className="btn btn-xs gap-1"
                                disabled={!!busy}
                                onClick={() => {
                                    setPanel(null);
                                    lib.duplicate(slot.id)
                                        .then((info) => info && toast.success(t("voice_changer.voice_duplicated", { name: info.name })))
                                        .catch(fail);
                                }}
                            >
                                <RiFileCopyLine aria-hidden /> {t("voice_changer.voice_duplicate")}
                            </button>
                            {isTemplateSlot(slot) ? (
                                <button
                                    type="button"
                                    className="btn btn-xs gap-1"
                                    disabled={partners.length === 0}
                                    title={partners.length === 0 ? t("voice_changer.voice_merge_none") : undefined}
                                    onClick={() => {
                                        setPartner(partners[0]?.id ?? "");
                                        setWeight(0.5);
                                        setMergeName(partners[0] ? `${slot.name} + ${partners[0].name}` : "");
                                        setPanel("merge");
                                    }}
                                >
                                    <RiGitMergeLine aria-hidden /> {t("voice_changer.voice_merge")}
                                </button>
                            ) : null}
                            <button type="button" className="btn btn-xs btn-error btn-outline gap-1" onClick={() => setPanel("delete")}>
                                <RiDeleteBin6Line aria-hidden /> {t("voice_changer.voice_delete")}
                            </button>
                        </div>
                    ) : null}
                    {panel === "rename" ? (
                        <form
                            className="flex items-center gap-1"
                            onSubmit={(e) => {
                                e.preventDefault();
                                const next = name.trim();
                                if (!next || next === slot.name) {
                                    setPanel(null);
                                    return;
                                }
                                lib.rename(slot.id, next)
                                    .then(() => setPanel(null))
                                    .catch(fail);
                            }}
                        >
                            <input
                                className="input input-bordered input-sm min-w-0 flex-1"
                                value={name}
                                maxLength={120}
                                aria-label={t("voice_changer.voice_rename")}
                                autoFocus
                                onChange={(e) => setName(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === "Escape") setPanel(null);
                                }}
                            />
                            <button type="submit" className="btn btn-primary btn-sm" disabled={!name.trim() || !!busy}>
                                {t("voice_changer.voice_save")}
                            </button>
                            <button type="button" className="btn btn-ghost btn-sm btn-square" aria-label={t("voice_changer.info_close_overlay")} onClick={() => setPanel(null)}>
                                <RiCloseLine />
                            </button>
                        </form>
                    ) : null}
                    {panel === "delete" ? (
                        <div className="flex flex-col gap-1.5">
                            <VcNote tone={inUse ? "warn" : "muted"}>
                                {inUse ? t("voice_changer.voice_delete_in_use") : t("voice_changer.voice_delete_confirm", { name: slot.name })}
                            </VcNote>
                            <div className="flex justify-end gap-1">
                                <button type="button" className="btn btn-ghost btn-xs" onClick={() => setPanel(null)}>
                                    {t("voice_changer.voice_cancel")}
                                </button>
                                <button
                                    type="button"
                                    className="btn btn-error btn-xs"
                                    disabled={inUse || !!busy}
                                    onClick={() => {
                                        lib.remove(slot.id)
                                            .then(() => toast.info(t("voice_changer.voice_deleted", { name: slot.name })))
                                            .catch(fail);
                                    }}
                                >
                                    {t("voice_changer.voice_delete")}
                                </button>
                            </div>
                        </div>
                    ) : null}
                    {panel === "merge" && partnerSlot ? (
                        <div className="flex flex-col gap-1.5">
                            <VcNote>{t("voice_changer.voice_merge_help")}</VcNote>
                            <select
                                className="select select-bordered select-sm w-full"
                                aria-label={t("voice_changer.voice_merge_with")}
                                value={partnerSlot.id}
                                onChange={(e) => {
                                    const p = partners.find((x) => x.id === e.target.value);
                                    setPartner(e.target.value);
                                    if (p) setMergeName(`${slot.name} + ${p.name}`);
                                }}
                            >
                                {partners.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        {p.name}
                                    </option>
                                ))}
                            </select>
                            <input
                                type="range"
                                min={0}
                                max={1}
                                step={0.05}
                                value={weight}
                                aria-label={t("voice_changer.voice_merge_weight")}
                                className="range range-sm range-primary"
                                onChange={(e) => setWeight(parseFloat(e.target.value))}
                            />
                            <div className="flex justify-between gap-2 text-xs text-base-content/60">
                                <span className="min-w-0 truncate">{`${slot.name} ${Math.round(weight * 100)}%`}</span>
                                <span className="min-w-0 truncate text-end">{`${partnerSlot.name} ${Math.round((1 - weight) * 100)}%`}</span>
                            </div>
                            <input
                                className="input input-bordered input-sm w-full"
                                value={mergeName}
                                maxLength={120}
                                aria-label={t("voice_changer.voice_merge_name")}
                                placeholder={t("voice_changer.voice_merge_name")}
                                onChange={(e) => setMergeName(e.target.value)}
                            />
                            <div className="flex justify-end gap-1">
                                <button type="button" className="btn btn-ghost btn-xs" onClick={() => setPanel(null)}>
                                    {t("voice_changer.voice_cancel")}
                                </button>
                                <button
                                    type="button"
                                    className="btn btn-primary btn-xs"
                                    disabled={!mergeName.trim() || !!busy}
                                    onClick={() => {
                                        lib.merge({ a: slot.id, b: partnerSlot.id, weight_a: weight, name: mergeName.trim() })
                                            .then((info) => {
                                                setPanel(null);
                                                if (info) toast.success(t("voice_changer.voice_merged", { name: info.name }));
                                            })
                                            .catch(fail);
                                    }}
                                >
                                    {t("voice_changer.voice_merge")}
                                </button>
                            </div>
                        </div>
                    ) : null}
                </div>
            ) : null}
        </li>
    );
};

/** Whether a physical drop position (Tauri) lies inside `el`. */
function positionInside(el: HTMLElement | null, pos: { x: number; y: number }): boolean {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const x = pos.x / ratio;
    const y = pos.y / ratio;
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
}

/**
 * Drop `.pth` / `.zip` voices here, or pick them. A dropped file that only one available family imports (not a
 * `.pth` / `.zip`) goes to that family's "new voice" form (`onFamilyFile`).
 */
const ImportDropZone: FC<{ onFamilyFile: (family: VcFamilyId, path: string) => void }> = ({ onFamilyFile }) => {
    const { t } = useTranslation();
    const lib = window.ApiServer.voiceChanger.library;
    const zone = useRef<HTMLDivElement>(null);
    const [drag, setDrag] = useState<"none" | "window" | "over">("none");

    const importPaths = useCallback(
        (paths: readonly string[]) => {
            const plan = lib.importFiles(paths);
            const caps = window.ApiServer.voiceChanger.state.capabilities;
            const familyFiles = plan.ignored.flatMap((p) => {
                const family = familyForImportFile(p, caps);
                return family ? [{ family, path: p }] : [];
            });
            const ignored = plan.ignored.filter((p) => !familyFiles.some((f) => f.path === p));
            if (ignored.length > 0) {
                toast.warning(t("voice_changer.import_ignored", { files: ignored.map((p) => p.split(/[\\/]/).pop()).join(", ") }));
            }
            if (plan.unpairedIndexes.length > 0) toast.info(t("voice_changer.import_index_unpaired"));
            // One form at a time: the first family file opens it; more are named so the user can add them next.
            if (familyFiles.length > 0) {
                onFamilyFile(familyFiles[0].family, familyFiles[0].path);
                if (familyFiles.length > 1) {
                    toast.info(t("voice_changer.family_drop_more", { files: familyFiles.slice(1).map((f) => f.path.split(/[\\/]/).pop()).join(", ") }));
                }
            }
        },
        [lib, t, onFamilyFile],
    );

    useEffect(() => {
        if (!isTauri()) return;
        let unlisten: (() => void) | null = null;
        let disposed = false;
        getCurrentWebview()
            .onDragDropEvent((ev) => {
                const p = ev.payload;
                if (p.type === "leave") setDrag("none");
                else if (p.type === "enter" || p.type === "over") setDrag(positionInside(zone.current, p.position) ? "over" : "window");
                else if (p.type === "drop") {
                    setDrag("none");
                    if (positionInside(zone.current, p.position)) importPaths(p.paths);
                }
            })
            .then((u) => {
                if (disposed) u();
                else unlisten = u;
            })
            .catch(() => {});
        return () => {
            disposed = true;
            unlisten?.();
        };
    }, [importPaths]);

    const pick = async () => {
        const picked = await open({
            multiple: true,
            directory: false,
            title: t("voice_changer.import_pick_title"),
            filters: [{ name: t("voice_changer.import_filter_voices"), extensions: [...VOICE_FILE_EXTENSIONS, ...INDEX_FILE_EXTENSIONS] }],
        });
        if (!picked) return;
        importPaths(Array.isArray(picked) ? picked : [picked]);
    };

    return (
        <div
            ref={zone}
            className={classNames(
                "flex flex-col items-center gap-1.5 rounded-lg border-2 border-dashed p-3 text-center text-xs transition-colors",
                drag === "over" ? "border-primary bg-primary/10" : drag === "window" ? "border-primary/50 bg-primary/5" : "border-base-content/20",
            )}
        >
            <RiDragDropLine className="text-xl text-base-content/50" aria-hidden />
            <span className="text-base-content/70">
                {drag === "none" ? t("voice_changer.import_drop_hint") : t("voice_changer.import_drop_here")}
            </span>
            <button type="button" className="btn btn-sm gap-1" onClick={() => void pick()}>
                <RiAddLine aria-hidden /> {t("voice_changer.import_pick")}
            </button>
        </div>
    );
};

const ImportJobRow: FC<{ job: VcImportJob; selectedDir: string; onRetry: (request: VcFamilySlotRequest) => void }> = ({ job, selectedDir, onRetry }) => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const title =
        job.kind === "default_voice"
            ? t("voice_changer.default_voice")
            : job.kind === "family" && job.family
              ? job.fileName
                  ? t("voice_changer.family_job_title", { family: familyShortName(t, job.family), name: job.fileName })
                  : familyName(t, job.family)
              : job.fileName;
    const finished = isFinished(job.stage);
    return (
        <li className="flex flex-col gap-1 rounded-lg border border-base-content/10 p-2 text-xs">
            <div className="flex min-w-0 items-center gap-1.5">
                <span className="min-w-0 flex-1 truncate font-semibold" title={job.source || undefined}>
                    {title}
                </span>
                {finished ? (
                    <button
                        type="button"
                        className="btn btn-ghost btn-xs btn-square"
                        aria-label={t("voice_changer.import_dismiss")}
                        onClick={() => vcApi.library.dismiss(job.key)}
                    >
                        <RiCloseLine />
                    </button>
                ) : (
                    <button type="button" className="btn btn-ghost btn-xs" disabled={job.cancelRequested} onClick={() => void vcApi.library.cancel(job.key)}>
                        {t("voice_changer.import_cancel")}
                    </button>
                )}
            </div>
            {!finished ? (
                <>
                    <progress
                        className="progress progress-primary h-1.5 w-full"
                        value={job.stage === "queued" ? undefined : Math.round(job.fraction * 1000)}
                        max={1000}
                        aria-label={title}
                    />
                    <span className="text-base-content/60">{job.message || t(`voice_changer.import_stage_${job.stage}`)}</span>
                </>
            ) : job.stage === "done" && job.slot ? (
                <>
                    <span className="text-success">
                        {job.kind === "family"
                            ? t("voice_changer.family_job_done", { name: job.slot.name })
                            : job.cached
                              ? t("voice_changer.import_done_cached", { name: job.slot.name })
                              : t("voice_changer.import_done", { name: job.slot.name })}
                    </span>
                    {job.warnings.map((w) => (
                        <VcNote key={w} tone="warn">
                            {w}
                        </VcNote>
                    ))}
                    {!sameSlotPath(job.slot.path, selectedDir) && !job.slot.problem ? (
                        <button type="button" className="btn btn-primary btn-xs self-start" onClick={() => void vcApi.selectVoice(job.slot!)}>
                            {t("voice_changer.import_use_voice")}
                        </button>
                    ) : null}
                </>
            ) : (
                <>
                    <VcNote tone="error">{importErrorText(t, job.error ?? "")}</VcNote>
                    {/* The "new voice" form closed when the job was queued: reopen it with what was sent. */}
                    {job.kind === "family" && job.request ? (
                        <button
                            type="button"
                            className="btn btn-xs self-start"
                            onClick={() => {
                                const request = JSON.parse(JSON.stringify(job.request)) as VcFamilySlotRequest;
                                vcApi.library.dismiss(job.key);
                                onRetry(request);
                            }}
                        >
                            {t("voice_changer.family_job_retry")}
                        </button>
                    ) : null}
                </>
            )}
        </li>
    );
};

/**
 * First run: the base models and the default voice in one card. The Download is the user's click;
 * the default voice is imported as soon as its file is installed (or with the button).
 */
const StarterCard: FC<{ hasVoices: boolean; ids: readonly string[] }> = ({ hasVoices, ids }) => {
    const { t } = useTranslation();
    const lib = window.ApiServer.voiceChanger.library;
    const jobs = useSnapshot(lib.state).jobs;
    const models = useSnapshot(window.ApiServer.models.state);
    const defaultState = models.states[DEFAULT_VOICE_ID]?.state;
    const defaultInstalled = defaultState === "installed";
    const importing = jobs.some((j) => j.kind === "default_voice" && !isFinished(j.stage));
    /** The default voice was not installed while this card was open: import it once it is. */
    const sawMissing = useRef(false);
    useEffect(() => {
        if (!models.loaded) return;
        if (!defaultInstalled) {
            sawMissing.current = true;
            return;
        }
        if (sawMissing.current && !hasVoices && !importing) {
            sawMissing.current = false;
            lib.importDefaultVoice();
        }
    }, [models.loaded, defaultInstalled, hasVoices, importing, lib]);

    return (
        <div className="flex flex-col gap-2 rounded-lg border border-primary/30 bg-primary/5 p-2.5">
            <span className="text-sm font-semibold">{t("voice_changer.starter_title")}</span>
            <VcNote>{t("voice_changer.starter_body")}</VcNote>
            <ModelRequirement ids={ids} reason={t("voice_changer.starter_needs")} />
            {defaultInstalled && !hasVoices ? (
                <button type="button" className="btn btn-primary btn-sm self-start" disabled={importing} onClick={() => lib.importDefaultVoice()}>
                    {importing ? t("voice_changer.starter_adding") : t("voice_changer.starter_add_default")}
                </button>
            ) : null}
        </div>
    );
};

/** The voice library of RVC mode. */
export const VoiceLibrary: FC = () => {
    const { t } = useTranslation();
    const vcApi = window.ApiServer.voiceChanger;
    const lib = useSnapshot(vcApi.library.state);
    const snap = useSnapshot(vcApi.state);
    const slotDir = snap.voiceChanger.rvcSlotDir;
    const caps = snap.capabilities;
    const [custom, setCustom] = useState(false);
    const [modelsDir, setModelsDir] = useState("");
    const hotkeyIds = useHotkeyVoiceIds();
    /** The "new voice" form: closed, or open (with the family and file to start with, or a failed request to retry). */
    const [creator, setCreator] = useState<{ key: number; family?: VcFamilyId; source?: string; request?: VcFamilySlotRequest } | null>(null);
    const openCreator = useCallback((family?: VcFamilyId, source?: string) => setCreator({ key: Date.now(), family, source }), []);
    const retryFamilyVoice = useCallback((request: VcFamilySlotRequest) => setCreator({ key: Date.now(), family: request.family, request }), []);
    const familiesAvailable = (caps.families ?? []).some((f) => f.available);
    // Reference names of zero-shot voices come from the shared voice library.
    useVoiceLibraryHost(!!lib.slots?.some((s) => !!s.family));

    useEffect(() => {
        if (!isTauri()) return;
        void vcApi.library.refresh();
        let cancelled = false;
        appDataDir()
            .then((d) => join(d, "voice-changer", "models"))
            .then((p) => {
                if (!cancelled) setModelsDir(p);
            })
            .catch(() => {});
        return () => {
            cancelled = true;
        };
    }, [vcApi]);

    if (!isTauri()) return <VcNote>{t("voice_changer.voices_desktop_only")}</VcNote>;

    const slots = lib.slots as VcSlotInfo[] | null;
    const hasVoices = !!slots && slots.length > 0;
    const listed = hasVoices ? slots!.some((s) => sameSlotPath(s.path, slotDir)) : false;
    const showFolder = custom || (slotDir.trim() !== "" && !listed) || (slots !== null && !hasVoices && !lib.slotStore);
    const manage = lib.slotStore;
    const canImport = lib.slotStore && caps.importer;
    const running = snap.isRunning && snap.voiceChanger.mode === "rvc";
    const vc = snap.voiceChanger;
    const starterIds = starterModelIds({
        pitchExtractor: vc.pitchExtractor,
        rvcExecutionProvider: vc.rvcExecutionProvider,
        fcpePath: vc.rvcFcpePath,
        rmvpePath: vc.rvcRmvpePath,
        capabilities: caps as typeof vcApi.state.capabilities,
    });

    return (
        <div className="flex flex-col gap-2">
            {slots !== null && !hasVoices && canImport ? <StarterCard hasVoices={hasVoices} ids={starterIds} /> : null}
            {hasVoices ? (
                <ul className="flex flex-col gap-1.5" aria-label={t("voice_changer.voices_list")}>
                    {slots!.map((s) => {
                        const selected = sameSlotPath(s.path, slotDir);
                        const hk = hotkeyIds.indexOf(s.id);
                        return (
                            <VoiceRow
                                key={s.id}
                                slot={s}
                                slots={slots!}
                                selected={selected}
                                inUse={selected && running}
                                busy={lib.busy[s.id]}
                                manage={manage}
                                slotFormats={caps.slot_formats}
                                hotkey={hk >= 0 ? hk : null}
                            />
                        );
                    })}
                </ul>
            ) : null}
            {slots !== null && !hasVoices && !canImport ? (
                <VcNote>
                    {t("voice_changer.rvc_voices_empty", {
                        path: modelsDir || "%AppData%/com.sigil.app/voice-changer/models",
                    })}
                </VcNote>
            ) : null}
            {lib.listError ? <VcNote tone="warn">{t("voice_changer.voices_list_failed", { message: lib.listError })}</VcNote> : null}
            {canImport ? <ImportDropZone onFamilyFile={openCreator} /> : null}
            {creator ? (
                <FamilyCreator
                    key={creator.key}
                    initial={creator.family ? { family: creator.family, source: creator.source, request: creator.request ?? null } : null}
                    onClose={() => setCreator(null)}
                />
            ) : lib.slotStore && familiesAvailable ? (
                <button type="button" className="btn btn-sm gap-1 self-start" onClick={() => openCreator()}>
                    <RiAddLine aria-hidden /> {t("voice_changer.family_new")}
                </button>
            ) : null}
            {lib.jobs.length > 0 ? (
                <ul className="flex flex-col gap-1.5" aria-label={t("voice_changer.import_jobs")}>
                    {(lib.jobs as VcImportJob[]).map((j) => (
                        <ImportJobRow key={j.key} job={j} selectedDir={slotDir} onRetry={retryFamilyVoice} />
                    ))}
                </ul>
            ) : null}
            <div className="flex items-center gap-1">
                <button type="button" className="btn btn-ghost btn-xs" onClick={() => setCustom((c) => !c)} aria-expanded={showFolder}>
                    {t("voice_changer.rvc_voice_custom")}
                </button>
                <span className="flex-1" />
                <Tooltip content={t("voice_changer.rvc_voices_rescan")}>
                    <button
                        type="button"
                        className="btn btn-ghost btn-xs btn-square"
                        aria-label={t("voice_changer.rvc_voices_rescan")}
                        disabled={lib.listing}
                        onClick={() => void vcApi.library.refresh()}
                    >
                        <RiRefreshLine className={classNames(lib.listing && "animate-spin")} />
                    </button>
                </Tooltip>
            </div>
            {showFolder ? (
                <VcPathField
                    labelKey="voice_changer.rvc_voice_folder"
                    infoKey="voice_changer.rvc_slot_help"
                    value={slotDir}
                    directory
                    onChange={(path) => {
                        vcApi.state.voiceChanger.rvcSlotDir = path;
                    }}
                    onCommit={() => void vcApi.selectModelSlot(vcApi.state.voiceChanger.rvcSlotDir)}
                />
            ) : null}
            <VoiceHotkeys />
            {hasVoices && canImport ? (
                <VcDetails titleKey="voice_changer.starter_models_title">
                    <ModelRequirement ids={starterIds} reason={t("voice_changer.starter_needs")} />
                    <button type="button" className="btn btn-xs self-start" onClick={() => vcApi.library.importDefaultVoice()}>
                        {t("voice_changer.starter_add_default")}
                    </button>
                </VcDetails>
            ) : null}
        </div>
    );
};

/** Slot ids of voice hotkeys 1-4 (empty until the hotkeys are wired in the app's settings). */
function useHotkeyVoiceIds(): string[] {
    const state = useSnapshot(window.ApiServer.state) as unknown as Record<string, unknown>;
    const raw = state.vcHotkeyVoices;
    return useMemo(() => (Array.isArray(raw) ? raw.map((v) => (typeof v === "string" ? v : "")) : []), [raw]);
}
