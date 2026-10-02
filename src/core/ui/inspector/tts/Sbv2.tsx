import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { SBV2_DEVICES, SBV2_THREADS } from "@/core/services/tts/schema/sbv2";
import {
  SBV2_IMPORT_EXTENSIONS,
  SBV2_REQUIREMENTS,
  importSbv2Model,
  isNeutralStyle,
  pickSpeaker,
  pickStyle,
  previewSbv2,
  refreshSbv2Models,
  releaseSbv2IfIdle,
  removeSbv2Model,
  sbv2Library,
  sbv2Speakers,
  type Sbv2Model,
} from "@/core/services/tts/services/sbv2";
import { errorMessage } from "@/core/services/tts/types";
import { formatBytes } from "@/core/services/diagnostics/format";
import type { RustModelEntryStatus } from "@/types";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import classNames from "classnames";
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiAddLine, RiDeleteBinLine, RiDragDropLine, RiPlayLine, RiRefreshLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { InputRange, InputSelect } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type Sbv2State = TTS_State["sbv2"];

/** Sample line of the preview (the voices speak Japanese). */
const PREVIEW_TEXT = "こんにちは。これが私の声です。";

/** Whether a physical drop position (Tauri) lies inside `el`. */
function positionInside(el: HTMLElement | null, pos: { x: number; y: number }): boolean {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const x = pos.x / ratio;
  const y = pos.y / ratio;
  return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
}

/** Drop `.aivmx` voices (or an ONNX export's folder) here, or pick a file. */
const ImportZone: FC<{ disabled: boolean; onImported: (m: Sbv2Model) => void }> = ({ disabled, onImported }) => {
  const { t } = useTranslation();
  const zone = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<"none" | "window" | "over">("none");
  const lib = useSnapshot(sbv2Library);

  const importPaths = useCallback(
    async (paths: readonly string[]) => {
      if (!paths.length) return;
      if (paths.length > 1) toast.info(t("tts.sbv2_import_one_at_a_time"));
      try {
        const m = await importSbv2Model(paths[0]);
        toast.success(t("tts.sbv2_imported", { name: m.name }));
        onImported(m);
      } catch (e) {
        toast.error(t("tts.sbv2_import_failed", { error: errorMessage(e) }));
      }
    },
    [onImported, t],
  );

  useEffect(() => {
    if (!isTauri() || disabled) return;
    let unlisten: (() => void) | null = null;
    let disposed = false;
    getCurrentWebview()
      .onDragDropEvent((ev) => {
        const p = ev.payload;
        if (p.type === "leave") setDrag("none");
        else if (p.type === "enter" || p.type === "over") setDrag(positionInside(zone.current, p.position) ? "over" : "window");
        else if (p.type === "drop") {
          setDrag("none");
          if (positionInside(zone.current, p.position)) void importPaths(p.paths);
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
  }, [importPaths, disabled]);

  const pick = async () => {
    const picked = await open({
      multiple: false,
      directory: false,
      title: t("tts.sbv2_import_title"),
      filters: [{ name: t("tts.sbv2_import_filter"), extensions: [...SBV2_IMPORT_EXTENSIONS] }],
    });
    if (typeof picked === "string" && picked) await importPaths([picked]);
  };

  return (
    <div
      ref={zone}
      className={classNames(
        "flex flex-col items-center gap-1.5 rounded-lg border-2 border-dashed p-3 text-center text-xs transition-colors",
        drag === "over" ? "border-primary bg-primary/10" : drag === "window" ? "border-primary/50 bg-primary/5" : "border-base-content/20",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <RiDragDropLine className="text-xl text-base-content/50" aria-hidden />
      <span className="text-base-content/70">{drag === "none" ? t("tts.sbv2_drop_hint") : t("tts.sbv2_drop_here")}</span>
      <button type="button" className="btn btn-sm gap-1" disabled={lib.importing} onClick={() => void pick()}>
        {lib.importing ? <span className="loading loading-spinner loading-xs" /> : <RiAddLine aria-hidden />}
        {lib.importing ? t("tts.sbv2_importing") : t("tts.sbv2_import")}
      </button>
    </div>
  );
};

/**
 * Panel of the `sbv2` backend (P6-G): Style-Bert-VITS2 / AivisSpeech voices on Sigil's own engine. The download card
 * (the Japanese BERT and dictionary), the imported voices (drop an `.aivmx`, pick one, delete one), speaker and style,
 * style strength, tempo variation and speed (next sentence), the device (processor / GPU) and threads (restart), a
 * preview, and the sink output device and volume.
 */
export const Sbv2: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.sbv2);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const lib = useSnapshot(sbv2Library);
  const models = (lib.models ?? []) as Sbv2Model[];
  const [previewing, setPreviewing] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const tauri = isTauri();
  useModelsLoaded();
  const storeStates = useSnapshot(window.ApiServer.models.state).states as Record<string, RustModelEntryStatus | undefined>;
  const ready = SBV2_REQUIREMENTS.every((id) => storeStates[id]?.state === "installed");

  const up = <K extends keyof Sbv2State>(key: K, v: Sbv2State[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.sbv2[key] = v;
    });

  useEffect(() => {
    if (tauri) void refreshSbv2Models().catch(() => {});
  }, [tauri]);

  const stopped = service.status === ServiceNetworkState.disconnected;
  const model = models.find((m) => m.id === data.model) ?? null;
  const speaker = pickSpeaker(model, data.speaker);
  const style = pickStyle(speaker, data.style);

  const modelOptions = useMemo(() => {
    const options = models.map((m) => ({ label: m.name, value: m.id }));
    if (!data.model) options.unshift({ label: t("tts.sbv2_choose_voice"), value: "" });
    else if (!models.some((m) => m.id === data.model)) options.push({ label: t("tts.sbv2_missing_voice", { id: data.model }), value: data.model });
    return options;
  }, [models, data.model, t]);

  const chooseModel = (id: string) => {
    window.ApiServer.patchService("tts", (s) => {
      const c = s.data.sbv2;
      if (c.model !== id) {
        c.model = id;
        // Speakers and styles belong to a model.
        c.speaker = "";
        c.style = "Neutral";
      }
    });
  };

  const onImported = useCallback((m: Sbv2Model) => {
    // A first voice (or none chosen yet) becomes the voice.
    if (!window.ApiServer.state.services.tts.data.sbv2.model) {
      window.ApiServer.patchService("tts", (s) => {
        s.data.sbv2.model = m.id;
        s.data.sbv2.speaker = "";
        s.data.sbv2.style = "Neutral";
      });
    }
  }, []);

  const remove = async () => {
    if (!model) return;
    if (!confirmRemove) {
      setConfirmRemove(true);
      return;
    }
    setConfirmRemove(false);
    try {
      await removeSbv2Model(model.id);
      chooseModel("");
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const preview = async () => {
    const state = window.ApiServer.state.services.tts.data;
    if (!state.sbv2.model) return;
    setPreviewing(true);
    try {
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.sbv2 }, window.ApiServer.state.audioOutputDevice ?? "");
      await previewSbv2(state.sbv2, PREVIEW_TEXT, target);
    } catch (e) {
      toast.error(t("tts.sbv2_preview_failed", { error: errorMessage(e) }));
    } finally {
      setPreviewing(false);
      releaseSbv2IfIdle();
    }
  };

  return (
    <>
      <Inspector.SubHeader>{t("tts.sbv2_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.sbv2_desc")}</Inspector.Description>
      {!tauri && <div className="text-xs text-base-content/60">{t("tts.local_neural_desktop_only")}</div>}
      {tauri && <ModelRequirement ids={SBV2_REQUIREMENTS} reason={t("tts.sbv2_requirement")} hideWhenReady />}

      <Inspector.Deactivatable active={stopped}>
        <div className="flex gap-2 items-end">
          <div className="flex-1 min-w-0">
            <InputSelect label="tts.sbv2_voice" value={data.model} options={modelOptions} onValueChange={chooseModel} />
          </div>
          <button
            type="button"
            className="btn btn-sm btn-ghost mb-1"
            title={t("tts.sbv2_refresh")}
            aria-label={t("tts.sbv2_refresh")}
            disabled={!tauri || lib.loading}
            onClick={() => void refreshSbv2Models().catch((e) => toast.error(errorMessage(e)))}
          >
            <RiRefreshLine aria-hidden />
          </button>
          {model && (
            <button
              type="button"
              className={classNames("btn btn-sm mb-1 gap-1", confirmRemove ? "btn-error" : "btn-ghost")}
              onClick={() => void remove()}
              onBlur={() => setConfirmRemove(false)}
              title={t("tts.sbv2_remove")}
            >
              <RiDeleteBinLine aria-hidden />
              {confirmRemove && t("tts.sbv2_remove_confirm")}
            </button>
          )}
        </div>
        {lib.error && <div className="text-xs text-error">{t("tts.sbv2_list_failed", { error: lib.error })}</div>}
        {models.length === 0 && lib.models !== null && <div className="text-xs text-base-content/60">{t("tts.sbv2_no_voices")}</div>}
        {model && (
          <div className="text-xs text-base-content/60 flex flex-col gap-0.5">
            <span>
              {[model.architecture, model.sampleRate ? `${(model.sampleRate / 1000).toFixed(1)} kHz` : null, model.sizeBytes ? formatBytes(model.sizeBytes) : null]
                .filter(Boolean)
                .join(" · ")}
            </span>
            {model.description && <span>{model.description}</span>}
            {!!model.creators?.length && <span>{t("tts.sbv2_creators", { names: model.creators.join(", ") })}</span>}
          </div>
        )}
      </Inspector.Deactivatable>
      {/* Adding a voice never disturbs the running one. */}
      <ImportZone disabled={!tauri} onImported={onImported} />
      <Inspector.Deactivatable active={stopped}>
        <InputSelect
          label="tts.sbv2_device"
          value={data.executionProvider}
          options={SBV2_DEVICES.map((d) => ({ label: t(`tts.sbv2_device_${d}`), value: d }))}
          onValueChange={(v) => up("executionProvider", v as Sbv2State["executionProvider"])}
        />
        {data.executionProvider === "cpu" && (
          <InputSelect
            label="tts.local_neural_threads"
            value={data.threads}
            options={SBV2_THREADS.map((v) => ({ label: v === "auto" ? t("tts.local_neural_threads_auto") : v, value: v }))}
            onValueChange={(v) => up("threads", v as Sbv2State["threads"])}
          />
        )}
      </Inspector.Deactivatable>

      {model && sbv2Speakers(model).length > 1 && (
        <InputSelect
          label="tts.sbv2_speaker"
          value={speaker?.name ?? ""}
          options={sbv2Speakers(model).map((s) => ({ label: s.name, value: s.name }))}
          onValueChange={(v) => {
            up("speaker", v);
            up("style", "Neutral");
          }}
        />
      )}
      {speaker && (
        <InputSelect
          label="tts.sbv2_style"
          value={style?.name ?? ""}
          options={speaker.styles.map((s) => ({ label: s.name, value: s.name }))}
          onValueChange={(v) => up("style", v)}
        />
      )}
      <InputRange
        label="tts.sbv2_style_weight"
        value={data.styleWeight}
        min="0"
        max="10"
        step="0.1"
        disabled={isNeutralStyle(style)}
        onChange={(e) => up("styleWeight", e.target.value)}
      />
      <div className="text-xs text-base-content/60">{isNeutralStyle(style) ? t("tts.sbv2_style_weight_neutral") : t("tts.sbv2_style_weight_hint")}</div>
      <InputRange label="tts.sbv2_sdp_ratio" value={data.sdpRatio} min="0" max="1" step="0.05" onChange={(e) => up("sdpRatio", e.target.value)} />
      <InputRange label="tts.sbv2_speed" value={data.speed} min="0.5" max="2" step="0.05" onChange={(e) => up("speed", e.target.value)} />
      <div className="flex justify-end">
        <button
          type="button"
          className="btn btn-sm btn-outline gap-1"
          disabled={!tauri || previewing || !ready || !data.model}
          title={ready ? undefined : t("tts.local_neural_preview_needs_download")}
          onClick={() => void preview()}
        >
          {previewing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.local_neural_preview")}
        </button>
      </div>
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />

      <Inspector.SubHeader>{t("tts.openai_speech_advanced")}</Inspector.SubHeader>
      <InputRange label="tts.sbv2_noise" value={data.noise} min="0" max="1.5" step="0.05" onChange={(e) => up("noise", e.target.value)} />
      <InputRange label="tts.sbv2_noise_w" value={data.noiseW} min="0" max="1.5" step="0.05" onChange={(e) => up("noiseW", e.target.value)} />
      <div className="text-xs text-base-content/60">{t("tts.sbv2_noise_hint")}</div>
    </>
  );
};
