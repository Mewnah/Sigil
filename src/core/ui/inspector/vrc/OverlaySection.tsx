import classNames from "classnames";
import type { TFunction } from "i18next";
import { FC, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { OthersEventSource, TextEventSource, type CaptionEventSource } from "@/types";
import { VRC_OVERLAY_ATTACH, type VRC_State } from "@/core/services/vrc/schema";
import type { CaptionWindowState } from "@/core/services/overlay/schema";
import { MAX_CLEAR_AFTER_MS, VR_LIMITS, WINDOW_LIMITS, type VrOverlayStatusView } from "@/core/services/overlay/config";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { InputCheckbox, InputColor, InputRange, InputSelect, InputText } from "../components/input";

/** Catalog id of the OpenVR runtime pack (Rust `vr_overlay::RUNTIME_ID`). */
const OPENVR_RUNTIME_ID = "runtime/openvr";
const OPENVR_IDS = [OPENVR_RUNTIME_ID] as const;
/** Most caption sources a display shows (schema `.max(4)`). */
const MAX_SOURCES = 4;

type VrOverlaySettings = VRC_State["overlay"];

/** The caption sources a display can show, with their labels (displays may show other players). */
const SOURCE_OPTIONS: { value: CaptionEventSource; label: string }[] = [
  { value: TextEventSource.stt, label: "stt.title" },
  { value: TextEventSource.translation, label: "transl.title" },
  { value: TextEventSource.textfield, label: "common.text_source_textfield" },
  { value: TextEventSource.transform, label: "transform.source_label" },
  { value: TextEventSource.any, label: "common.text_source_any" },
  { value: OthersEventSource.stt, label: "common.text_source_others_stt" },
  { value: OthersEventSource.translation, label: "common.text_source_others_translation" },
];

const HINT_TONE = {
  muted: "text-base-content/70",
  warning: "text-warning",
  error: "text-error",
  success: "text-success",
} as const;

const Hint: FC<{ tone?: keyof typeof HINT_TONE; children: ReactNode }> = ({ tone = "muted", children }) => (
  <div className={classNames("text-xs px-1", HINT_TONE[tone])}>{children}</div>
);

/** One toggle per caption source, in a fixed order; at most {@link MAX_SOURCES} at once. */
const SourceToggles: FC<{ value: readonly CaptionEventSource[]; onChange: (value: CaptionEventSource[]) => void }> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const full = value.length >= MAX_SOURCES;
  const toggle = (source: CaptionEventSource, on: boolean) => {
    const next = on ? [...value, source] : value.filter((s) => s !== source);
    onChange(SOURCE_OPTIONS.map((o) => o.value).filter((s) => next.includes(s)).slice(0, MAX_SOURCES));
  };
  return (
    <div className="flex flex-col">
      <span className="text-xs font-medium text-base-content/80 pb-1">{t("overlay.field_sources")}</span>
      {SOURCE_OPTIONS.map((o) => {
        const checked = value.includes(o.value);
        return (
          <InputCheckbox
            key={o.value}
            label={o.label}
            value={checked}
            className={classNames("ps-2", { "opacity-50": !checked && full })}
            onChange={(on) => {
              // The input is controlled: an ignored change leaves it unchecked.
              if (on && !checked && full) return;
              toggle(o.value, on);
            }}
          />
        );
      })}
      {value.length === 0 && <Hint tone="warning">{t("overlay.sources_none")}</Hint>}
      {full && <Hint>{t("overlay.sources_full", { count: MAX_SOURCES })}</Hint>}
    </div>
  );
};

const CaptionWindowSettings: FC = () => {
  const { t } = useTranslation();
  const s = useSnapshot(window.ApiServer.state.captionWindow) as CaptionWindowState;
  const status = useSnapshot(window.ApiServer.overlay.state).window;
  const up = <K extends keyof CaptionWindowState>(key: K, value: CaptionWindowState[K]) => {
    window.ApiServer.state.captionWindow[key] = value;
  };

  return (
    <>
      <Inspector.SubHeader>{t("overlay.window_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("overlay.window_desc")}</Inspector.Description>
      <InputCheckbox label="overlay.window_enable" value={s.enable} onChange={(v) => up("enable", v)} />
      {s.enable && status.error && <Hint tone="error">{t("overlay.window_error", { error: status.error })}</Hint>}
      {s.enable && !status.error && <Hint tone={status.open ? "success" : "muted"}>{t(status.open ? "overlay.window_open" : "overlay.window_opening")}</Hint>}
      <SourceToggles value={s.sources as CaptionEventSource[]} onChange={(v) => up("sources", v)} />
      <InputCheckbox label="overlay.field_interim" value={s.interim} onChange={(v) => up("interim", v)} />
      <InputCheckbox label="overlay.window_always_on_top" value={s.alwaysOnTop} onChange={(v) => up("alwaysOnTop", v)} />
      <InputCheckbox label="overlay.window_click_through" value={s.clickThrough} onChange={(v) => up("clickThrough", v)} />
      <Hint>{t("overlay.window_click_through_desc")}</Hint>
      <InputText label="overlay.window_font_family" value={s.fontFamily} placeholder="Outfit" onChange={(e) => up("fontFamily", e.target.value)} />
      <InputText
        type="number"
        label="overlay.field_font_size"
        min={WINDOW_LIMITS.fontSize[0]}
        max={WINDOW_LIMITS.fontSize[1]}
        value={s.fontSize}
        onChange={(e) => up("fontSize", e.target.value)}
      />
      <InputColor label="overlay.field_text_color" value={s.textColor} onChange={(v) => up("textColor", v)} />
      <InputColor label="overlay.field_background_color" value={s.backgroundColor} onChange={(v) => up("backgroundColor", v)} />
      <InputText
        type="number"
        label="overlay.field_max_lines"
        min={WINDOW_LIMITS.maxLines[0]}
        max={WINDOW_LIMITS.maxLines[1]}
        value={s.maxLines}
        onChange={(e) => up("maxLines", e.target.value)}
      />
      <InputText
        type="number"
        label="overlay.field_clear_after"
        min={0}
        max={MAX_CLEAR_AFTER_MS}
        step={500}
        value={s.clearAfterMs}
        onChange={(e) => up("clearAfterMs", e.target.value)}
      />
      <Hint>{t("overlay.field_clear_after_desc")}</Hint>
      <div className="flex gap-2">
        <button className="btn btn-sm flex-1" disabled={!status.open} onClick={() => window.ApiServer.overlay.resetWindowPosition()}>
          {t("overlay.btn_reset_position")}
        </button>
        <button className="btn btn-sm flex-1" disabled={!s.enable} onClick={() => window.ApiServer.overlay.showTestCaption()}>
          {t("overlay.btn_test")}
        </button>
      </div>
      <Hint>{t("overlay.window_vr_tip")}</Hint>
    </>
  );
};

/** One line saying what the SteamVR overlay is doing. */
function vrStatusText(t: TFunction, vr: VrOverlayStatusView): { tone: keyof typeof HINT_TONE; text: string } {
  const detail = vr.error ? ` (${vr.error})` : "";
  switch (vr.state) {
    case "running":
      return vr.reason === "overlayError"
        ? { tone: "warning", text: t("overlay.vr_running_with_error", { error: vr.error ?? "" }) }
        : { tone: "success", text: t("overlay.vr_running") };
    case "waiting":
      switch (vr.reason) {
        case "steamvrNotInstalled":
          return { tone: "muted", text: t("overlay.vr_waiting_not_installed") };
        case "steamvrQuit":
          return { tone: "muted", text: t("overlay.vr_waiting_quit") };
        case "connectFailed":
          return { tone: "warning", text: t("overlay.vr_waiting_failed") + detail };
        default:
          return { tone: "muted", text: t("overlay.vr_waiting_not_running") };
      }
    case "unavailable":
      return { tone: "warning", text: t("overlay.vr_unavailable") };
    case "error":
      return vr.reason === "steamvrTooOld"
        ? { tone: "error", text: t("overlay.vr_too_old") + detail }
        : { tone: "error", text: t("overlay.vr_error", { error: vr.error ?? "" }) };
    default:
      return { tone: "muted", text: t("overlay.vr_off") };
  }
}

const VrOverlaySettings: FC = () => {
  const { t } = useTranslation();
  const s = useSnapshot(window.ApiServer.state.services.vrc.data.overlay) as VrOverlaySettings;
  const vr = useSnapshot(window.ApiServer.overlay.state).vr as VrOverlayStatusView;
  const up = <K extends keyof VrOverlaySettings>(key: K, value: VrOverlaySettings[K]) => {
    window.ApiServer.patchService("vrc", (st) => {
      st.data.overlay[key] = value;
    });
  };
  const status = vrStatusText(t, vr);
  const attachOptions = VRC_OVERLAY_ATTACH.map((a) => ({ value: a, label: t(`overlay.vr_attach_${a}`) }));
  const nearHead = s.attach === "head" || s.attach === "world";

  return (
    <>
      <Inspector.SubHeader>{t("overlay.vr_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("overlay.vr_desc")}</Inspector.Description>
      <InputCheckbox label="overlay.vr_enable" value={s.enable} onChange={(v) => up("enable", v)} />
      <Hint tone={status.tone}>{status.text}</Hint>
      {!vr.runtimeInstalled && (
        <ModelRequirement
          ids={OPENVR_IDS}
          reason={t("overlay.vr_requires")}
          hideWhenReady
          onReady={() => window.ApiServer.overlay.refreshVrStatus()}
        />
      )}
      <SourceToggles value={s.sources as CaptionEventSource[]} onChange={(v) => up("sources", v)} />
      <InputCheckbox label="overlay.field_interim" value={s.interim} onChange={(v) => up("interim", v)} />
      <InputSelect label="overlay.vr_attach" options={attachOptions} value={s.attach} onValueChange={(v) => up("attach", v as VrOverlaySettings["attach"])} />
      <div className="flex gap-2">
        {s.attach === "world" && (
          <button className="btn btn-sm flex-1" disabled={vr.state !== "running"} onClick={() => window.ApiServer.overlay.recenterVr()}>
            {t("overlay.btn_recenter")}
          </button>
        )}
        <button className="btn btn-sm flex-1" disabled={!s.enable} onClick={() => window.ApiServer.overlay.showTestCaption()}>
          {t("overlay.btn_test")}
        </button>
      </div>
      {(s.attach === "leftHand" || s.attach === "rightHand") && <Hint>{t("overlay.vr_attach_hand_desc")}</Hint>}
      <InputText
        type="number"
        label="overlay.vr_width"
        min={VR_LIMITS.widthM[0]}
        max={VR_LIMITS.widthM[1]}
        step={0.05}
        value={s.widthM}
        onChange={(e) => up("widthM", e.target.value)}
      />
      {nearHead && (
        <>
          <InputText
            type="number"
            label="overlay.vr_distance"
            min={VR_LIMITS.distanceM[0]}
            max={VR_LIMITS.distanceM[1]}
            step={0.05}
            value={s.distanceM}
            onChange={(e) => up("distanceM", e.target.value)}
          />
          <InputText
            type="number"
            label="overlay.vr_offset_y"
            min={VR_LIMITS.offsetYM[0]}
            max={VR_LIMITS.offsetYM[1]}
            step={0.05}
            value={s.offsetYM}
            onChange={(e) => up("offsetYM", e.target.value)}
          />
        </>
      )}
      <InputText
        type="number"
        label="overlay.field_font_size"
        min={VR_LIMITS.fontSizePx[0]}
        max={VR_LIMITS.fontSizePx[1]}
        value={s.fontSizePx}
        onChange={(e) => up("fontSizePx", e.target.value)}
      />
      <InputRange
        label="overlay.vr_opacity"
        min={VR_LIMITS.opacity[0]}
        max={VR_LIMITS.opacity[1]}
        step={0.05}
        value={s.opacity}
        onChange={(e) => up("opacity", e.target.value)}
      />
      <InputText
        type="number"
        label="overlay.field_max_lines"
        min={VR_LIMITS.maxLines[0]}
        max={VR_LIMITS.maxLines[1]}
        value={s.maxLines}
        onChange={(e) => up("maxLines", e.target.value)}
      />
      <InputColor label="overlay.field_text_color" value={s.textColor} onChange={(v) => up("textColor", v)} />
      <InputColor label="overlay.field_background_color" value={s.backgroundColor} onChange={(v) => up("backgroundColor", v)} />
      <InputText
        type="number"
        label="overlay.field_clear_after"
        min={0}
        max={MAX_CLEAR_AFTER_MS}
        step={500}
        value={s.clearAfterMs}
        onChange={(e) => up("clearAfterMs", e.target.value)}
      />
      <Hint>{t("overlay.vr_fallback_tip")}</Hint>
    </>
  );
};

/**
 * Caption window and SteamVR overlay settings (P6-B), mounted at the end of the VRChat inspector (`inspector_vrc.tsx`).
 * Edits the root key `captionWindow` (`src/core/services/overlay/schema.ts`) and `services.vrc.data.overlay`
 * (`VRC_Overlay_Schema`); the live state comes from `window.ApiServer.overlay.state`.
 */
export const OverlaySection: FC = () => (
  <>
    <CaptionWindowSettings />
    <VrOverlaySettings />
  </>
);

export default OverlaySection;
