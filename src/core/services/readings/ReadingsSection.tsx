import { isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { STT_Backends } from "@/core/services/stt/schema";
import Inspector from "@/core/ui/inspector/components";
import { ModelRequirement } from "@/core/ui/inspector/components/ModelRequirement";
import { InputCheckbox, InputSelect } from "@/core/ui/inspector/components/input";
import { OthersEventSource, TextEventSource, type CaptionEventSource, type ReadingsEvent } from "@/types";
import { JAPANESE_DICTIONARY_ID, MUSIC_GATE_MODEL_ID } from "./annotator";
import { READINGS_PINYIN_STYLES, type ReadingsState } from "./schema";

const DICTIONARY_IDS = [JAPANESE_DICTIONARY_ID] as const;
const MUSIC_GATE_IDS = [MUSIC_GATE_MODEL_ID] as const;
/** Most sources at once (schema `.max(4)`). */
const MAX_SOURCES = 4;

/** Caption topics readings can annotate, with their labels. */
const SOURCE_OPTIONS: { value: CaptionEventSource; label: string }[] = [
  { value: TextEventSource.stt, label: "stt.title" },
  { value: TextEventSource.translation, label: "transl.title" },
  { value: OthersEventSource.stt, label: "common.text_source_others_stt" },
  { value: OthersEventSource.translation, label: "common.text_source_others_translation" },
];

const Hint: FC<{ tone?: "muted" | "warning" | "error"; children: ReactNode }> = ({ tone = "muted", children }) => (
  <div
    className={classNames("text-xs px-1 break-words", {
      "text-base-content/70": tone === "muted",
      "text-warning": tone === "warning",
      "text-error": tone === "error",
    })}
  >
    {children}
  </div>
);

const SourceToggles: FC<{ value: readonly CaptionEventSource[]; onChange: (value: CaptionEventSource[]) => void }> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const toggle = (source: CaptionEventSource, on: boolean) => {
    const next = on ? [...value, source] : value.filter((s) => s !== source);
    onChange(SOURCE_OPTIONS.map((o) => o.value).filter((s) => next.includes(s)).slice(0, MAX_SOURCES));
  };
  return (
    <div className="flex flex-col">
      <span className="text-xs font-medium text-base-content/80 pb-1">{t("readings.field_sources")}</span>
      {SOURCE_OPTIONS.map((o) => (
        <InputCheckbox key={o.value} label={o.label} value={value.includes(o.value)} className="ps-2" onChange={(on) => toggle(o.value, on)} />
      ))}
      {value.length === 0 && <Hint tone="warning">{t("readings.sources_none")}</Hint>}
    </div>
  );
};

/** The last published readings, as a check that it works. */
const LastReadings: FC<{ last: ReadingsEvent }> = ({ last }) => {
  const { t } = useTranslation();
  const lines = [last.hiragana, last.romaji, last.pinyin].filter((l): l is string => !!l);
  return (
    <div className="flex flex-col gap-0.5 rounded-box bg-base-200/60 p-2 text-xs" lang={last.lang}>
      <span className="font-medium text-base-content/70">{t("readings.last_title")}</span>
      <span className="break-words">{last.value}</span>
      {lines.map((l, i) => (
        <span key={i} className="break-words text-base-content/80">
          {l}
        </span>
      ))}
    </div>
  );
};

/**
 * The music / singing filter's model, shown while the filter is on in the `local` STT settings (the toggle is
 * `musicGate` in the local panel). Exported so the local panel can show it under its toggle.
 */
export const MusicGateRequirement: FC = () => {
  const { t } = useTranslation();
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  if (!isTauri() || stt.backend !== STT_Backends.local || !stt.local.musicGate) return null;
  return (
    <>
      <Inspector.SubHeader>{t("readings.music_gate_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("readings.music_gate_desc")}</Inspector.Description>
      <ModelRequirement ids={MUSIC_GATE_IDS} reason={t("readings.music_gate_reason")} />
    </>
  );
};

/**
 * Readings settings (P6-L), mounted at the end of the STT inspector (`src/core/ui/inspector/stt/index.tsx`): enable,
 * sources, hiragana / romaji / pinyin options and the Japanese dictionary's `<ModelRequirement>`. The music filter's
 * model card ({@link MusicGateRequirement}) sits under its toggle in the local STT panel (moved there by P6-I0).
 */
export const ReadingsSection: FC = () => {
  const { t } = useTranslation();
  const s = useSnapshot(window.ApiServer.state.readings) as ReadingsState;
  const status = useSnapshot(window.ApiServer.readings.state);
  const tauri = isTauri();
  const up = <K extends keyof ReadingsState>(key: K, value: ReadingsState[K]) => {
    window.ApiServer.state.readings[key] = value;
  };
  const japanese = s.hiragana || s.romaji;
  const styleOptions = READINGS_PINYIN_STYLES.map((v) => ({ value: v, label: t(`readings.pinyin_style_${v}`) }));

  return (
    <>
      <Inspector.SubHeader>{t("readings.title")}</Inspector.SubHeader>
      <Inspector.Description>{t("readings.description")}</Inspector.Description>
      <InputCheckbox label="readings.enable" value={s.enable} onChange={(v) => up("enable", v)} />
      {s.enable && (
        <>
          <SourceToggles value={s.sources as CaptionEventSource[]} onChange={(v) => up("sources", v)} />
          <span className="text-xs font-medium text-base-content/80 pt-1">{t("readings.japanese")}</span>
          <InputCheckbox label="readings.hiragana" className="ps-2" value={s.hiragana} onChange={(v) => up("hiragana", v)} />
          <InputCheckbox label="readings.romaji" className="ps-2" value={s.romaji} onChange={(v) => up("romaji", v)} />
          {tauri && japanese && <ModelRequirement ids={DICTIONARY_IDS} reason={t("readings.dictionary_reason")} />}
          <span className="text-xs font-medium text-base-content/80 pt-1">{t("readings.chinese")}</span>
          <InputCheckbox label="readings.pinyin" className="ps-2" value={s.pinyin} onChange={(v) => up("pinyin", v)} />
          {s.pinyin && (
            <InputSelect
              label="readings.pinyin_style"
              value={s.pinyinStyle}
              options={styleOptions}
              onValueChange={(v) => up("pinyinStyle", v as ReadingsState["pinyinStyle"])}
            />
          )}
          <InputCheckbox label="readings.interim" value={s.interim} onChange={(v) => up("interim", v)} />
          <Hint>{t("readings.interim_desc")}</Hint>
          <Hint>{t("readings.where_desc")}</Hint>
          {!tauri && <Hint tone="warning">{t("readings.desktop_only")}</Hint>}
          {status.lastError && <Hint tone="error">{t("readings.error", { error: status.lastError })}</Hint>}
          {status.last && <LastReadings last={status.last as ReadingsEvent} />}
        </>
      )}
    </>
  );
};

export default ReadingsSection;
