import type { STT_State } from "@/core/services/stt/schema";
import {
  GEMINI_LIVE_API_VERSIONS,
  GEMINI_LIVE_ROTATE_MAX,
  GEMINI_LIVE_ROTATE_MIN,
} from "@/core/services/stt/schema/geminiLive";
import { parseVocabulary, rotateAfterMs } from "@/core/services/stt/services/geminiLive";
import { canonicalLanguage } from "@/core/services/translation/languages";
import { TRANSLATION_SERVICE_SKIPS_LIVE_TRANSLATE } from "@/core/services/translation/services/geminiLive";
import { ServiceNetworkState } from "@/types";
import { FC, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputRustAudioInput, InputSelect, InputText } from "../components/input";
import { Advanced, useLanguageName } from "./common";

/**
 * Settings panel of the `geminiLive` STT backend (P6-K): Gemini 3.5 Transcribe Live on the user's key, or the Live
 * Translate preview (captions and translations from one session). The key falls back to the Translation settings'
 * Gemini key; model ids, the API version and the rotation interval are editable (preview services change).
 */

/**
 * Spoken-language hints for Transcribe Live (BCP-47 with a region, the form Google's guide uses). Detection is the
 * default; a stored value outside the list stays selectable.
 */
const SPOKEN_LANGUAGES = [
  "en-US", "en-GB", "en-AU", "en-IN", "ja-JP", "ko-KR", "zh-CN", "zh-TW", "yue-HK", "es-ES", "es-US", "pt-BR",
  "pt-PT", "fr-FR", "fr-CA", "de-DE", "it-IT", "nl-NL", "ru-RU", "uk-UA", "pl-PL", "cs-CZ", "sk-SK", "hu-HU",
  "ro-RO", "bg-BG", "hr-HR", "sr-RS", "sl-SI", "el-GR", "tr-TR", "sv-SE", "da-DK", "nb-NO", "fi-FI", "et-EE",
  "lv-LV", "lt-LT", "ca-ES", "ar-SA", "ar-EG", "he-IL", "fa-IR", "ur-PK", "hi-IN", "bn-IN", "ta-IN", "te-IN",
  "mr-IN", "gu-IN", "kn-IN", "ml-IN", "pa-IN", "th-TH", "vi-VN", "id-ID", "ms-MY", "fil-PH", "sw-KE",
] as const;

/** Target languages offered for Live Translate (canonical codes; Gemini covers 70+ languages). */
const TARGET_LANGUAGES = [
  "en", "ja", "ko", "zh", "zh-Hant", "es", "pt", "pt-PT", "fr", "de", "it", "nl", "ru", "uk", "pl", "cs", "sk", "hu",
  "ro", "bg", "hr", "sr", "sl", "el", "tr", "sv", "da", "nb", "fi", "et", "lv", "lt", "ca", "ar", "he", "fa", "ur",
  "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "th", "vi", "id", "ms", "tl", "sw",
] as const;

export const GeminiLive: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.geminiLive);
  const llm = useSnapshot(window.ApiServer.state.services.translation.data.llm);
  const translation = useSnapshot(window.ApiServer.translation.serviceState);
  const languageName = useLanguageName();
  const up = <K extends keyof STT_State["geminiLive"]>(key: K, v: STT_State["geminiLive"][K]) =>
    window.ApiServer.patchService("stt", s => { s.data.geminiLive[key] = v; });

  const translate = pr.mode === "translate";
  const ownKey = pr.apiKey.trim() !== "";
  const sharedKey = llm.geminiKey.trim() !== "";

  const spokenOptions = useMemo(() => {
    const options = SPOKEN_LANGUAGES.map((code) => ({ value: code as string, label: `${languageName(code)} (${code})` }))
      .sort((a, b) => a.label.localeCompare(b.label));
    const stored = pr.language.trim();
    if (stored && stored !== "auto" && !options.some((o) => o.value === stored))
      options.unshift({ value: stored, label: `${languageName(stored)} (${stored})` });
    return [{ value: "auto", label: t("stt.gemini_language_auto") }, ...options];
  }, [languageName, pr.language, t]);

  const targetOptions = useMemo(() => {
    const options = TARGET_LANGUAGES.map((code) => ({ value: code as string, label: languageName(code) }))
      .sort((a, b) => a.label.localeCompare(b.label));
    const stored = canonicalLanguage(pr.translateTo) || pr.translateTo.trim();
    if (stored && !options.some((o) => o.value === stored)) options.unshift({ value: stored, label: languageName(stored) });
    return options;
  }, [languageName, pr.translateTo]);

  const targetValue = canonicalLanguage(pr.translateTo) || pr.translateTo;
  const vocabularyCount = useMemo(() => parseVocabulary(pr.vocabulary).length, [pr.vocabulary]);
  const rotateMinutes = rotateAfterMs(pr.rotateMinutes) / 60_000;
  const translatedTwice = !TRANSLATION_SERVICE_SKIPS_LIVE_TRANSLATE && translation.status !== ServiceNetworkState.disconnected;

  return <>
    <Inspector.SubHeader>{t("stt.gemini_title")}</Inspector.SubHeader>
    <Inspector.Description>{t("stt.gemini_desc")}</Inspector.Description>

    <InputText
      label="stt.gemini_key"
      type="password"
      autoComplete="off"
      value={pr.apiKey}
      placeholder={sharedKey ? t("stt.gemini_key_placeholder_shared") : t("stt.gemini_key_placeholder")}
      onChange={e => up("apiKey", e.target.value)}
    />
    {!ownKey && sharedKey && <Inspector.Description>{t("stt.gemini_key_shared")}</Inspector.Description>}
    {!ownKey && !sharedKey && <span role="status" className="text-xs text-warning">{t("stt.gemini_key_missing")}</span>}
    <span className="text-xs">
      <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href="https://aistudio.google.com/apikey">
        {t("stt.gemini_get_key")}
      </a>
    </span>

    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />

    <InputSelect
      label="stt.gemini_mode"
      value={pr.mode}
      options={[
        { value: "transcribe", label: t("stt.gemini_mode_transcribe") },
        { value: "translate", label: t("stt.gemini_mode_translate") },
      ]}
      onValueChange={e => up("mode", e === "translate" ? "translate" : "transcribe")}
    />
    <Inspector.Description>{translate ? t("stt.gemini_mode_translate_desc") : t("stt.gemini_mode_transcribe_desc")}</Inspector.Description>

    {!translate && <>
      <InputSelect label="common.field_language" value={pr.language.trim() || "auto"} options={spokenOptions} onValueChange={e => up("language", e)} />
      <Inspector.Description>{t("stt.gemini_language_desc")}</Inspector.Description>

      <InputSelect
        label="stt.gemini_formatting"
        value={pr.formatting}
        options={[
          { value: "verbatim", label: t("stt.gemini_formatting_verbatim") },
          { value: "smart", label: t("stt.gemini_formatting_smart") },
        ]}
        onValueChange={e => up("formatting", e === "smart" ? "smart" : "verbatim")}
      />
      <Inspector.Description>{t("stt.gemini_formatting_desc")}</Inspector.Description>

      <InputText
        label="stt.gemini_vocabulary"
        value={pr.vocabulary}
        placeholder={t("stt.gemini_vocabulary_placeholder")}
        onChange={e => up("vocabulary", e.target.value)}
      />
      <Inspector.Description>
        {t("stt.gemini_vocabulary_desc")}
        {vocabularyCount > 0 && <> {t("stt.gemini_vocabulary_count", { count: vocabularyCount })}</>}
      </Inspector.Description>
    </>}

    {translate && <>
      <InputSelect label="stt.gemini_translate_to" value={targetValue} options={targetOptions} onValueChange={e => up("translateTo", e)} />
      <InputCheckbox label="stt.gemini_echo" value={pr.echoTargetLanguage} onChange={e => up("echoTargetLanguage", e)} />
      <Inspector.Description>{t("stt.gemini_echo_desc")}</Inspector.Description>
      {translatedTwice && (
        <span role="status" className="text-xs text-warning">{t("stt.gemini_status_translation_running")}</span>
      )}
    </>}

    <InputCheckbox label="stt.field_enable_interim_results" value={pr.interim} onChange={e => up("interim", e)} />

    <Inspector.Description>{t("stt.gemini_privacy")}</Inspector.Description>

    <Advanced title={t("stt.gemini_advanced")}>
      {translate
        ? <InputText label="stt.gemini_translate_model" value={pr.translateModel} placeholder="gemini-3.5-live-translate-preview" onChange={e => up("translateModel", e.target.value)} />
        : <InputText label="stt.gemini_model" value={pr.model} placeholder="gemini-3.5-transcribe-live" onChange={e => up("model", e.target.value)} />}
      <Inspector.Description>{t("stt.gemini_model_desc")}</Inspector.Description>

      <InputText
        label="stt.gemini_rotate"
        type="number"
        min={GEMINI_LIVE_ROTATE_MIN}
        max={GEMINI_LIVE_ROTATE_MAX}
        step={0.25}
        value={pr.rotateMinutes}
        onChange={e => up("rotateMinutes", e.target.value)}
      />
      <Inspector.Description>{t("stt.gemini_rotate_desc", { minutes: rotateMinutes })}</Inspector.Description>

      <InputSelect
        label="stt.gemini_api_version"
        value={pr.apiVersion}
        options={GEMINI_LIVE_API_VERSIONS.map((v) => ({ value: v, label: v }))}
        onValueChange={e => up("apiVersion", e === "v1alpha" ? "v1alpha" : "v1beta")}
      />
      <Inspector.Description>{t("stt.gemini_api_version_desc")}</Inspector.Description>
    </Advanced>
  </>;
};
