import { Translation_State } from "@/core/services/translation/schema";
import { Translation_LibreTranslateService } from "@/core/services/translation/services/libretranslate";
import { FC, memo, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputSelect, InputSelectOption } from "../components/input";
import { Hint, LANGUAGE_PROBE_DEBOUNCE_MS, mergeOption, TextField } from "./common";
import { LIBRE_PRESETS, libreServerKind, type LibreServerKind } from "./logic";

type LibreSlice = Translation_State["libretranslate"];

/** Server answer of the language probe: null = not asked yet / no answer. */
type Probe = { kind: "idle" } | { kind: "ok"; options: InputSelectOption[] } | { kind: "failed" };

/**
 * LibreTranslate panel: any server with the LibreTranslate API (LibreTranslate itself, or LTEngine, LibreTranslate's
 * local LLM server, through the preset), its language pair from the server's list (source "auto" lets the server
 * detect it), auto-detection and interims. The server is probed while the URL is typed.
 *
 * Owned by P4-F (moved from inspector_translation.tsx by P4-S0; adds the server presets).
 */
export const LibreTranslatePanel: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.libretranslate);
  const [probe, setProbe] = useState<Probe>({ kind: "idle" });
  const up = <K extends keyof LibreSlice>(key: K, v: LibreSlice[K]) => {
    window.ApiServer.state.services.translation.data.libretranslate[key] = v;
  };

  useEffect(() => {
    let cancel = false;
    // Debounced: the effect re-runs on every keystroke in the URL field. The endpoint is passed per
    // call, so probing here never changes what a running translation talks to.
    const timer = setTimeout(() => {
      if (!pr.endpoint.trim()) {
        setProbe({ kind: "idle" });
        return;
      }
      void Translation_LibreTranslateService.getLanguages(pr.endpoint, pr.apiKey ?? "").then((list) => {
        if (cancel) return;
        if (!Array.isArray(list) || list.length === 0) {
          setProbe({ kind: "failed" });
          return;
        }
        setProbe({
          kind: "ok",
          options: list.map((l) => ({ label: `${l.name} (${l.code})`, value: String(l.code) })),
        });
      });
    }, LANGUAGE_PROBE_DEBOUNCE_MS);
    return () => {
      cancel = true;
      clearTimeout(timer);
    };
  }, [pr.endpoint, pr.apiKey]);

  // "Other address" keeps the URL (typed next) and is shown until the URL changes; otherwise the choice follows the URL.
  const [otherFor, setOtherFor] = useState<string | null>(null);
  const kind: LibreServerKind = otherFor !== null && otherFor === pr.endpoint ? "custom" : libreServerKind(pr.endpoint);
  const applyPreset = (value: string) => {
    if (value === "custom") {
      setOtherFor(pr.endpoint);
      return;
    }
    setOtherFor(null);
    const preset = LIBRE_PRESETS[value as Exclude<LibreServerKind, "custom">];
    if (!preset) return;
    for (const [key, v] of Object.entries(preset) as [keyof LibreSlice, LibreSlice[keyof LibreSlice]][]) {
      (window.ApiServer.state.services.translation.data.libretranslate as Record<keyof LibreSlice, unknown>)[key] = v;
    }
  };

  // Some servers (LTEngine) list "auto" as a language; it is offered once, as the source choice.
  const langOpts = probe.kind === "ok" ? probe.options.filter((o) => o.value.toLowerCase() !== "auto") : [];
  const fromOpts = langOpts.length
    ? [
        { label: t("transl.azure_from_auto"), value: "auto" },
        ...(pr.languageFrom.trim().toLowerCase() === "auto" ? langOpts : mergeOption(langOpts, pr.languageFrom)),
      ]
    : [];
  const toOpts = mergeOption(langOpts, pr.language);

  return <>
    <Inspector.SubHeader>{t("transl.libretranslate_title")}</Inspector.SubHeader>
    <Hint>{t("transl.libretranslate_desc")}</Hint>
    <InputSelect
      label="transl.libre_field_server"
      options={[
        { value: "libretranslate", label: t("transl.libre_server_libretranslate") },
        { value: "ltengine", label: t("transl.libre_server_ltengine") },
        { value: "custom", label: t("transl.libre_server_custom") },
      ]}
      value={kind}
      onValueChange={applyPreset}
    />
    {kind === "ltengine" && <Hint>{t("transl.libre_ltengine_hint")}</Hint>}
    <TextField
      label="transl.libretranslate_endpoint"
      value={pr.endpoint}
      placeholder="http://localhost:5000"
      onChange={(e) => up("endpoint", e.target.value)}
    />
    {probe.kind === "failed" && <Hint tone="warning">{t("transl.libre_probe_failed")}</Hint>}
    {probe.kind === "ok" && <Hint tone="success">{t("transl.libre_probe_ok", { count: langOpts.length })}</Hint>}
    <TextField
      label="transl.libretranslate_api_key"
      type="password"
      autoComplete="new-password"
      value={pr.apiKey}
      onChange={(e) => up("apiKey", e.target.value)}
    />
    {fromOpts.length > 0 ? (
      <InputSelect
        label="transl.libretranslate_from_language"
        options={fromOpts}
        value={pr.languageFrom.trim().toLowerCase() === "auto" ? "auto" : pr.languageFrom}
        onValueChange={(e) => up("languageFrom", e as string)}
      />
    ) : (
      <TextField
        label="transl.libretranslate_from_language"
        value={pr.languageFrom}
        placeholder="auto"
        onChange={(e) => up("languageFrom", e.target.value)}
      />
    )}
    {toOpts.length > 0 ? (
      <InputSelect
        label="transl.libretranslate_to_language"
        options={toOpts}
        value={pr.language}
        onValueChange={(e) => up("language", e as string)}
      />
    ) : (
      <TextField
        label="transl.libretranslate_to_language"
        value={pr.language}
        onChange={(e) => up("language", e.target.value)}
      />
    )}
    <InputCheckbox
      label="transl.libretranslate_auto_detect"
      value={pr.autoDetect}
      onChange={(v) => up("autoDetect", v)}
    />
    <Hint>{pr.languageFrom.trim() === "auto" ? t("transl.libre_auto_source_hint") : t("transl.libre_auto_detect_hint")}</Hint>
    <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
  </>;
});
