import { llmServerUi } from "@/core/services/llm_server";
import { bundledTranslatorOnCpu, interimHeldOffByCpu, interimTranslationsOn } from "@/core/services/translation/interimPolicy";
import type { Translation_Backends, Translation_State } from "@/core/services/translation/schema";
import classNames from "classnames";
import { ComponentProps, FC, ReactNode, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { InputSelect, InputText, type InputSelectOption } from "../components/input";
import { formatScaled, languageOptions, parseBoundedNumber, pickerValue } from "./logic";

// Shared by the translation inspector panels (P4-S0 split of inspector_translation.tsx). Owned by P4-F.

/** Debounce for probing a server while its URL / key is being typed. */
export const LANGUAGE_PROBE_DEBOUNCE_MS = 400;

/** Keeps a stored value selectable when the option list does not (yet) contain it. */
export function mergeOption(opts: InputSelectOption[], current: string): InputSelectOption[] {
  if (!current || opts.some((o) => o.value === current)) return opts;
  return [{ label: current, value: current }, ...opts];
}

/** Persisted translation settings (valtio proxy): panels write slice fields directly, like the other inspectors. */
export function translationData(): Translation_State {
  return window.ApiServer.state.services.translation.data;
}

/**
 * What the "Interim results" switch of the `llm` and `auto` panels shows (`interim: "auto"` is on, except while the
 * bundled translator runs on the processor): `on` for the checkbox, `heldOff` for the hint beside it.
 */
export function useInterimSwitch(backend: Translation_Backends.llm | Translation_Backends.auto): { on: boolean; heldOff: boolean } {
  const data = useSnapshot(window.ApiServer.state.services.translation.data) as Translation_State;
  const status = useSnapshot(llmServerUi).status;
  const onCpu = bundledTranslatorOnCpu(data.llm, status);
  return { on: interimTranslationsOn(data, backend, onCpu), heldOff: interimHeldOffByCpu(data, backend, onCpu) };
}

/** Setter for one field of one backend slice. */
export function sliceSetter<S extends keyof Translation_State>(slice: S) {
  return <K extends keyof Translation_State[S]>(key: K, value: Translation_State[S][K]) => {
    (translationData()[slice] as Translation_State[S])[key] = value;
  };
}

const HINT_TONE = {
  muted: "text-base-content/70",
  warning: "text-warning",
  error: "text-error",
  success: "text-success",
} as const;

/** One line of help or status under a field. */
export const Hint: FC<{ tone?: keyof typeof HINT_TONE; className?: string; children: ReactNode }> = ({ tone = "muted", className, children }) => (
  <div className={classNames("text-xs px-1", HINT_TONE[tone], className)}>{children}</div>
);

/** Content that stays usable while translation runs, inside a panel that is otherwise locked (usage, downloads). */
export const LiveArea: FC<{ className?: string; children: ReactNode }> = ({ className, children }) => (
  <div className={classNames("pointer-events-auto flex flex-col gap-2", className)}>{children}</div>
);

/** A collapsible group for settings most people never change. */
export const Details: FC<{
  title: string;
  defaultOpen?: boolean;
  className?: string;
  onToggle?: (open: boolean) => void;
  children: ReactNode;
}> = ({ title, defaultOpen, className, onToggle, children }) => (
  <details
    open={defaultOpen}
    onToggle={onToggle ? (e) => onToggle(e.currentTarget.open) : undefined}
    className={classNames(
      "rounded-lg border border-base-content/10 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden",
      className,
    )}
  >
    <summary className="px-3 py-2 text-sm font-medium">{title}</summary>
    <div className="px-3 pb-3 pt-2 flex flex-col gap-2 border-t border-base-content/10">{children}</div>
  </details>
);

/**
 * A number field that commits only valid values in [min, max] (stored units) and shows `value / scale` (e.g. seconds
 * for a millisecond setting). The text can be empty or partly typed meanwhile; leaving the field shows the stored value.
 */
export const NumberField: FC<{
  label: string;
  value: number;
  min: number;
  max: number;
  /** Stored units per shown unit (1000 = the field shows seconds of a millisecond value). */
  scale?: number;
  /** The stored value is an integer (default). */
  integer?: boolean;
  step?: number;
  disabled?: boolean;
  onCommit: (value: number) => void;
}> = ({ label, value, min, max, scale = 1, integer = true, step, disabled, onCommit }) => {
  const { t } = useTranslation();
  const [text, setText] = useState(() => formatScaled(value, scale));
  useEffect(() => setText(formatScaled(value, scale)), [value, scale]);
  const invalid = text.trim() !== "" && parseBoundedNumber(text, { min, max, scale, integer }) === null;
  return (
    <>
      <InputText
        label={label}
        aria-label={t(label)}
        aria-invalid={invalid || undefined}
        type="number"
        inputMode={integer && scale === 1 ? "numeric" : "decimal"}
        min={min / scale}
        max={max / scale}
        step={step ?? (integer && scale === 1 ? 1 : "any")}
        value={text}
        disabled={disabled}
        onChange={(e) => {
          setText(e.target.value);
          const v = parseBoundedNumber(e.target.value, { min, max, scale, integer });
          if (v !== null && v !== value) onCommit(v);
        }}
        onBlur={() => setText(formatScaled(value, scale))}
      />
      {invalid && (
        <Hint tone="warning">{t("transl.number_range", { min: formatScaled(min, scale), max: formatScaled(max, scale) })}</Hint>
      )}
    </>
  );
};

/**
 * A language picker over the translation language table (names in the UI language). `emptyLabel` adds a first
 * option with the value "" (automatic source, any language). A stored code that is not in the table stays selectable.
 */
export const LanguageSelect: FC<{
  label: string;
  value: string;
  onChange: (code: string) => void;
  emptyLabel?: string;
}> = ({ label, value, onChange, emptyLabel }) => {
  const { i18n } = useTranslation();
  const shown = pickerValue(value);
  const options: InputSelectOption[] = [
    ...(emptyLabel !== undefined ? [{ label: emptyLabel, value: "" }] : []),
    ...mergeOption([...languageOptions(i18n.language || "en")], shown),
  ];
  return <InputSelect label={label} options={options} value={shown} onValueChange={onChange} />;
};

/**
 * `InputText` whose accessible name is its label (the base input otherwise names itself after its placeholder or
 * "Input", which hides the visible label from screen readers).
 */
export const TextField: FC<ComponentProps<typeof InputText>> = (props) => {
  const { t } = useTranslation();
  return <InputText aria-label={t(props.label)} {...props} />;
};

/** Options 1…n as a select (thread counts, beam widths). */
export function countOptions(max: number, label: (n: number) => string, min = 1): InputSelectOption[] {
  const out: InputSelectOption[] = [];
  for (let n = min; n <= max; n++) out.push({ value: String(n), label: label(n) });
  return out;
}
