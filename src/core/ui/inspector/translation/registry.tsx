import { Translation_Backends } from "@/core/services/translation/schema";
import { FC } from "react";
import { Azure } from "./Azure";
import { Bergamot } from "./Bergamot";
import { Deepl } from "./Deepl";
import { LibreTranslatePanel } from "./LibreTranslate";
import { Llm } from "./Llm";
import { backendLabelKey } from "./logic";
import { Nllb } from "./Nllb";
import { OnDevice } from "./OnDevice";
import { Router } from "./Router";

/**
 * Translation inspector registry (P4-S0; frozen names, owned by P4-F): the provider list order and one panel per
 * backend. The mapped type forces a panel for every `Translation_Backends` value. The shell
 * (`inspector_translation.tsx`) renders the selected panel inside `Inspector.Deactivatable` and
 * `TranslationLiveSettings` outside it.
 */
export interface TranslationPanelEntry {
  /** i18n key of the provider list label. */
  label: string;
  Panel: FC;
}

export const TRANSLATION_PANELS: { [k in Translation_Backends]: TranslationPanelEntry } = {
  [Translation_Backends.auto]: { label: backendLabelKey(Translation_Backends.auto), Panel: Router },
  [Translation_Backends.llm]: { label: backendLabelKey(Translation_Backends.llm), Panel: Llm },
  [Translation_Backends.deepl]: { label: backendLabelKey(Translation_Backends.deepl), Panel: Deepl },
  [Translation_Backends.bergamot]: { label: backendLabelKey(Translation_Backends.bergamot), Panel: Bergamot },
  [Translation_Backends.nllb]: { label: backendLabelKey(Translation_Backends.nllb), Panel: Nllb },
  [Translation_Backends.azure]: { label: backendLabelKey(Translation_Backends.azure), Panel: Azure },
  [Translation_Backends.libretranslate]: { label: backendLabelKey(Translation_Backends.libretranslate), Panel: LibreTranslatePanel },
  [Translation_Backends.onDevice]: { label: backendLabelKey(Translation_Backends.onDevice), Panel: OnDevice },
};

/** Provider list order. */
export const TRANSLATION_PROVIDER_ORDER: readonly Translation_Backends[] = [
  Translation_Backends.auto,
  Translation_Backends.llm,
  Translation_Backends.deepl,
  Translation_Backends.bergamot,
  Translation_Backends.nllb,
  Translation_Backends.azure,
  Translation_Backends.libretranslate,
  Translation_Backends.onDevice,
];
