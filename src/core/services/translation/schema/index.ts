import { zSafe } from "@/utils";
import { z } from "zod";
import { Translation_Azure_Schema } from "./azure";
import { Translation_Backends } from "./backends";
import { Translation_Bergamot_Schema } from "./bergamot";
import { Translation_Deepl_Schema } from "./deepl";
import { Translation_Context_Schema, Translation_Glossary_Schema, Translation_Partials_Schema } from "./live";
import { Translation_Llm_Schema } from "./llm";
import { Translation_LibreTranslate_Schema } from "./libretranslate";
import { Translation_Nllb_Schema } from "./nllb";
import { Translation_OnDevice_Schema } from "./onDevice";
import { Translation_Router_Schema } from "./router";

// One slice per backend (P4-S0 split of the former translation/schema.ts). Owned by P4-C after the scaffold; the
// keys and defaults below are frozen for P4-D / P4-E / P4-F (add keys, never rename or retype them).
// Imports stay leaf-only (zod, utils, llm/types): src/core/schema.ts loads this module at startup.
export { keepObjects, SHARED_PAIR_BACKENDS, TRANSLATION_ENGINE_IDS, Translation_Backends, zInterimSwitch } from "./backends";
export type { InterimSwitch, TranslationEngineId } from "./backends";
export { Translation_Azure_Schema } from "./azure";
export { Translation_Bergamot_Schema } from "./bergamot";
export { Translation_Deepl_Schema } from "./deepl";
export {
  Translation_Context_Schema,
  Translation_Glossary_Schema,
  Translation_GlossaryEntry_Schema,
  Translation_Partials_Schema,
  TRANSLATION_GLOSSARY_MAX,
} from "./live";
export {
  Translation_Llm_Schema,
  TRANSLATION_LLM_DEFAULT_MODEL,
  TRANSLATION_LLM_SERVER_VARIANTS,
  TRANSLATION_PROMPT_FORMATS,
} from "./llm";
export type { TranslationPromptFormat } from "./llm";
export { Translation_LibreTranslate_Schema } from "./libretranslate";
export { NLLB_MODELS, Translation_Nllb_Schema } from "./nllb";
export { Translation_OnDevice_Schema } from "./onDevice";
export {
  cleanChain,
  Translation_PairOverride_Schema,
  Translation_Router_Schema,
  TRANSLATION_DEFAULT_CHAIN,
  TRANSLATION_INTERIM_ENGINES,
} from "./router";
export type { TranslationInterimEngine } from "./router";

const zodTranslation_Backends = z.nativeEnum(Translation_Backends);

export const Service_Translation_Schema = z.object({
  backend: zSafe(zodTranslation_Backends, Translation_Backends.azure),
  autoStart: zSafe(z.coerce.boolean(), false),
  inputField: zSafe(z.coerce.boolean(), true),
  /**
   * Language pair of the new backends (`SHARED_PAIR_BACKENDS`: llm, deepl, bergamot, nllb, auto), canonical BCP-47.
   * `sourceLanguage: ""` = the STT event's `lang`, else detected by the engine. The legacy backends (azure,
   * libretranslate, onDevice) keep their per-slice `languageFrom` / `language`.
   */
  sourceLanguage: zSafe(z.coerce.string(), ""),
  targetLanguage: zSafe(z.coerce.string(), "en"),
  azure: Translation_Azure_Schema,
  libretranslate: Translation_LibreTranslate_Schema,
  onDevice: Translation_OnDevice_Schema,
  llm: Translation_Llm_Schema,
  deepl: Translation_Deepl_Schema,
  bergamot: Translation_Bergamot_Schema,
  nllb: Translation_Nllb_Schema,
  auto: Translation_Router_Schema,
  /** Live settings (apply to the next line without a restart). */
  glossary: Translation_Glossary_Schema,
  context: Translation_Context_Schema,
  partials: Translation_Partials_Schema,
}).default({});

export type Translation_State = z.infer<typeof Service_Translation_Schema>;
export type Translation_GlossaryEntry = Translation_State["glossary"][number];
export type Translation_PairOverride = Translation_State["auto"]["pairOverrides"][number];
