/**
 * Canonical translation languages: lower-case BCP-47 base codes (`en`, `ja`, `ko`, `zh` = Simplified Chinese),
 * with script or region subtags only where they change the language (`zh-Hant`, `pt-PT`). ISO 639-1 codes are used
 * where they exist, ISO 639-3 otherwise (`yue`, `ceb`, the Arabic varieties). Every engine request carries canonical
 * codes; engines map them (`toFlores` for NLLB, `toAzure`, `toIso6391` for Bergamot / LibreTranslate; DeepL codes in
 * `services/deepl.ts`). Must agree with Rust `mt_local/flores.rs`: `__tests__/languages.fixture.json` is the shared
 * fixture (`languages.test.ts` keeps it equal to this table).
 *
 * The table is the union of NLLB-200's languages (FLORES-200 codes, the long tail), Azure Translator's and the LLM
 * presets' main languages. Owned by P4-C.
 */

export interface TranslationLanguage {
  /** Canonical code. */
  code: string;
  /** English name (the UI shows `Intl.DisplayNames` in the UI language; prompts use this name). */
  name: string;
  /** FLORES-200 code (NLLB); null = NLLB lacks it. */
  flores: string | null;
  /** Azure Translator code; null = unsupported. */
  azure: string | null;
  /** ISO 639-1 (Bergamot, LibreTranslate); null = none. */
  iso6391: string | null;
}

type Row = readonly [code: string, name: string, flores: string | null, azure: string | null, iso6391: string | null];

// Sorted by code. Arabic varieties and other ISO 639-3-only languages have no ISO 639-1 code on purpose: engines
// keyed by ISO 639-1 (Bergamot, LibreTranslate) would translate them as Modern Standard Arabic etc.
const ROWS: readonly Row[] = [
  ["ace", "Acehnese", "ace_Latn", null, null],
  ["acm", "Mesopotamian Arabic", "acm_Arab", null, null],
  ["acq", "Ta'izzi-Adeni Arabic", "acq_Arab", null, null],
  ["aeb", "Tunisian Arabic", "aeb_Arab", null, null],
  ["af", "Afrikaans", "afr_Latn", "af", "af"],
  ["ajp", "South Levantine Arabic", "ajp_Arab", null, null],
  ["ak", "Akan", "aka_Latn", null, "ak"],
  ["am", "Amharic", "amh_Ethi", "am", "am"],
  ["apc", "North Levantine Arabic", "apc_Arab", null, null],
  ["ar", "Arabic", "arb_Arab", "ar", "ar"],
  ["ars", "Najdi Arabic", "ars_Arab", null, null],
  ["ary", "Moroccan Arabic", "ary_Arab", null, null],
  ["arz", "Egyptian Arabic", "arz_Arab", null, null],
  ["as", "Assamese", "asm_Beng", "as", "as"],
  ["ast", "Asturian", "ast_Latn", null, null],
  ["awa", "Awadhi", "awa_Deva", null, null],
  ["ay", "Aymara", "ayr_Latn", null, "ay"],
  ["az", "Azerbaijani", "azj_Latn", "az", "az"],
  ["azb", "South Azerbaijani", "azb_Arab", null, null],
  ["ba", "Bashkir", "bak_Cyrl", "ba", "ba"],
  ["ban", "Balinese", "ban_Latn", null, null],
  ["be", "Belarusian", "bel_Cyrl", null, "be"],
  ["bem", "Bemba", "bem_Latn", null, null],
  ["bg", "Bulgarian", "bul_Cyrl", "bg", "bg"],
  ["bho", "Bhojpuri", "bho_Deva", "bho", null],
  ["bjn", "Banjar", "bjn_Latn", null, null],
  ["bm", "Bambara", "bam_Latn", null, "bm"],
  ["bn", "Bengali", "ben_Beng", "bn", "bn"],
  ["bo", "Tibetan", "bod_Tibt", "bo", "bo"],
  ["bs", "Bosnian", "bos_Latn", "bs", "bs"],
  ["bug", "Buginese", "bug_Latn", null, null],
  ["ca", "Catalan", "cat_Latn", "ca", "ca"],
  ["ceb", "Cebuano", "ceb_Latn", null, null],
  ["cjk", "Chokwe", "cjk_Latn", null, null],
  ["ckb", "Central Kurdish", "ckb_Arab", "ku", null],
  ["crh", "Crimean Tatar", "crh_Latn", null, null],
  ["cs", "Czech", "ces_Latn", "cs", "cs"],
  ["cy", "Welsh", "cym_Latn", "cy", "cy"],
  ["da", "Danish", "dan_Latn", "da", "da"],
  ["de", "German", "deu_Latn", "de", "de"],
  ["dik", "Southwestern Dinka", "dik_Latn", null, null],
  ["dyu", "Dyula", "dyu_Latn", null, null],
  ["dz", "Dzongkha", "dzo_Tibt", null, "dz"],
  ["ee", "Ewe", "ewe_Latn", null, "ee"],
  ["el", "Greek", "ell_Grek", "el", "el"],
  ["en", "English", "eng_Latn", "en", "en"],
  ["eo", "Esperanto", "epo_Latn", null, "eo"],
  ["es", "Spanish", "spa_Latn", "es", "es"],
  ["et", "Estonian", "est_Latn", "et", "et"],
  ["eu", "Basque", "eus_Latn", "eu", "eu"],
  ["fa", "Persian", "pes_Arab", "fa", "fa"],
  ["ff", "Fula", "fuv_Latn", null, "ff"],
  ["fi", "Finnish", "fin_Latn", "fi", "fi"],
  ["fj", "Fijian", "fij_Latn", "fj", "fj"],
  ["fo", "Faroese", "fao_Latn", "fo", "fo"],
  ["fon", "Fon", "fon_Latn", null, null],
  ["fr", "French", "fra_Latn", "fr", "fr"],
  ["fur", "Friulian", "fur_Latn", null, null],
  ["ga", "Irish", "gle_Latn", "ga", "ga"],
  ["gd", "Scottish Gaelic", "gla_Latn", null, "gd"],
  ["gl", "Galician", "glg_Latn", "gl", "gl"],
  ["gn", "Guarani", "grn_Latn", null, "gn"],
  ["gu", "Gujarati", "guj_Gujr", "gu", "gu"],
  ["ha", "Hausa", "hau_Latn", "ha", "ha"],
  ["he", "Hebrew", "heb_Hebr", "he", "he"],
  ["hi", "Hindi", "hin_Deva", "hi", "hi"],
  ["hne", "Chhattisgarhi", "hne_Deva", "hne", null],
  ["hr", "Croatian", "hrv_Latn", "hr", "hr"],
  ["ht", "Haitian Creole", "hat_Latn", "ht", "ht"],
  ["hu", "Hungarian", "hun_Latn", "hu", "hu"],
  ["hy", "Armenian", "hye_Armn", "hy", "hy"],
  ["id", "Indonesian", "ind_Latn", "id", "id"],
  ["ig", "Igbo", "ibo_Latn", "ig", "ig"],
  ["ilo", "Ilocano", "ilo_Latn", null, null],
  ["is", "Icelandic", "isl_Latn", "is", "is"],
  ["it", "Italian", "ita_Latn", "it", "it"],
  ["ja", "Japanese", "jpn_Jpan", "ja", "ja"],
  ["jv", "Javanese", "jav_Latn", null, "jv"],
  ["ka", "Georgian", "kat_Geor", "ka", "ka"],
  ["kab", "Kabyle", "kab_Latn", null, null],
  ["kac", "Jingpho", "kac_Latn", null, null],
  ["kam", "Kamba", "kam_Latn", null, null],
  ["kbp", "Kabiyè", "kbp_Latn", null, null],
  ["kea", "Kabuverdianu", "kea_Latn", null, null],
  ["kg", "Kikongo", "kon_Latn", null, "kg"],
  ["ki", "Kikuyu", "kik_Latn", null, "ki"],
  ["kk", "Kazakh", "kaz_Cyrl", "kk", "kk"],
  ["km", "Khmer", "khm_Khmr", "km", "km"],
  ["kmb", "Kimbundu", "kmb_Latn", null, null],
  ["kmr", "Northern Kurdish", "kmr_Latn", "kmr", null],
  ["kn", "Kannada", "kan_Knda", "kn", "kn"],
  ["ko", "Korean", "kor_Hang", "ko", "ko"],
  ["kr", "Kanuri", "knc_Latn", null, "kr"],
  ["ks", "Kashmiri", "kas_Arab", "ks", "ks"],
  ["ky", "Kyrgyz", "kir_Cyrl", "ky", "ky"],
  ["lb", "Luxembourgish", "ltz_Latn", null, "lb"],
  ["lg", "Ganda", "lug_Latn", "lug", "lg"],
  ["li", "Limburgish", "lim_Latn", null, "li"],
  ["lij", "Ligurian", "lij_Latn", null, null],
  ["lmo", "Lombard", "lmo_Latn", null, null],
  ["ln", "Lingala", "lin_Latn", "ln", "ln"],
  ["lo", "Lao", "lao_Laoo", "lo", "lo"],
  ["lt", "Lithuanian", "lit_Latn", "lt", "lt"],
  ["ltg", "Latgalian", "ltg_Latn", null, null],
  ["lua", "Luba-Kasai", "lua_Latn", null, null],
  ["luo", "Luo", "luo_Latn", null, null],
  ["lus", "Mizo", "lus_Latn", null, null],
  ["lv", "Latvian", "lvs_Latn", "lv", "lv"],
  ["mag", "Magahi", "mag_Deva", null, null],
  ["mai", "Maithili", "mai_Deva", "mai", null],
  ["mg", "Malagasy", "plt_Latn", "mg", "mg"],
  ["mi", "Maori", "mri_Latn", "mi", "mi"],
  ["min", "Minangkabau", "min_Latn", null, null],
  ["mk", "Macedonian", "mkd_Cyrl", "mk", "mk"],
  ["ml", "Malayalam", "mal_Mlym", "ml", "ml"],
  ["mn", "Mongolian", "khk_Cyrl", "mn-Cyrl", "mn"],
  ["mni", "Meitei (Manipuri)", "mni_Beng", "mni", null],
  ["mos", "Mossi", "mos_Latn", null, null],
  ["mr", "Marathi", "mar_Deva", "mr", "mr"],
  ["ms", "Malay", "zsm_Latn", "ms", "ms"],
  ["mt", "Maltese", "mlt_Latn", "mt", "mt"],
  ["my", "Burmese", "mya_Mymr", "my", "my"],
  ["nb", "Norwegian Bokmål", "nob_Latn", "nb", "nb"],
  ["ne", "Nepali", "npi_Deva", "ne", "ne"],
  ["nl", "Dutch", "nld_Latn", "nl", "nl"],
  ["nn", "Norwegian Nynorsk", "nno_Latn", null, "nn"],
  ["nso", "Northern Sotho", "nso_Latn", "nso", null],
  ["nus", "Nuer", "nus_Latn", null, null],
  ["ny", "Chichewa", "nya_Latn", "nya", "ny"],
  ["oc", "Occitan", "oci_Latn", null, "oc"],
  ["om", "Oromo", "gaz_Latn", null, "om"],
  ["or", "Odia", "ory_Orya", "or", "or"],
  ["pa", "Punjabi", "pan_Guru", "pa", "pa"],
  ["pag", "Pangasinan", "pag_Latn", null, null],
  ["pap", "Papiamento", "pap_Latn", null, null],
  ["pl", "Polish", "pol_Latn", "pl", "pl"],
  ["prs", "Dari", "prs_Arab", "prs", null],
  ["ps", "Pashto", "pbt_Arab", "ps", "ps"],
  ["pt", "Portuguese", "por_Latn", "pt", "pt"],
  ["pt-PT", "Portuguese (Portugal)", "por_Latn", "pt-PT", "pt"],
  ["qu", "Quechua", "quy_Latn", null, "qu"],
  ["rn", "Rundi", "run_Latn", "run", "rn"],
  ["ro", "Romanian", "ron_Latn", "ro", "ro"],
  ["ru", "Russian", "rus_Cyrl", "ru", "ru"],
  ["rw", "Kinyarwanda", "kin_Latn", "rw", "rw"],
  ["sa", "Sanskrit", "san_Deva", null, "sa"],
  ["sat", "Santali", "sat_Beng", null, null],
  ["sc", "Sardinian", "srd_Latn", null, "sc"],
  ["scn", "Sicilian", "scn_Latn", null, null],
  ["sd", "Sindhi", "snd_Arab", "sd", "sd"],
  ["sg", "Sango", "sag_Latn", null, "sg"],
  ["shn", "Shan", "shn_Mymr", null, null],
  ["si", "Sinhala", "sin_Sinh", "si", "si"],
  ["sk", "Slovak", "slk_Latn", "sk", "sk"],
  ["sl", "Slovenian", "slv_Latn", "sl", "sl"],
  ["sm", "Samoan", "smo_Latn", "sm", "sm"],
  ["sn", "Shona", "sna_Latn", "sn", "sn"],
  ["so", "Somali", "som_Latn", "so", "so"],
  ["sq", "Albanian", "als_Latn", "sq", "sq"],
  ["sr", "Serbian", "srp_Cyrl", "sr-Cyrl", "sr"],
  ["ss", "Swati", "ssw_Latn", null, "ss"],
  ["st", "Southern Sotho", "sot_Latn", "st", "st"],
  ["su", "Sundanese", "sun_Latn", null, "su"],
  ["sv", "Swedish", "swe_Latn", "sv", "sv"],
  ["sw", "Swahili", "swh_Latn", "sw", "sw"],
  ["szl", "Silesian", "szl_Latn", null, null],
  ["ta", "Tamil", "tam_Taml", "ta", "ta"],
  ["taq", "Tamasheq", "taq_Latn", null, null],
  ["te", "Telugu", "tel_Telu", "te", "te"],
  ["tg", "Tajik", "tgk_Cyrl", null, "tg"],
  ["th", "Thai", "tha_Thai", "th", "th"],
  ["ti", "Tigrinya", "tir_Ethi", "ti", "ti"],
  ["tk", "Turkmen", "tuk_Latn", "tk", "tk"],
  ["tl", "Tagalog", "tgl_Latn", "fil", "tl"],
  ["tn", "Tswana", "tsn_Latn", "tn", "tn"],
  ["tpi", "Tok Pisin", "tpi_Latn", null, null],
  ["tr", "Turkish", "tur_Latn", "tr", "tr"],
  ["ts", "Tsonga", "tso_Latn", null, "ts"],
  ["tt", "Tatar", "tat_Cyrl", "tt", "tt"],
  ["tum", "Tumbuka", "tum_Latn", null, null],
  ["tw", "Twi", "twi_Latn", null, "tw"],
  ["tzm", "Central Atlas Tamazight", "tzm_Tfng", null, null],
  ["ug", "Uyghur", "uig_Arab", "ug", "ug"],
  ["uk", "Ukrainian", "ukr_Cyrl", "uk", "uk"],
  ["umb", "Umbundu", "umb_Latn", null, null],
  ["ur", "Urdu", "urd_Arab", "ur", "ur"],
  ["uz", "Uzbek", "uzn_Latn", "uz", "uz"],
  ["vec", "Venetian", "vec_Latn", null, null],
  ["vi", "Vietnamese", "vie_Latn", "vi", "vi"],
  ["war", "Waray", "war_Latn", null, null],
  ["wo", "Wolof", "wol_Latn", null, "wo"],
  ["xh", "Xhosa", "xho_Latn", "xh", "xh"],
  ["yi", "Yiddish", "ydd_Hebr", null, "yi"],
  ["yo", "Yoruba", "yor_Latn", "yo", "yo"],
  ["yue", "Cantonese", "yue_Hant", "yue", null],
  ["zh", "Chinese (Simplified)", "zho_Hans", "zh-Hans", "zh"],
  ["zh-Hant", "Chinese (Traditional)", "zho_Hant", "zh-Hant", "zh"],
  ["zu", "Zulu", "zul_Latn", "zu", "zu"],
];

export const TRANSLATION_LANGUAGES: readonly TranslationLanguage[] = ROWS.map(([code, name, flores, azure, iso6391]) => ({
  code,
  name,
  flores,
  azure,
  iso6391,
}));

const BY_CODE = new Map(TRANSLATION_LANGUAGES.map((l) => [l.code.toLowerCase(), l]));

/**
 * Aliases that are neither a canonical code nor one of the table's FLORES / ISO 639-3 / Azure codes: legacy and
 * macro-language codes, Whisper's own codes (`jw`, `haw`-style), and Chinese script / region forms.
 */
const ALIASES: Record<string, string> = {
  "zh-hans": "zh",
  "zh-cn": "zh",
  "zh-sg": "zh",
  "zh-my": "zh",
  "zh-hant": "zh-Hant",
  "zh-tw": "zh-Hant",
  "zh-hk": "zh-Hant",
  "zh-mo": "zh-Hant",
  cmn: "zh",
  "zh-yue": "yue",
  "pt-br": "pt",
  "pt-pt": "pt-PT",
  no: "nb",
  nor: "nb",
  fil: "tl",
  iw: "he",
  in: "id",
  ji: "yi",
  jw: "jv",
  mo: "ro",
  "sr-cyrl": "sr",
  "mn-cyrl": "mn",
};

/** FLORES-200 codes (`jpn-jpan` after `_` → `-`), their ISO 639-3 part (`jpn`) and Azure codes, to canonical. */
const REVERSE = (() => {
  const map = new Map<string, string>();
  const add = (key: string | null, code: string) => {
    if (!key) return;
    const k = key.toLowerCase().replace(/_/g, "-");
    if (!map.has(k) && !BY_CODE.has(k)) map.set(k, code);
  };
  for (const l of TRANSLATION_LANGUAGES) {
    add(l.flores, l.code);
    add(l.flores?.split("_")[0] ?? null, l.code);
  }
  // Azure codes after FLORES: Azure's `ku` (Central Kurdish) must not shadow anything else.
  for (const l of TRANSLATION_LANGUAGES) add(l.azure, l.code);
  return map;
})();

const TRADITIONAL_REGIONS = new Set(["tw", "hk", "mo"]);

/**
 * Canonical code for any BCP-47-ish input (`EN_us` → `en`, `zh-TW` → `zh-Hant`, `jpn_Jpan` → `ja`, `fil` → `tl`);
 * "" for empty, `auto` or unknown input.
 */
export function canonicalLanguage(code: string | null | undefined): string {
  const raw = (code ?? "").trim().replace(/_/g, "-").toLowerCase();
  if (!raw || raw === "auto" || raw === "und") return "";
  const alias = ALIASES[raw];
  if (alias) return alias;
  const exact = BY_CODE.get(raw);
  if (exact) return exact.code;
  const reverse = REVERSE.get(raw);
  if (reverse) return reverse;

  const [base, ...rest] = raw.split("-");
  if (base === "zh" || base === "cmn" || base === "zho") {
    return rest.includes("hant") || rest.some((s) => TRADITIONAL_REGIONS.has(s)) ? "zh-Hant" : "zh";
  }
  if ((base === "pt" || base === "por") && rest.includes("pt")) return "pt-PT";
  return ALIASES[base] ?? BY_CODE.get(base)?.code ?? REVERSE.get(base) ?? "";
}

/** Table row of a canonicalizable code; undefined for unknown codes. */
export function languageInfo(code: string | null | undefined): TranslationLanguage | undefined {
  const canonical = canonicalLanguage(code);
  return canonical ? BY_CODE.get(canonical.toLowerCase()) : undefined;
}

export function toFlores(code: string): string | null {
  return languageInfo(code)?.flores ?? null;
}

export function toAzure(code: string): string | null {
  return languageInfo(code)?.azure ?? null;
}

export function toIso6391(code: string): string | null {
  return languageInfo(code)?.iso6391 ?? null;
}

/** English name of a language ("Japanese"); the raw code for unknown codes; "" for empty. */
export function languageName(code: string | null | undefined): string {
  return languageInfo(code)?.name ?? (code ?? "").trim();
}

/** Whether two codes name the same canonical language (both known). */
export function sameLanguage(a: string | null | undefined, b: string | null | undefined): boolean {
  const ca = canonicalLanguage(a);
  return ca !== "" && ca === canonicalLanguage(b);
}

/** Chinese, Japanese, Korean and Cantonese: LocalAgreement tokenizes them with `Intl.Segmenter`, not spaces. */
export function isCjk(code: string): boolean {
  const base = canonicalLanguage(code).split("-")[0];
  return base === "ja" || base === "zh" || base === "ko" || base === "yue";
}

/**
 * Languages written without spaces between words (scriptio continua): LocalAgreement splits them into words with
 * `Intl.Segmenter`, and joined sentence fragments get no space between them. Korean spaces its words, so it is not
 * one of them (`isCjk` still includes it).
 */
export function isUnspaced(code: string): boolean {
  const base = canonicalLanguage(code).split("-")[0];
  return UNSPACED.has(base);
}

const UNSPACED = new Set(["ja", "zh", "yue", "th", "lo", "km", "my", "bo", "dz", "shn"]);

/** Characters of scripts written without spaces (Han, kana, Thai, Lao, Khmer, Myanmar, Tibetan). */
export const UNSPACED_SCRIPT_RE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}\p{Script=Tibetan}]/u;

/** Script → language for scripts that (nearly) identify one language. */
const SCRIPT_LANGUAGES: readonly (readonly [RegExp, string])[] = [
  [/\p{Script=Hangul}/u, "ko"],
  [/\p{Script=Thai}/u, "th"],
  [/\p{Script=Lao}/u, "lo"],
  [/\p{Script=Khmer}/u, "km"],
  [/\p{Script=Myanmar}/u, "my"],
  [/\p{Script=Tibetan}/u, "bo"],
  [/\p{Script=Georgian}/u, "ka"],
  [/\p{Script=Armenian}/u, "hy"],
  [/\p{Script=Hebrew}/u, "he"],
  [/\p{Script=Greek}/u, "el"],
  [/\p{Script=Bengali}/u, "bn"],
  [/\p{Script=Tamil}/u, "ta"],
  [/\p{Script=Telugu}/u, "te"],
  [/\p{Script=Kannada}/u, "kn"],
  [/\p{Script=Malayalam}/u, "ml"],
  [/\p{Script=Gujarati}/u, "gu"],
  [/\p{Script=Gurmukhi}/u, "pa"],
  [/\p{Script=Sinhala}/u, "si"],
  [/\p{Script=Ethiopic}/u, "am"],
  [/\p{Script=Devanagari}/u, "hi"],
];

/**
 * Best-effort source language from the script alone, for engines that need one when neither the settings nor the
 * STT event name it: kana → `ja`, Han only → `zh`, Hangul → `ko`, Arabic script → `ar` / `fa` / `ur` by their
 * distinctive letters, Cyrillic → `uk` with Ukrainian letters else `ru`, other distinctive scripts → their main
 * language. Latin and mixed-script text → "" (unknown).
 */
export function guessLanguage(text: string): string {
  const letters = [...text].filter((c) => /\p{L}/u.test(c));
  if (letters.length === 0) return "";
  const count = (re: RegExp) => letters.filter((c) => re.test(c)).length;
  const half = letters.length / 2;

  const kana = count(/[\p{Script=Hiragana}\p{Script=Katakana}]/u);
  const han = count(/\p{Script=Han}/u);
  if (kana > 0 && kana + han >= half) return "ja";
  if (han >= half) return "zh";

  const arabic = count(/\p{Script=Arabic}/u);
  if (arabic >= half) {
    if (/[ےٹڈڑں]/u.test(text)) return "ur";
    if (/[پچژگی]/u.test(text) && !/[يى]/u.test(text)) return "fa";
    return "ar";
  }
  const cyrillic = count(/\p{Script=Cyrillic}/u);
  if (cyrillic >= half) return /[іїєґІЇЄҐ]/u.test(text) ? "uk" : "ru";

  for (const [re, code] of SCRIPT_LANGUAGES) {
    if (count(re) >= half) return code;
  }
  return "";
}
