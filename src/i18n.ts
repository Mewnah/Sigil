import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import Backend, { HttpBackendOptions } from 'i18next-http-backend';
import { fileOpen } from "browser-fs-access";

/** Locale folder names under public/i18n/ (underscore convention). */
export const i18nLanguages = [
  { name: 'English', code: 'en' },
  { name: 'Español', code: 'es' },
  { name: 'हिन्दी', code: 'hi' },
  { name: 'العربية', code: 'ar' },
  { name: 'Français', code: 'fr' },
  { name: 'Português (Brasil)', code: 'pt_br' },
  { name: 'Русский', code: 'ru' },
  { name: '日本語', code: 'ja' },
  { name: 'Deutsch', code: 'de' },
  { name: '한국어', code: 'ko' },
  { name: 'Italiano', code: 'it' },
  { name: 'Bahasa Indonesia', code: 'id' },
  { name: 'Türkçe', code: 'tr' },
  { name: 'Tiếng Việt', code: 'vi' },
  { name: 'Polski', code: 'pl' },
  { name: 'ไทย', code: 'th' },
  { name: 'বাংলা', code: 'bn' },
  { name: 'اردو', code: 'ur' },
  { name: '中文（简体）', code: 'zh_cn' },
  { name: '中文（繁体）', code: 'zh_tw' },
];

/** RTL UI: smoke-test `ar` / `ur` for inspector, dropdowns, and toasts after substantive layout changes. */
const RTL_LOCALES = new Set(['ar', 'ur']);

let documentDirHooked = false;

/** BCP 47-style `lang` for `<html>` (best-effort for underscore locale ids). */
function languageToHtmlLang(code: string): string {
  if (code === 'zh_cn') return 'zh-CN';
  if (code === 'zh_tw') return 'zh-TW';
  if (code === 'pt_br') return 'pt-BR';
  if (code === 'dev') return 'en';
  return code.replace(/_/g, '-');
}

function syncDocumentLocale(lng: string) {
  if (typeof document === 'undefined') return;
  const dir = RTL_LOCALES.has(lng) ? 'rtl' : 'ltr';
  document.documentElement.setAttribute('dir', dir);
  document.documentElement.setAttribute('lang', languageToHtmlLang(lng));
}

export async function loadLanguageFile() {
  try {
    const f = await fileOpen({
      description: 'Json file',
      mimeTypes: ['application/json'],
      extensions: ['.json'],
    });
    const json = await new Response(f).json();
    i18n.addResourceBundle("dev", "translation", json, true, true);
    await changeLanguage("dev");
  } catch (error) {

  }
}

export async function changeLanguage(language: string): Promise<void> {
  await i18n.changeLanguage(language);
  syncDocumentLocale(language);
}

export async function initI18n(selectedLanguage: string) {
  const lngCodes = i18nLanguages.map(l => l.code);
  const result = await i18n
    .use(Backend)
    .use(initReactI18next)
    .init<HttpBackendOptions>({
      backend: {
        loadPath: '/i18n/{{lng}}/{{ns}}.json'
      },
      preload: [selectedLanguage],
      supportedLngs: [...lngCodes, "dev", "zh"],
      fallbackLng: lngCodes[0] || 'en',
      debug: import.meta.env.TAURI_DEBUG,
      lng: selectedLanguage,

      interpolation: {
        escapeValue: false
      }
    });

  syncDocumentLocale(i18n.language || selectedLanguage);
  if (!documentDirHooked) {
    documentDirHooked = true;
    i18n.on('languageChanged', (lng) => {
      syncDocumentLocale(lng);
    });
  }

  return result;
}
