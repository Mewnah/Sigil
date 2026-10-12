import { OpenAiAudioPreset, STT_Backends } from "@/core/services/stt/schema";

/**
 * The engine list of the Speech to Text inspector, grouped (pure: the inspector renders it, the unit test checks it).
 *
 * - Recommended: the built-in engine (`local`), which picks and runs the speech models itself.
 * - Online (your own account or key): Groq, Deepgram, Azure, Gemini Live.
 * - Chrome / Edge microphone page: the two browser engines, in their own group so they stay easy to find.
 * - Other engines: Whisper, Vosk, your own server; Web Speech only where it can work (the web build).
 *
 * The Moonshine server backend is never listed: it was retired, and profiles on it move to the built-in engine at
 * launch (`planRetiredBackendMigration`).
 */

/** Provider list value of the Groq preset (stored as backend `openai_audio` + preset `groq`). */
export const PROVIDER_GROQ = "groq";

export type SttEngineOption = { label: string; value: string };
export type SttEngineGroup = { label: string; options: SttEngineOption[] };

type Translate = (key: string) => string;

/** The list value of the profile's engine: Groq is `openai_audio` with the Groq preset. */
export function sttProviderValue(backend: STT_Backends, openAiAudioPreset: OpenAiAudioPreset): string {
  return backend === STT_Backends.openai_audio && openAiAudioPreset === OpenAiAudioPreset.groq ? PROVIDER_GROQ : backend;
}

export interface SttEngineGroupsInput {
  /** Running in the Tauri desktop app: Web Speech cannot work in its webview, so it is not offered there. */
  desktop: boolean;
  /** `sttProviderValue` of the profile: it is always in the list, so the selector never shows "Select". */
  current: string;
}

export function sttEngineGroups(t: Translate, { desktop, current }: SttEngineGroupsInput): SttEngineGroup[] {
  const groups: SttEngineGroup[] = [
    { label: t("stt.group_recommended"), options: [{ label: t("stt.provider_local"), value: STT_Backends.local }] },
    {
      label: t("stt.group_online"),
      options: [
        { label: t("stt.provider_groq"), value: PROVIDER_GROQ },
        { label: "Deepgram", value: STT_Backends.deepgram },
        { label: "Azure", value: STT_Backends.azure },
        { label: t("stt.provider_gemini_live"), value: STT_Backends.geminiLive },
      ],
    },
    {
      label: t("stt.group_browser"),
      options: [
        { label: "Chrome", value: STT_Backends.chrome },
        { label: "Edge", value: STT_Backends.edge },
      ],
    },
    {
      label: t("stt.group_other"),
      options: [
        { label: "Whisper", value: STT_Backends.whisper },
        { label: "Vosk", value: STT_Backends.vosk },
        { label: t("stt.provider_openai_audio"), value: STT_Backends.openai_audio },
        ...(desktop ? [] : [{ label: t("stt.provider_native"), value: STT_Backends.native }]),
      ],
    },
  ];

  // A profile on an engine that is not listed (Web Speech on the desktop before its offer is answered, the retired
  // Moonshine server in the web build, which has no built-in engine to move it to) keeps showing its own name.
  const listed = groups.some((g) => g.options.some((o) => o.value === current));
  if (!listed) {
    const native = current === STT_Backends.native || current === STT_Backends.browser;
    const label = native ? t("stt.provider_native") : current === STT_Backends.moonshine ? t("stt.provider_moonshine") : current;
    groups[groups.length - 1].options.push({ label, value: current });
  }
  return groups;
}
