/**
 * Server-side render tests of the experimental ggml voices panel (`src/core/ui/inspector/tts/GgmlTts.tsx`, P6-F): what
 * each engine shows, the downloads the plan asks for, and the local-only hints. Inputs, the requirement card, the
 * voice-library picker and the output picker are stubbed to plain markup carrying their props.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { proxy } from "valtio";
import i18n from "i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key}${JSON.stringify(opts)}` : key),
    i18n: { language: "en" },
  }),
}));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn() }) }));
vi.mock("@/core/ui/inspector/components", () => ({
  default: {
    SubHeader: ({ children }: { children: ReactNode }) => <h3>{children}</h3>,
    Description: ({ children }: { children: ReactNode }) => <p>{children}</p>,
  },
}));
vi.mock("@/core/ui/inspector/components/input", () => ({
  InputSelect: ({ label, value, options }: { label: string; value: string; options: { label: string; value: string }[] }) => (
    <label data-field={label}>
      <select defaultValue={value}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  ),
  InputText: ({ label, value, placeholder }: { label: string; value: string; placeholder?: string }) => <input data-field={label} defaultValue={value} placeholder={placeholder} />,
  InputRange: ({ label, value }: { label: string; value: string }) => <input data-field={label} defaultValue={value} />,
  InputCheckbox: ({ label, value }: { label: string; value: boolean }) => <input type="checkbox" data-field={label} defaultChecked={value} />,
}));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({
  ModelRequirement: ({ ids }: { ids: string[] }) => <div data-requirement={ids.join(",")} />,
  useModelsLoaded: () => {},
}));
vi.mock("@/core/ui/inspector/components/ReferenceVoiceRecorder", () => ({
  ReferenceVoicePicker: ({ value, needsTranscript, allowNone }: { value: string; needsTranscript?: boolean; allowNone?: boolean }) => (
    <div data-picker={value} data-transcript={String(!!needsTranscript)} data-none={String(allowNone)} />
  ),
}));
vi.mock("@/core/ui/inspector/tts/Routing", () => ({
  TtsOutputDevice: ({ label, value }: { label: string; value: string }) => <div data-output={value}>{label}</div>,
}));

import { Service_TTS_Schema, type TTS_State } from "@/core/services/tts/schema";
import { ggmlUi } from "@/core/services/tts/services/ggmlTts";
import { GgmlTts } from "@/core/ui/inspector/tts/GgmlTts";

function render(): string {
  return renderToStaticMarkup(<GgmlTts />).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

let data: TTS_State;
let states: Record<string, { state: string }>;

beforeEach(() => {
  data = proxy(Service_TTS_Schema.parse({ backend: "ggmlTts" }));
  states = proxy({});
  ggmlUi.last = null;
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models: { state: proxy({ states, catalog: [], pending: {}, loaded: true }) },
      sysinfo: { state: proxy({ info: null }) },
      tts: { serviceState: proxy({ status: "disconnected", error: "", pending: 0, speaking: false }) },
      patchService: vi.fn(),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("<GgmlTts> panel", () => {
  it("shows Magpie's voices, languages and downloads, with the tokenizer hint", () => {
    const html = render();
    expect(html).toContain('data-requirement="tts/magpie-tts-357m-v2602,tts/magpie-tts-357m-v2602-tokenizer,tts/nemo-nano-codec-22khz,runtime/nemo-speech-vulkan"');
    for (const v of ["John", "Sofia", "Aria", "Jason", "Leo"]) expect(html).toContain(`value="${v}"`);
    expect(html).toContain('<option value="de">German</option>');
    expect(html).not.toContain('<option value="ja">');
    expect(html).toContain("tts.ggml_magpie_tokenizer_hint");
    expect(html).toContain("tts.ggml_magpie_languages_hint");
    expect(html).not.toContain("data-picker");
    // Magpie has CUDA, Vulkan and processor packs; no CUDA 13 one.
    expect(html).toContain('value="cuda"');
    expect(html).not.toContain('value="cuda13"');
  });

  it("shows Breeze's description, guidance and the optional voice to clone", () => {
    data.ggmlTts.engine = "breeze";
    let html = render();
    expect(html).toContain("tts.ggml_breeze_description");
    expect(html).toContain('data-field="tts.ggml_breeze_cfg"');
    expect(html).toContain('data-picker="" data-transcript="true" data-none="true"');
    expect(html).toContain("tts.ggml_no_clone_breeze");
    expect(html).toContain('data-requirement="tts/breeze-tts-2-q8_0,runtime/breeze-server-vulkan"');
    expect(html).not.toContain("tts.ggml_magpie_tokenizer_hint");
    data.ggmlTts.referenceVoiceId = "alice-1";
    html = render();
    expect(html).toContain("tts.ggml_breeze_direction");
    expect(html).toContain("tts.ggml_clone_hint");
  });

  it("asks OmniVoice users for consent before cloning", () => {
    data.ggmlTts.engine = "omnivoice";
    let html = render();
    expect(html).not.toContain("tts.ggml_omnivoice_consent");
    expect(html).toContain('data-field="tts.ggml_quality"');
    data.ggmlTts.referenceVoiceId = "alice-1";
    html = render();
    expect(html).toContain('data-field="tts.ggml_omnivoice_consent"');
    expect(html).toContain("tts.ggml_omnivoice_consent_missing");
    expect(html).toContain("tts.ggml_omnivoice_disclose_off");
  });

  it("points S2 users to the pack build and offers the low video memory switch", () => {
    data.ggmlTts.engine = "s2";
    const html = render();
    expect(html).toContain("tts.ggml_s2_pack_hint");
    expect(html).toContain('data-field="tts.ggml_s2_low_vram"');
    expect(html).toContain("tts.ggml_no_clone_s2");
    expect(html).toContain('data-requirement="tts/s2-pro-q4_k_m,tts/s2-pro-tokenizer,runtime/s2-server-vulkan"');
  });

  it("shows the language as set, and names one the checkpoint cannot speak", () => {
    // Korean chosen with v2607, then the model went back to v2602 (older settings; the panel now resets it).
    data.ggmlTts.language = "ko";
    let html = render();
    expect(html).toContain('<option value="ko" selected="">Korean</option>');
    expect(html).toContain("tts.ggml_magpie_language_unsupported");
    // Choosing "Automatic" is a real change then, not the value already shown.
    expect(html).toContain('<option value="auto">tts.ggml_language_auto</option>');
    data.ggmlTts.model = "v2607";
    html = render();
    expect(html).toContain('<option value="ko" selected="">Korean</option>');
    expect(html).not.toContain("tts.ggml_magpie_language_unsupported");
    // OmniVoice speaks languages beyond its short list; the one set stays visible and valid.
    data.ggmlTts.engine = "omnivoice";
    data.ggmlTts.language = "sw";
    html = render();
    expect(html).toContain('<option value="sw" selected="">Swahili</option>');
    expect(html).not.toContain("tts.ggml_magpie_language_unsupported");
  });

  it("shows a refused variant and the last sentence of the engine on screen", () => {
    data.ggmlTts.engine = "breeze";
    data.ggmlTts.variant = "cuda13";
    ggmlUi.last = { engine: "breeze", firstAudioMs: 280, audioSec: 1.5, totalMs: 900, at: 1 };
    const html = render();
    expect(html).toContain("tts.ggml_variant_missing");
    expect(html).toContain('tts.ggml_last_sentence{"first":280,"seconds":1.5}');
  });
});
