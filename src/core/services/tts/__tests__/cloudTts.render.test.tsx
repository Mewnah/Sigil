/**
 * Server-side render tests of the Edge and Gemini TTS panels (`src/core/ui/inspector/tts/{Edge,Gemini}.tsx`, P6-H):
 * the voice pickers, the fallback section with its Kokoro requirement, and the Gemini key / model fields. Inputs, the
 * requirement card and the output picker are stubbed to plain markup carrying their props.
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
vi.mock("@tauri-apps/plugin-shell", () => ({ open: vi.fn(async () => undefined) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn(), warning: vi.fn() }) }));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/core/ui/inspector/components", () => ({
  default: {
    SubHeader: ({ children }: { children: ReactNode }) => <h3>{children}</h3>,
    Description: ({ children }: { children: ReactNode }) => <p>{children}</p>,
    Deactivatable: ({ active, children }: { active: boolean; children: ReactNode }) => <div data-active={String(active)}>{children}</div>,
  },
}));
vi.mock("@/core/ui/inspector/components/input", () => ({
  InputSelect: ({ label, value, options }: { label: string; value: string; options: { label: string; value: string }[] }) => (
    <label data-field={label} data-value={value}>
      <select defaultValue={value}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  ),
  InputText: ({ label, value, placeholder }: { label: string; value: string; placeholder?: string }) => (
    <input data-field={label} defaultValue={value} placeholder={placeholder} />
  ),
  InputRange: ({ label, value }: { label: string; value: string }) => <input data-field={label} defaultValue={value} />,
  InputCheckbox: ({ label, value }: { label: string; value: boolean }) => <input type="checkbox" data-field={label} defaultChecked={value} />,
}));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({
  ModelRequirement: ({ ids }: { ids: string[] }) => <div data-requirement={ids.join(",")} />,
  useModelsLoaded: () => {},
}));
vi.mock("@/core/ui/inspector/tts/Routing", () => ({
  TtsOutputDevice: ({ label, value }: { label: string; value: string }) => <div data-output={value}>{label}</div>,
}));

import { Service_TTS_Schema, type TTS_State } from "@/core/services/tts/schema";
import { FALLBACK_MODEL, edgeVoicesState } from "@/core/services/tts/services/edge";
import { GEMINI_TTS_MODELS, GEMINI_VOICES } from "@/core/services/tts/services/gemini";
import { Edge } from "@/core/ui/inspector/tts/Edge";
import { Gemini } from "@/core/ui/inspector/tts/Gemini";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

let data: TTS_State;
let serviceState: { status: string };

beforeEach(() => {
  data = proxy(Service_TTS_Schema.parse({}));
  serviceState = proxy({ status: "disconnected" });
  edgeVoicesState.voices = [];
  edgeVoicesState.loading = false;
  edgeVoicesState.error = "";
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      tts: { serviceState },
      patchService: vi.fn(),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Edge panel", () => {
  it("offers a typed voice while the list is not loaded, and the Kokoro fallback", () => {
    edgeVoicesState.error = "offline";
    const out = html(<Edge />);
    expect(out).toContain('data-field="tts.field_voice"');
    expect(out).toContain('placeholder="en-US-AvaMultilingualNeural"');
    expect(out).toContain('tts.edge_voices_failed{"error":"offline"}');
    expect(out).toContain('data-field="tts.cloud_fallback_enable"');
    expect(out).toContain('data-field="tts.cloud_fallback_after"');
    expect(out).toContain(`data-requirement="${FALLBACK_MODEL}"`);
    // Stopped: the fallback switch is editable.
    expect(out).toContain('data-active="true"');
  });

  it("picks language and voice from the service's list", () => {
    edgeVoicesState.voices = [
      { shortName: "ja-JP-KeitaNeural", friendlyName: "", locale: "ja-JP", gender: "Male" },
      { shortName: "ja-JP-NanamiNeural", friendlyName: "", locale: "ja-JP", gender: "Female" },
      { shortName: "en-US-AvaMultilingualNeural", friendlyName: "", locale: "en-US", gender: "Female" },
    ];
    data.edge.voice = "ja-JP-NanamiNeural";
    const out = html(<Edge />);
    expect(out).toContain('data-field="tts.edge_language" data-value="ja-JP"');
    expect(out).toContain('data-field="tts.field_voice" data-value="ja-JP-NanamiNeural"');
    // Only the voices of the chosen language.
    expect(out).toContain('value="ja-JP-KeitaNeural"');
    expect(out).not.toContain('<option value="en-US-AvaMultilingualNeural"');
    expect(out).toContain('<option value="en-US"');
  });

  it("accepts the listed voices whose locale has a script subtag", () => {
    edgeVoicesState.voices = [
      { shortName: "iu-Cans-CA-SiqiniqNeural", friendlyName: "", locale: "iu-Cans-CA", gender: "Female" },
      { shortName: "iu-Latn-CA-SiqiniqNeural", friendlyName: "", locale: "iu-Latn-CA", gender: "Female" },
    ];
    data.edge.voice = "iu-Cans-CA-SiqiniqNeural";
    const out = html(<Edge />);
    expect(out).toContain('data-field="tts.edge_language" data-value="iu-Cans-CA"');
    expect(out).toContain('data-field="tts.field_voice" data-value="iu-Cans-CA-SiqiniqNeural"');
    expect(out).not.toContain("tts.edge_voice_invalid");
  });

  it("flags a voice name the service would refuse, and hides the fallback options when it is off", () => {
    data.edge.voice = "Ava";
    data.edge.fallbackToLocal = false;
    serviceState.status = "connected";
    const out = html(<Edge />);
    expect(out).toContain('tts.edge_voice_invalid{"voice":"Ava"}');
    expect(out).not.toContain("data-requirement");
    expect(out).not.toContain('data-field="tts.cloud_fallback_after"');
    // Running: the fallback switch is locked.
    expect(out).toContain('data-active="false"');
  });
});

describe("Gemini panel", () => {
  it("asks for a key and lists the models and voices", () => {
    const out = html(<Gemini />);
    expect(out).toContain('data-field="tts.gemini_api_key"');
    expect(out).toContain("tts.gemini_key_hint");
    expect(out).toContain(`data-field="tts.gemini_model" data-value="${GEMINI_TTS_MODELS[0].id}"`);
    for (const m of GEMINI_TTS_MODELS) expect(out).toContain(`value="${m.id}"`);
    expect(out).not.toContain('data-field="tts.gemini_model_id"');
    expect(out).toContain('data-field="tts.field_voice" data-value="Kore"');
    expect(out.match(/tts\.gemini_voice_option/g)?.length).toBe(GEMINI_VOICES.length);
    expect(out).toContain('<option value="custom voice">tts.gemini_voice_custom</option>');
    expect(out).not.toContain('data-field="tts.gemini_voice_id"');
    expect(out).toContain(`data-requirement="${FALLBACK_MODEL}"`);
    expect(out).toContain('data-field="tts.gemini_base_url"');
  });

  it("shows the id fields for a model and a voice it does not list", () => {
    data.gemini.apiKey = "k";
    data.gemini.model = "gemini-9-tts";
    data.gemini.voice = "voice_abc123";
    const out = html(<Gemini />);
    expect(out).not.toContain("tts.gemini_key_hint");
    expect(out).toContain('data-field="tts.gemini_model" data-value="custom model"');
    expect(out).toContain('data-field="tts.gemini_model_id"');
    // A designed voice (or an extended-library name) keeps its own field.
    expect(out).toContain('data-field="tts.field_voice" data-value="custom voice"');
    expect(out).toContain('<input data-field="tts.gemini_voice_id" placeholder="Kore" value="voice_abc123"/>');
    expect(out).toContain("tts.gemini_voice_id_hint");
  });
});
