/**
 * Server-side render tests of the P6-G panels (`src/core/ui/inspector/tts/{Sbv2,GptSovits}.tsx`): the Style-Bert-VITS2
 * voice list, speakers, styles, device and import zone, and the GPT-SoVITS server, reference and streaming fields for a
 * server on this PC and one on the network. Inputs, the requirement card, the reference picker and the output picker are
 * stubbed to plain markup carrying their props.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key}${JSON.stringify(opts)}` : key),
    i18n: { language: "en" },
  }),
}));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/webview", () => ({ getCurrentWebview: () => ({ onDragDropEvent: vi.fn(async () => () => {}) }) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn(async () => null) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), success: vi.fn(), info: vi.fn() }) }));
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
  InputRange: ({ label, value, disabled }: { label: string; value: string; disabled?: boolean }) => (
    <input data-field={label} defaultValue={value} disabled={disabled} />
  ),
  InputCheckbox: ({ label, value }: { label: string; value: boolean }) => <input type="checkbox" data-field={label} defaultChecked={value} />,
  InputCode: ({ label, value }: { label: string; value: string }) => <textarea data-field={label} defaultValue={value} />,
}));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({
  ModelRequirement: ({ ids }: { ids: string[] }) => <div data-requirement={ids.join(",")} />,
  useModelsLoaded: () => {},
}));
vi.mock("@/core/ui/inspector/components/ReferenceVoiceRecorder", () => ({
  ReferenceVoicePicker: ({ value, needsTranscript }: { value: string; needsTranscript?: boolean }) => (
    <div data-reference={value} data-needs-transcript={String(!!needsTranscript)} />
  ),
}));
vi.mock("@/core/ui/inspector/tts/Routing", () => ({
  TtsOutputDevice: ({ label, value }: { label: string; value: string }) => <div data-output={value}>{label}</div>,
}));

import { Service_TTS_Schema, type TTS_State } from "@/core/services/tts/schema";
import { SBV2_REQUIREMENTS, sbv2Library, type Sbv2Model } from "@/core/services/tts/services/sbv2";
import { GptSovits } from "@/core/ui/inspector/tts/GptSovits";
import { Sbv2 } from "@/core/ui/inspector/tts/Sbv2";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

const MAO: Sbv2Model = {
  id: "mao-abc123",
  name: "まお",
  format: "aivmx",
  speakers: ["まお", "Second"],
  styles: ["ノーマル", "あまあま"],
  voices: [
    { name: "まお", id: 0, styles: [{ name: "ノーマル", id: 0 }, { name: "あまあま", id: 1 }], languages: ["ja"] },
    { name: "Second", id: 1, styles: [{ name: "ノーマル", id: 0 }], languages: ["ja"] },
  ],
  description: "A calm voice",
  creators: ["Aivis"],
  architecture: "Style-Bert-VITS2 (JP-Extra)",
  sampleRate: 44_100,
  sizeBytes: 250 * 1024 * 1024,
};

let data: TTS_State;
let serviceState: { status: string };
let store: { states: Record<string, unknown>; catalog: unknown[] };

beforeEach(() => {
  data = proxy(Service_TTS_Schema.parse({}));
  serviceState = proxy({ status: "disconnected" });
  store = proxy({ states: {}, catalog: [] });
  sbv2Library.models = [];
  sbv2Library.loading = false;
  sbv2Library.importing = false;
  sbv2Library.error = null;
  vi.stubGlobal("window", {
    devicePixelRatio: 1,
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      tts: { serviceState },
      models: { state: store },
      patchService: vi.fn(),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Style-Bert-VITS2 panel", () => {
  it("asks for the downloads and a first voice", () => {
    const out = html(<Sbv2 />);
    expect(out).toContain(`data-requirement="${SBV2_REQUIREMENTS.join(",")}"`);
    expect(out).toContain('data-field="tts.sbv2_voice" data-value=""');
    expect(out).toContain('<option value="" selected="">tts.sbv2_choose_voice</option>');
    expect(out).toContain("tts.sbv2_no_voices");
    expect(out).toContain("tts.sbv2_drop_hint");
    expect(out).toContain("tts.sbv2_import");
    // The processor by default, with its threads.
    expect(out).toContain('data-field="tts.sbv2_device" data-value="cpu"');
    expect(out).toContain('data-field="tts.local_neural_threads" data-value="auto"');
    // No voice chosen and nothing downloaded: no speaker or style, and no preview.
    expect(out).not.toContain('data-field="tts.sbv2_speaker"');
    expect(out).not.toContain('data-field="tts.sbv2_style"');
    expect(out).toMatch(/<button[^>]*disabled=""[^>]*title="tts.local_neural_preview_needs_download"/);
  });

  it("shows the chosen voice's facts, speakers and styles", () => {
    sbv2Library.models = [MAO];
    data.sbv2.model = MAO.id;
    data.sbv2.speaker = "まお";
    data.sbv2.style = "あまあま";
    for (const id of SBV2_REQUIREMENTS) store.states[id] = { id, state: "installed" };
    const out = html(<Sbv2 />);
    expect(out).toContain(`data-field="tts.sbv2_voice" data-value="${MAO.id}"`);
    expect(out).toContain("Style-Bert-VITS2 (JP-Extra) · 44.1 kHz");
    expect(out).toContain("A calm voice");
    expect(out).toContain('tts.sbv2_creators{"names":"Aivis"}');
    expect(out).toContain('data-field="tts.sbv2_speaker" data-value="まお"');
    expect(out).toContain('<option value="Second">Second</option>');
    expect(out).toContain('data-field="tts.sbv2_style" data-value="あまあま"');
    // A real style: the strength applies.
    expect(out).toMatch(/data-field="tts.sbv2_style_weight"(?![^>]*disabled)/);
    expect(out).toContain("tts.sbv2_style_weight_hint");
    // Downloads ready and a voice chosen: the preview is enabled.
    expect(out).not.toContain("tts.local_neural_preview_needs_download");
    expect(out).toContain("tts.sbv2_remove");
  });

  it("falls back to the neutral style, flags a missing voice and locks the engine while it runs", () => {
    sbv2Library.models = [MAO];
    data.sbv2.model = "gone-000001";
    data.sbv2.executionProvider = "dml";
    serviceState.status = "connected";
    const out = html(<Sbv2 />);
    expect(out).toContain('tts.sbv2_missing_voice{"id":"gone-000001"}');
    expect(out).not.toContain('data-field="tts.sbv2_speaker"');
    // A GPU device has no thread setting; the engine settings are locked while it runs.
    expect(out).toContain('data-field="tts.sbv2_device" data-value="dml"');
    expect(out).not.toContain('data-field="tts.local_neural_threads"');
    expect(out).toContain('data-active="false"');

    data.sbv2.model = MAO.id;
    data.sbv2.style = "Neutral";
    const neutral = html(<Sbv2 />);
    // The manifest calls the neutral style ノーマル; "Neutral" picks it (id 0) and the strength is moot.
    expect(neutral).toContain('data-field="tts.sbv2_style" data-value="ノーマル"');
    expect(neutral).toMatch(/data-field="tts.sbv2_style_weight"[^>]*disabled=""/);
    expect(neutral).toContain("tts.sbv2_style_weight_neutral");
  });
});

describe("GPT-SoVITS panel", () => {
  it("offers the voice library and a clip on this PC for a local server", () => {
    const out = html(<GptSovits />);
    expect(out).toContain("python api_v2.py -a 127.0.0.1 -p 9880");
    expect(out).toContain('data-field="tts.gpt_sovits_gpt_weights"');
    expect(out).toContain('data-field="tts.gpt_sovits_sovits_weights"');
    expect(out).toContain('data-reference="" data-needs-transcript="true"');
    expect(out).not.toContain("tts.gpt_sovits_library_needs_local");
    // No library voice: the clip path, with a file picker for this PC.
    expect(out).toContain('data-field="tts.gpt_sovits_ref_path"');
    expect(out).toContain("tts.gpt_sovits_ref_browse");
    // Streaming on by default, with its mode.
    expect(out).toContain('data-field="tts.gpt_sovits_streaming"');
    expect(out).toContain('data-field="tts.gpt_sovits_streaming_mode" data-value="fragments"');
    expect(out).toContain('<option value="fast">tts.gpt_sovits_streaming_fast</option>');
    expect(out).toContain('data-field="tts.gpt_sovits_text_lang" data-value="auto"');
    expect(out).not.toContain("tts.openai_speech_extra_invalid");
  });

  it("hides the clip path behind a library voice on this PC", () => {
    data.gptSovits.referenceVoiceId = "alice-1a2b3c";
    const out = html(<GptSovits />);
    expect(out).toContain('data-reference="alice-1a2b3c"');
    expect(out).not.toContain('data-field="tts.gpt_sovits_ref_path"');
    expect(out).toContain('placeholder="tts.gpt_sovits_prompt_from_library"');
  });

  it("asks for a path on the server for a server elsewhere, and flags bad extra fields", () => {
    data.gptSovits.baseUrl = "http://192.168.1.20:9880";
    data.gptSovits.referenceVoiceId = "alice-1a2b3c";
    data.gptSovits.streaming = false;
    data.gptSovits.extra = "[1, 2]";
    const out = html(<GptSovits />);
    expect(out).toContain("tts.gpt_sovits_library_needs_local");
    expect(out).toContain('data-field="tts.gpt_sovits_ref_path"');
    // The file picker would pick a file on this PC, which that server cannot read.
    expect(out).not.toContain("tts.gpt_sovits_ref_browse");
    expect(out).not.toContain('data-field="tts.gpt_sovits_streaming_mode"');
    expect(out).toContain("tts.openai_speech_extra_invalid");
  });
});
