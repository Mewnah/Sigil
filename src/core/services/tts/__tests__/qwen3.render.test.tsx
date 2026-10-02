/**
 * Server-side render tests of the Qwen3-TTS panel (`src/core/ui/inspector/tts/Qwen3.tsx`, P6-E): what each mode shows,
 * the downloads the plan asks for, and the governor's reasons. Inputs, the requirement card, the voice-library picker
 * and the output picker are stubbed to plain markup carrying their props.
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
    Deactivatable: ({ active, children }: { active: boolean; children: ReactNode }) => <div data-active={String(active)}>{children}</div>,
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
import { qwen3Ui } from "@/core/services/tts/services/qwen3";
import { Qwen3 } from "@/core/ui/inspector/tts/Qwen3";

function render(): string {
  return renderToStaticMarkup(<Qwen3 />).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

let data: TTS_State;
let recommendations: Record<string, unknown>;
let states: Record<string, { state: string }>;

beforeEach(() => {
  data = proxy(Service_TTS_Schema.parse({ backend: "qwen3" }));
  recommendations = proxy({ qwen3_tts: { choice: "tts/qwen3-tts-1.7b", alternatives: ["tts/qwen3-tts-0.6b"], device: "gpu", reason_code: "fits" } });
  states = proxy({});
  qwen3Ui.last = null;
  qwen3Ui.swapBlocked = null;
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models: { state: proxy({ states, catalog: [], pending: {}, loaded: true }) },
      resources: { state: proxy({ recommendations }) },
      sysinfo: { state: proxy({ info: null }) },
      tts: { serviceState: proxy({ status: "disconnected", error: "", pending: 0, speaking: false }) },
      patchService: vi.fn(),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("<Qwen3> panel", () => {
  it("shows the built-in voices, style presets and what to download", () => {
    const html = render();
    expect(html).toContain('data-requirement="tts/qwen3-tts-1.7b,tts/qwen3-tts-tokenizer,runtime/qwentts-server-vulkan"');
    for (const id of ["ryan", "aiden", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee"]) expect(html).toContain(`value="${id}"`);
    expect(html).toContain("tts.qwen3_style_happy");
    expect(html).toContain('tts.qwen3_reason_fits{"size":"1.7B"}');
    expect(html).toContain('tts.qwen3_model_auto{"size":"1.7B"}');
    expect(html).not.toContain("tts.qwen3_instruction_needs_large");
    expect(html).toContain("tts.qwen3_pack_hint");
    expect(html).not.toContain("data-picker");
    expect(html).toContain('<option value="ja">Japanese</option>');
  });

  it("follows the governor to 0.6B next to the voice changer and says instructions need 1.7B", () => {
    recommendations.qwen3_tts = { choice: "tts/qwen3-tts-0.6b", alternatives: [], device: "gpu", reason_code: "vc_running" };
    const html = render();
    expect(html).toContain('data-requirement="tts/qwen3-tts-0.6b,tts/qwen3-tts-tokenizer,runtime/qwentts-server-vulkan"');
    expect(html).toContain('tts.qwen3_reason_vc_running{"size":"0.6B"}');
    expect(html).toContain("tts.qwen3_instruction_needs_large");
  });

  it("mounts the voice-library picker for cloning", () => {
    data.qwen3.mode = "clone";
    data.qwen3.referenceVoiceId = "alice-1a2b3c";
    const html = render();
    expect(html).toContain('data-picker="alice-1a2b3c" data-transcript="true" data-none="false"');
    expect(html).toContain('data-requirement="tts/qwen3-tts-1.7b-base,tts/qwen3-tts-tokenizer,runtime/qwentts-server-vulkan"');
    expect(html).not.toContain('value="sohee"');
    expect(html).not.toContain("tts.qwen3_style_happy");
  });

  it("asks for a description in design mode and locks the size to 1.7B", () => {
    data.qwen3.mode = "design";
    data.qwen3.model = "0.6b";
    const html = render();
    expect(html).toContain("tts.qwen3_design_description");
    expect(html).toContain("tts.qwen3_design_large_only");
    expect(html).toContain('data-active="false"');
    expect(html).toContain('data-requirement="tts/qwen3-tts-1.7b-voicedesign,tts/qwen3-tts-tokenizer,runtime/qwentts-server-vulkan"');
    expect(html).toContain("tts.qwen3_design_description_missing");
    expect(html).toContain('placeholder="tts.qwen3_seed_design_placeholder"');
  });

  it("warns about the processor and about a size that is not downloaded", () => {
    recommendations.qwen3_tts = { choice: "tts/qwen3-tts-0.6b", alternatives: [], device: "cpu", reason_code: "no_gpu" };
    qwen3Ui.swapBlocked = "tts/qwen3-tts-0.6b";
    qwen3Ui.last = { firstAudioMs: 180, audioSec: 2.4, totalMs: 2600, at: 1 };
    const html = render();
    expect(html).toContain("tts.qwen3_cpu_slow");
    expect(html).toContain("tts.qwen3_reason_cpu");
    expect(html).toContain("runtime/qwentts-server-cpu");
    expect(html).toContain('tts.qwen3_swap_blocked{"size":"0.6B"}');
    expect(html).toContain('tts.qwen3_last_sentence{"first":180,"seconds":2.4}');
  });

  it("shows the server import hint only while the server pack is missing", () => {
    // Only the model is missing: it downloads normally, so no import hint.
    states["runtime/qwentts-server-vulkan"] = { state: "installed" };
    expect(render()).not.toContain("tts.qwen3_pack_hint");
    delete states["runtime/qwentts-server-vulkan"];
    for (const id of ["tts/qwen3-tts-1.7b", "tts/qwen3-tts-tokenizer"]) states[id] = { state: "installed" };
    expect(render()).toContain("tts.qwen3_pack_hint");
    states["runtime/qwentts-server-vulkan"] = { state: "installed" };
    expect(render()).not.toContain("tts.qwen3_pack_hint");
  });

  it("asks for the NVIDIA runtime with the CUDA server pack", () => {
    data.qwen3.variant = "cuda13";
    expect(render()).toContain('data-requirement="tts/qwen3-tts-1.7b,tts/qwen3-tts-tokenizer,runtime/qwentts-server-cuda13,runtime/nvidia-cuda13"');
  });
});
