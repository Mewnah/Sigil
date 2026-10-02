/**
 * Server-side render tests of the cloning panels (`src/core/ui/inspector/tts/{Pocket,ChatterboxLocal}.tsx`, P6-I):
 * the downloads they ask for, the voice picker's mode, the hints, and the execution-provider line. Inputs, the
 * requirement card, the voice-library picker and the output picker are stubbed to plain markup carrying their props.
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
vi.mock("@/core/services/diagnostics/bridge", async () => {
  const { proxy: p } = await import("valtio");
  return { diagnosticsState: p({ outcomes: {} as Record<string, unknown> }) };
});
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
  InputRange: ({ label, value }: { label: string; value: string }) => <input data-field={label} defaultValue={value} />,
}));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({
  ModelRequirement: ({ ids }: { ids: string[] }) => <div data-requirement={ids.join(",")} />,
  useModelsLoaded: () => {},
}));
vi.mock("@/core/ui/inspector/components/ReferenceVoiceRecorder", () => ({
  ReferenceVoicePicker: ({ value, allowNone, label }: { value: string; allowNone?: boolean; label?: string }) => (
    <div data-picker={value} data-none={String(allowNone)} data-label={label} />
  ),
}));
vi.mock("@/core/ui/inspector/tts/Routing", () => ({
  TtsOutputDevice: ({ label, value }: { label: string; value: string }) => <div data-output={value}>{label}</div>,
}));

import { diagnosticsState } from "@/core/services/diagnostics/bridge";
import { Service_TTS_Schema, type TTS_State } from "@/core/services/tts/schema";
import { CHATTERBOX_LOCAL_GPU_FLOW, CHATTERBOX_LOCAL_MODEL, CHATTERBOX_PREVIEW_TEXT } from "@/core/services/tts/services/chatterboxLocal";
import { POCKET_DEFAULT_MODEL } from "@/core/services/tts/services/pocket";
import { ChatterboxLocal } from "@/core/ui/inspector/tts/ChatterboxLocal";
import { Pocket } from "@/core/ui/inspector/tts/Pocket";

const decode = (s: string) => s.replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

let data: TTS_State;

beforeEach(() => {
  data = proxy(Service_TTS_Schema.parse({}));
  (diagnosticsState as unknown as { outcomes: Record<string, unknown> }).outcomes = {};
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models: { state: proxy({ states: {}, catalog: [], pending: {}, loaded: true }), load: vi.fn(async () => {}) },
      tts: { serviceState: proxy({ status: "disconnected", error: "", pending: 0, speaking: false }) },
      patchService: vi.fn(),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Pocket panel", () => {
  it("asks for the chosen model and offers the built-in voice", () => {
    const html = decode(renderToStaticMarkup(<Pocket />));
    expect(html).toContain(`data-requirement="${POCKET_DEFAULT_MODEL}"`);
    expect(html).toContain('data-label="tts.cloning_voice"');
    expect(html).toContain('data-none="undefined"');
    expect(html).toContain("tts.pocket_builtin_voice");
    expect(html).toContain("tts.pocket_english_only");
    expect(html).toContain('data-field="tts.cloning_temperature"');
  });

  it("drops the built-in hint once a voice is chosen and follows the model choice", () => {
    data.pocket.referenceVoiceId = "alice";
    data.pocket.model = "tts/pocket-tts-en";
    const html = decode(renderToStaticMarkup(<Pocket />));
    expect(html).not.toContain("tts.pocket_builtin_voice");
    expect(html).toContain('data-requirement="tts/pocket-tts-en"');
  });
});

describe("Chatterbox panel", () => {
  it("requires a voice and offers the graphics-card decoder unless the CPU is chosen", () => {
    let html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).toContain(`data-requirement="${CHATTERBOX_LOCAL_MODEL}"`);
    expect(html).toContain(`data-requirement="${CHATTERBOX_LOCAL_GPU_FLOW}"`);
    expect(html).toContain('data-none="false"');
    expect(html).toContain('data-field="tts.chatterbox_local_language"');
    expect(html).not.toContain("tts.chatterbox_local_cpu_note");
    data.chatterboxLocal.ep = "cpu";
    html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).not.toContain(`data-requirement="${CHATTERBOX_LOCAL_GPU_FLOW}"`);
    expect(html).toContain("tts.chatterbox_local_cpu_note");
  });

  it("shows where it runs and why it fell back", () => {
    const outcomes = (diagnosticsState as unknown as { outcomes: Record<string, unknown> }).outcomes;
    outcomes["tts.chatterbox"] = { component: "tts.chatterbox", requested: "cuda", active: "cuda", reason: null, at_unix_ms: 1 };
    let html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).toContain('tts.chatterbox_local_running_on{"ep":"CUDA"}');
    outcomes["tts.chatterbox"] = { component: "tts.chatterbox", requested: "dml", active: "cpu", reason: "DirectML failed", at_unix_ms: 2 };
    html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).toContain('tts.chatterbox_local_fell_back{"ep":"CPU","reason":"DirectML failed"}');
    expect(html).toContain("tts.chatterbox_local_cpu_note");
  });

  it("offers a sample line in the chosen language", () => {
    data.chatterboxLocal.language = "ko";
    let html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).toContain(`placeholder="${CHATTERBOX_PREVIEW_TEXT.ko}"`);
    data.chatterboxLocal.language = "sw";
    html = decode(renderToStaticMarkup(<ChatterboxLocal />));
    expect(html).toContain(`placeholder="${CHATTERBOX_PREVIEW_TEXT.sw}"`);
  });
});
