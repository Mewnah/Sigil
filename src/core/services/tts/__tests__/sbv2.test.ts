import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: () => false }));

import { invoke } from "@tauri-apps/api/core";
import type { RustCatalogEntry, RustModelEntryStatus } from "@/types";
import { Service_TTS_Schema, TTS_State } from "../schema";
import {
  SBV2_BERT_ID,
  SBV2_DICTIONARY_ID,
  SBV2_REQUIREMENTS,
  TTS_Sbv2Service,
  importSbv2Model,
  isNeutralStyle,
  pickSpeaker,
  pickStyle,
  sbv2Library,
  sbv2Speakers,
  sbv2SpeakSettings,
  sbv2StartConfig,
  stopSbv2Engine,
  type Sbv2Model,
} from "../services/sbv2";
import type { ITTSReceiver, TtsSinkContext } from "../types";

// P6-G: the sbv2 backend (Style-Bert-VITS2 / AivisSpeech voices on Sigil's own engine, Rust tts-helpers `sbv2`).

const invokeMock = vi.mocked(invoke);

function receiver() {
  return { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() } satisfies ITTSReceiver;
}

const flush = async (n = 30) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};

function entry(id: string, title = id): RustCatalogEntry {
  return {
    id,
    family: id.split("/")[0],
    kind: "model",
    version: "1",
    title,
    license: "MIT",
    status: "published",
    requires: [],
    files: [{ path: "f", size: 1000, sha256: "0".repeat(64), sources: [] }],
  } as unknown as RustCatalogEntry;
}

function status(id: string, state: RustModelEntryStatus["state"]): RustModelEntryStatus {
  return { id, state, installed_version: state === "installed" ? "1" : null, installed_path: null, bytes_on_disk: 0, adopted: false, error: null };
}

const CATALOG = [entry(SBV2_BERT_ID, "Japanese BERT"), entry(SBV2_DICTIONARY_ID, "Japanese dictionary")];

const MODEL: Sbv2Model = {
  id: "mao-abc123",
  name: "まお",
  format: "aivmx",
  speakers: ["まお", "Second"],
  styles: ["ノーマル", "あまあま"],
  voices: [
    { name: "まお", id: 0, styles: [{ name: "ノーマル", id: 0 }, { name: "あまあま", id: 1 }], languages: ["ja"] },
    { name: "Second", id: 1, styles: [{ name: "ノーマル", id: 0 }], languages: ["ja"] },
  ],
};

let data: TTS_State;
let models: { load: ReturnType<typeof vi.fn>; state: { catalog: RustCatalogEntry[]; states: Record<string, RustModelEntryStatus> } };

function installed() {
  models.state.states = Object.fromEntries(SBV2_REQUIREMENTS.map((id) => [id, status(id, "installed")]));
}

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } }, interpolation: { escapeValue: false } });
});

beforeEach(async () => {
  invokeMock.mockReset();
  invokeMock.mockResolvedValue(undefined);
  data = proxy(Service_TTS_Schema.parse({ sbv2: { model: "mao-abc123" } }));
  models = { load: vi.fn(async () => {}), state: { catalog: CATALOG, states: {} } };
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      models,
      tts: { serviceState: { status: "disconnected" } },
    },
  });
  // Let any lifecycle step of a previous test settle.
  await stopSbv2Engine();
  invokeMock.mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function calls(cmd: string) {
  return invokeMock.mock.calls.filter(([c]) => c === cmd).map(([, args]) => args as Record<string, unknown>);
}

describe("sbv2 settings", () => {
  it("defaults off and keeps saved values safe", () => {
    const d = Service_TTS_Schema.parse({}).sbv2;
    expect(d).toMatchObject({ model: "", speaker: "", style: "Neutral", styleWeight: "1", speed: "1", sdpRatio: "0.2", executionProvider: "cpu", threads: "auto" });
    const bad = Service_TTS_Schema.parse({ sbv2: { executionProvider: "tpu", threads: "99", sdpRatio: "x" } }).sbv2;
    expect(bad.executionProvider).toBe("cpu");
    expect(bad.threads).toBe("auto");
  });

  it("maps the slice onto the start config and the sentence request", () => {
    const c = Service_TTS_Schema.parse({
      sbv2: { model: " mao-abc123 ", executionProvider: "dml", threads: "4", speaker: " Second ", style: "あまあま", styleWeight: "50", speed: "0.1", sdpRatio: "0.5", noise: "abc" },
    }).sbv2;
    expect(sbv2StartConfig(c)).toEqual({ model: "mao-abc123", ep: "dml", threads: 4 });
    expect(sbv2StartConfig({ ...c, threads: "auto" }).threads).toBeNull();
    expect(sbv2SpeakSettings(c)).toEqual({
      model: "mao-abc123",
      speaker: "Second",
      style: "あまあま",
      styleWeight: 20,
      sdpRatio: 0.5,
      noise: 0.6,
      noiseW: 0.8,
      speed: 0.5,
    });
  });

  it("picks speakers and styles by name, falling back to the first speaker and the neutral style", () => {
    expect(pickSpeaker(MODEL, "Second")?.id).toBe(1);
    expect(pickSpeaker(MODEL, "nobody")?.name).toBe("まお");
    expect(pickSpeaker(null, "x")).toBeNull();
    const mao = pickSpeaker(MODEL, "")!;
    expect(pickStyle(mao, "あまあま")?.id).toBe(1);
    expect(pickStyle(mao, "Neutral")?.id).toBe(0);
    expect(isNeutralStyle(pickStyle(mao, "Neutral"))).toBe(true);
    expect(isNeutralStyle(pickStyle(mao, "あまあま"))).toBe(false);
    // A row without the detailed list (the scaffold's shape) still lists speakers and styles.
    const flat = sbv2Speakers({ ...MODEL, voices: undefined });
    expect(flat.map((s) => s.name)).toEqual(["まお", "Second"]);
    expect(flat[0].styles.map((s) => s.id)).toEqual([0, 1]);
  });

  it("keeps the catalog ids in step with Rust and the catalog fragment", () => {
    const rust = readFileSync(resolve(__dirname, "../../../../../src-tauri/src/services/tts_helpers/sbv2.rs"), "utf8");
    expect(rust).toContain(`pub const BERT_MODEL_ID: &str = "${SBV2_BERT_ID}";`);
    expect(rust).toContain(`pub const DICTIONARY_ID: &str = "${SBV2_DICTIONARY_ID}";`);
    const fragment = JSON.parse(readFileSync(resolve(__dirname, "../../../../../src-tauri/model-catalog/runtime-sbv2.json"), "utf8")) as {
      entries: { id: string }[];
    };
    expect(fragment.entries.map((e) => e.id).sort()).toEqual([...SBV2_REQUIREMENTS].sort());
  });
});

describe("sbv2 voice import", () => {
  it("runs a second drop after the first and stays busy until the last one ends", async () => {
    const pending: { path: string; resolve: (m: Sbv2Model) => void; reject: (e: unknown) => void }[] = [];
    invokeMock.mockImplementation(async (cmd, args) => {
      if (cmd === "plugin:tts-helpers|sbv2_import_model") {
        return new Promise<Sbv2Model>((resolve, reject) => pending.push({ path: String((args as Record<string, unknown>)?.path), resolve, reject }));
      }
      if (cmd === "plugin:tts-helpers|sbv2_list_models") return [MODEL];
      return undefined;
    });
    const a = importSbv2Model("C:/voices/a.aivmx");
    const b = importSbv2Model("C:/voices/b.aivmx");
    await flush();
    expect(sbv2Library.importing).toBe(true);
    expect(pending.map((p) => p.path)).toEqual(["C:/voices/a.aivmx"]);
    // The first fails: the second still runs, and the panel stays busy meanwhile.
    pending[0].reject("broken graph");
    await expect(a).rejects.toBe("broken graph");
    await flush();
    expect(pending.map((p) => p.path)).toEqual(["C:/voices/a.aivmx", "C:/voices/b.aivmx"]);
    expect(sbv2Library.importing).toBe(true);
    pending[1].resolve(MODEL);
    await expect(b).resolves.toEqual(MODEL);
    await flush();
    expect(sbv2Library.importing).toBe(false);
    expect(sbv2Library.models).toEqual([MODEL]);
  });
});

describe("TTS_Sbv2Service", () => {
  it("refuses to start without a voice or before the downloads", async () => {
    data.sbv2.model = "";
    const r = receiver();
    await new TTS_Sbv2Service(r).start(data);
    expect(r.onStop).toHaveBeenCalledWith("tts.sbv2_no_voice");

    data.sbv2.model = "mao-abc123";
    const r2 = receiver();
    await new TTS_Sbv2Service(r2).start(data);
    expect(r2.onStop).toHaveBeenCalledWith("tts.sbv2_not_downloaded");
    expect(calls("plugin:tts-helpers|tts_helper_start")).toHaveLength(0);
  });

  it("starts the engine with the voice and speaks each sentence into the sink", async () => {
    installed();
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:tts-helpers|tts_helper_start") return { engine: "sbv2", baseUrl: "", apiKey: "", model: "mao-abc123", pcmSampleRate: 44100 };
      if (cmd === "plugin:tts-helpers|sbv2_speak") return { sampleRate: 44100, frames: 100, firstAudioMs: 5, totalMs: 9 };
      return undefined;
    });
    const r = receiver();
    const svc = new TTS_Sbv2Service(r);
    await svc.start(data);
    expect(r.onStart).toHaveBeenCalled();
    expect(calls("plugin:tts-helpers|tts_helper_start")).toEqual([{ engine: "sbv2", config: { model: "mao-abc123", ep: "cpu", threads: null } }]);

    // Style changes apply to the next sentence.
    data.sbv2.style = "あまあま";
    const sink: TtsSinkContext = { utteranceId: "u1", target: { route: "device", device: "", volume: 1 } };
    await svc.synthesizeToSink("こんにちは。", sink, new AbortController().signal);
    const speak = calls("plugin:tts-helpers|sbv2_speak");
    expect(speak).toHaveLength(1);
    expect(speak[0]).toMatchObject({ utteranceId: "u1", request: { model: "mao-abc123", text: "こんにちは。", style: "あまあま" } });
  });

  it("stop / skip cancel the sentence without an error; other failures are named", async () => {
    installed();
    const ctl = new AbortController();
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:tts-helpers|sbv2_speak") {
        ctl.abort();
        throw "cancelled";
      }
      return undefined;
    });
    const svc = new TTS_Sbv2Service(receiver());
    await svc.start(data);
    const sink: TtsSinkContext = { utteranceId: "u2", target: { route: "device", device: "", volume: 1 } };
    await expect(svc.synthesizeToSink("はい。", sink, ctl.signal)).resolves.toBeUndefined();
    expect(calls("plugin:tts-helpers|sbv2_cancel")).toContainEqual({ utteranceId: "u2" });

    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:tts-helpers|sbv2_speak") throw "BERT inference failed";
      return undefined;
    });
    await expect(svc.synthesizeToSink("はい。", { ...sink, utteranceId: "u3" }, new AbortController().signal)).rejects.toThrow(
      "Style-Bert-VITS2: BERT inference failed",
    );
  });

  it("serializes stop before the next start, so a restart never cancels its own load", async () => {
    installed();
    const order: string[] = [];
    let releaseStop: () => void = () => {};
    invokeMock.mockImplementation(async (cmd) => {
      order.push(cmd as string);
      if (cmd === "plugin:tts-helpers|tts_helper_stop") await new Promise<void>((r) => (releaseStop = r));
      return undefined;
    });
    const first = new TTS_Sbv2Service(receiver());
    await first.start(data);
    first.stop();
    const r = receiver();
    const second = new TTS_Sbv2Service(r);
    const started = second.start(data);
    await flush();
    // The new start waits for the stop to finish.
    expect(order.filter((c) => c === "plugin:tts-helpers|tts_helper_start")).toHaveLength(1);
    releaseStop();
    await started;
    const tail = order.slice(order.indexOf("plugin:tts-helpers|sbv2_cancel"));
    expect(tail).toEqual(["plugin:tts-helpers|sbv2_cancel", "plugin:tts-helpers|tts_helper_stop", "plugin:tts-helpers|tts_helper_start"]);
    expect(r.onStart).toHaveBeenCalled();
  });

  it("a stop during the start wins", async () => {
    installed();
    let finish: () => void = () => {};
    invokeMock.mockImplementation(async (cmd) => {
      if (cmd === "plugin:tts-helpers|tts_helper_start") await new Promise<void>((r) => (finish = r));
      return undefined;
    });
    const r = receiver();
    const svc = new TTS_Sbv2Service(r);
    const started = svc.start(data);
    await flush();
    svc.stop();
    finish();
    await started;
    expect(r.onStart).not.toHaveBeenCalled();
    expect(r.onStop).toHaveBeenCalledTimes(1);
    expect(r.onStop).toHaveBeenCalledWith();
  });
});
