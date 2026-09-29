/**
 * End-to-end check of the VOICEVOX backend against a local stand-in for AivisSpeech Engine (same
 * HTTP API as VOICEVOX), using the real `fetch` over loopback. Nothing leaves 127.0.0.1.
 */
import i18n from "i18next";
import { createServer, IncomingMessage, Server, ServerResponse } from "node:http";
import { AddressInfo } from "node:net";
import { proxy } from "valtio";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { Service_TTS_Schema, TTS_State } from "../schema";
import { TTS_VoicevoxService, fetchVoicevoxSpeakers } from "../services/voicevox";
import { applyVoicevoxPreset } from "../voicevoxPresets";

type Received = { method: string; url: string; body: string };

const AIVIS_STYLE_ID = 888753760;
const WAV = new Uint8Array([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4]);

let server: Server;
let base: string;
const received: Received[] = [];

function handle(req: IncomingMessage, res: ServerResponse) {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    received.push({ method: req.method ?? "", url: req.url ?? "", body });
    const url = new URL(req.url ?? "/", "http://x");
    if (req.method === "GET" && url.pathname === "/version") {
      res.setHeader("content-type", "application/json");
      res.end('"1.1.0"');
    } else if (req.method === "GET" && url.pathname === "/speakers") {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify([{ name: "まい", speaker_uuid: "e756b8e4", styles: [{ name: "ノーマル", id: AIVIS_STYLE_ID }] }]));
    } else if (req.method === "POST" && url.pathname === "/audio_query") {
      if (url.searchParams.get("speaker") !== String(AIVIS_STYLE_ID)) {
        res.statusCode = 422;
        res.end("{}");
        return;
      }
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ accent_phrases: [], speedScale: 1, pitchScale: 0.02, intonationScale: 1, volumeScale: 1, kana: "" }));
    } else if (req.method === "POST" && url.pathname === "/synthesis") {
      res.setHeader("content-type", "audio/wav");
      res.end(Buffer.from(WAV));
    } else {
      res.statusCode = 404;
      res.end();
    }
  });
}

let data: TTS_State;
let enqueueVoiceClip: ReturnType<typeof vi.fn>;

beforeAll(async () => {
  await i18n.init({ lng: "en", resources: { en: { translation: {} } } });
  server = createServer(handle);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

afterEach(() => {
  vi.unstubAllGlobals();
  received.length = 0;
});

function install() {
  data = proxy(Service_TTS_Schema.parse({}));
  enqueueVoiceClip = vi.fn(async () => {});
  vi.stubGlobal("window", {
    ApiServer: {
      state: { services: { tts: { data } }, audioOutputDevice: "" },
      sound: { enqueueVoiceClip, clearVoiceClips: vi.fn() },
    },
  });
}

async function waitFor(check: () => boolean, ms = 2_000) {
  const until = Date.now() + ms;
  while (!check()) {
    if (Date.now() > until) throw new Error("timed out");
    await new Promise((r) => setTimeout(r, 5));
  }
}

describe("AivisSpeech preset against a local engine", () => {
  it("connects, lists speakers and synthesizes with pitch reset to 0", async () => {
    install();
    // The user picks the AivisSpeech preset (pitch was moved earlier), then points it at the test port.
    data.voicevox.pitchScale = "0.1";
    Object.assign(data.voicevox, applyVoicevoxPreset("aivisspeech"));
    expect(data.voicevox.pitchScale).toBe("0");
    data.voicevox.host = `${base}/`;

    const speakers = await fetchVoicevoxSpeakers(data.voicevox.host);
    data.voicevox.speaker = String(speakers[0].styles[0].id);

    const receiver = { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() };
    const service = new TTS_VoicevoxService(receiver);
    await service.start(data);
    expect(receiver.onStop).not.toHaveBeenCalled();
    expect(receiver.onStart).toHaveBeenCalledTimes(1);

    service.play("こんにちは & ようこそ");
    await waitFor(() => enqueueVoiceClip.mock.calls.length === 1);

    const query = received.find((r) => r.url.startsWith("/audio_query"));
    expect(query?.url).toBe(`/audio_query?text=${encodeURIComponent("こんにちは & ようこそ")}&speaker=${AIVIS_STYLE_ID}`);
    const synthesis = received.find((r) => r.url.startsWith("/synthesis"));
    expect(synthesis?.url).toBe(`/synthesis?speaker=${AIVIS_STYLE_ID}`);
    expect(JSON.parse(synthesis!.body)).toMatchObject({ pitchScale: 0, intonationScale: 1, speedScale: 1, volumeScale: 1 });

    const [audio, options] = enqueueVoiceClip.mock.calls[0];
    expect(new Uint8Array(audio as ArrayBuffer)).toEqual(WAV);
    expect(options).toEqual({ volume: 1, rate: 1, device_name: "" });
    expect(receiver.onError).not.toHaveBeenCalled();

    service.stop();
    expect(receiver.onStop).toHaveBeenCalledTimes(1);
  });

  it("reports a closed port as 'not running'", async () => {
    install();
    // Grab a free port, then close it so nothing listens there.
    const probe = createServer();
    await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve));
    const port = (probe.address() as AddressInfo).port;
    await new Promise<void>((resolve) => probe.close(() => resolve()));

    data.voicevox.host = `http://127.0.0.1:${port}`;
    const receiver = { onStart: vi.fn(), onStop: vi.fn(), onError: vi.fn() };
    await new TTS_VoicevoxService(receiver).start(data);
    expect(receiver.onStart).not.toHaveBeenCalled();
    expect(receiver.onStop).toHaveBeenCalledWith("tts.voicevox_not_running");
  });
});
