/**
 * The one-time prompt that offers the built-in engine to a desktop profile on native Web Speech, Chrome or Edge:
 * its wording per engine, the numbers it quotes (one constant, checked against the benchmark table), and what each
 * answer does. Real English strings plus the staged ones (`sttTestI18n`).
 */
import fs from "node:fs";
import path from "node:path";
import i18n from "i18next";
import { proxy } from "valtio";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(async () => ""), isTauri: () => false }));
vi.mock("react-toastify", () => ({
  toast: Object.assign(vi.fn(), {
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    success: vi.fn(),
    update: vi.fn(),
    dismiss: vi.fn(),
  }),
}));
vi.mock("@/core/services/systemLog", () => ({ pushSystemLog: vi.fn() }));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));
vi.mock("@/utils/rustAudioInput", () => ({ migrateLegacyNativeInputDevice: vi.fn(async () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ ask: vi.fn(async () => false), confirm: vi.fn(async () => false), message: vi.fn(async () => "Cancel") }));
vi.mock("@/shared/services/pubsub", () => ({ PUBSUB_STT_EXTERNAL_COMMAND: "stt.external.command" }));
// Real engines pull in SDKs and Tauri APIs; the service is tested against a fake engine.
vi.mock("../services/azure", () => ({ STT_AzureService: class {} }));
vi.mock("../services/deepgram", () => ({ STT_DeepgramService: class {} }));
vi.mock("../services/native", () => ({ STT_NativeService: class {} }));
vi.mock("../services/whisper", () => ({ STT_WhisperService: class {} }));
vi.mock("../services/vosk", () => ({ STT_VoskService: class {} }));
vi.mock("../services/moonshine", () => ({ STT_MoonshineService: class {} }));
vi.mock("../services/openaiAudio", () => ({ STT_OpenAI_AudioService: class {} }));
vi.mock("../services/externalMic", () => ({ STT_ExternalMicService: class {} }));

import { ask } from "@tauri-apps/plugin-dialog";
import { toast } from "react-toastify";
import Service_STT, { localBackendCarryOver, type SttBackendMap } from "../index";
import { LOCAL_OFFER_DOWNLOAD_SIZE, LOCAL_OFFER_ENGLISH_ERRORS, localOfferDialog } from "../localOffer";
import { Service_STT_Schema, STT_Backends, type STT_State } from "../schema";
import type { ISTTReceiver, ISTTService } from "../types";
import { initSttTestI18n } from "./sttTestI18n";

const t = (key: string, options?: Record<string, unknown>): string => i18n.t(key, options) as string;

beforeAll(async () => {
  await initSttTestI18n();
});

describe("the prompt's numbers", () => {
  /** The `Finals, clean` cell of the benchmark table row that starts with `engine`. */
  function finalsClean(engine: string): number {
    const doc = fs.readFileSync(path.resolve(__dirname, "../../../../../docs/benchmarks/SPEECH_BENCHMARKS.md"), "utf8");
    const section = doc.split(/^### /m).find((s) => s.startsWith("Browser speech recognition (Chrome and Edge Web Speech, the old default), English"));
    expect(section, "the benchmark section the prompt cites").toBeDefined();
    const row = section!.split("\n").find((line) => line.startsWith(`| ${engine}`));
    expect(row, `a table row for ${engine}`).toBeDefined();
    const cells = row!.split("|").map((c) => c.trim());
    return Number.parseFloat(cells[3]); // | engine | screening | finals clean | ...
  }

  it("are the rounded Finals, clean figures of the English Web Speech section of SPEECH_BENCHMARKS.md", () => {
    expect(finalsClean("Parakeet Ultra")).toBe(6.1);
    expect(finalsClean("Chrome Web Speech")).toBe(20.8);
    expect(finalsClean("Edge Web Speech")).toBe(9.8);
    expect(LOCAL_OFFER_ENGLISH_ERRORS.ours).toBe(Math.round(finalsClean("Parakeet Ultra")));
    expect(LOCAL_OFFER_ENGLISH_ERRORS.chrome).toBe(Math.round(finalsClean("Chrome Web Speech")));
    expect(LOCAL_OFFER_ENGLISH_ERRORS.edge).toBe(Math.round(finalsClean("Edge Web Speech")));
  });
});

describe("localOfferDialog", () => {
  const DOWNLOAD = `The first time you press Start, Sigil asks before downloading the speech models (about ${LOCAL_OFFER_DOWNLOAD_SIZE}).`;
  // Native and other profiles keep their microphone and language (localBackendCarryOver); Chrome and Edge keep theirs in the browser page.
  const LAST_TWO = `Your microphone and language stay the same. ${DOWNLOAD}`;
  const LAST_TWO_BROWSER = `Sigil starts with your system's default microphone and detects your language automatically; you can change both in Speech to Text. ${DOWNLOAD}`;

  it("says exactly what the user asked for when the profile is on Chrome", () => {
    expect(localOfferDialog("chrome", t)).toEqual({
      title: "Switch to Sigil's built-in speech recognition?",
      message:
        "You're using Chrome's speech recognition, which sends your voice to Google. Sigil now has its own speech recognition that runs on your PC. " +
        "In our English test it made about 6 mistakes per 100 words, versus about 21 for Chrome. " +
        "It also works without internet and keeps your voice on your PC. " +
        "Sigil starts with your system's default microphone and detects your language automatically; you can change both in Speech to Text. The first time you press Start, Sigil asks before downloading the speech models (about 1.7 GB).",
      okLabel: "Switch to built-in",
      cancelLabel: "Keep Chrome",
    });
  });

  it("says the same for Edge with Microsoft, the Edge figure and its own Keep button", () => {
    const edge = localOfferDialog("edge", t);
    expect(edge.message).toContain("You're using Edge's speech recognition, which sends your voice to Microsoft.");
    expect(edge.message).toContain("made about 6 mistakes per 100 words, versus about 10 for Edge.");
    expect(edge.message).not.toContain("Google");
    expect(edge.message).toContain(LAST_TWO_BROWSER);
    expect(edge.okLabel).toBe("Switch to built-in");
    expect(edge.cancelLabel).toBe("Keep Edge");
  });

  it("tells a native profile that its engine does not work in the desktop app", () => {
    const native = localOfferDialog("native", t);
    expect(native.message.startsWith("Your current speech recognition doesn't work in the desktop app. Switch to Sigil's built-in speech recognition, which runs on your PC?")).toBe(true);
    expect(native.message.endsWith(LAST_TWO)).toBe(true);
    expect(native.message).not.toContain("mistakes per 100 words");
    expect([native.title, native.okLabel, native.cancelLabel]).toEqual(["Switch to Sigil's built-in speech recognition?", "Switch to built-in", "Not now"]);
  });

  it("has a plain wording for any other engine, with no claim about the current one", () => {
    const other = localOfferDialog("other", t);
    expect(other.message).toBe(`Switch to Sigil's built-in speech recognition, which runs on your PC? ${LAST_TWO}`);
    expect(other.cancelLabel).toBe("Not now");
  });

  it("leaves no raw key or placeholder in any wording", () => {
    for (const kind of ["chrome", "edge", "native", "other"] as const) {
      const d = localOfferDialog(kind, t);
      for (const text of Object.values(d)) {
        expect(text).not.toMatch(/\bstt\.[a-z_]+\b/);
        expect(text).not.toContain("{{");
      }
    }
  });
});

// ---------------------------------------------------------------------------------------------------------------
// What each answer does

class FakeEngine implements ISTTService {
  static instances: FakeEngine[] = [];
  startAbortSignal: AbortSignal | null = null;
  started = 0;
  stopped = 0;
  constructor(public receiver: ISTTReceiver) {
    FakeEngine.instances.push(this);
  }
  start(_params: STT_State) {
    this.started++;
  }
  stop() {
    this.stopped++;
  }
  dispose() {}
}

const fakeMap = Object.fromEntries(Object.values(STT_Backends).map((b) => [b, FakeEngine])) as SttBackendMap;
const defaults = Service_STT_Schema.parse({});
let isApp = true;

function makeService(overrides: Partial<STT_State> = {}, root: Record<string, unknown> = {}) {
  const data = proxy({ ...defaults, ...overrides });
  const state = { services: { stt: { data } }, onboardingComplete: true, ...root };
  vi.stubGlobal("window", {
    Config: { isApp: () => isApp },
    ApiServer: {
      state,
      patchService: (key: "stt", fn: (s: { data: STT_State }) => void) => fn(state.services[key]),
    },
    ApiShared: {
      pubsub: {
        publishText: vi.fn(),
        subscribe: vi.fn(() => "stream-sub"),
        unsubscribe: vi.fn(),
        broadcastLocalWebSocketOnly: vi.fn(),
      },
    },
  });
  const service = new Service_STT(fakeMap);
  service.updateReplacementsCache();
  return { service, data };
}

const flush = () => vi.advanceTimersByTimeAsync(0);
const asked = () => vi.mocked(ask).mock.calls[0] as [string, { title: string; okLabel: string; cancelLabel: string }];

describe("Service_STT offer of the built-in engine", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeEngine.instances = [];
    isApp = true;
    vi.stubGlobal("navigator", { hardwareConcurrency: 12 });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  for (const [backend, browser, keep] of [
    [STT_Backends.chrome, "Chrome", "Keep Chrome"],
    [STT_Backends.edge, "Edge", "Keep Edge"],
  ] as const) {
    describe(`a profile on ${browser}`, () => {
      // Profiles from earlier builds already ran the migration once, which only recorded a working engine.
      const profile = (extra: Partial<STT_State> = {}) => ({ backend, desktopDefaultMigrated: true, autoStart: true, ...extra });

      it(`asks once with the ${browser} wording and does not auto-start meanwhile`, async () => {
        const { service, data } = makeService(profile());
        await service.init();
        await flush();
        // Pending until the startup offers run (after i18n); nothing starts and nothing changes meanwhile.
        expect(data.backend).toBe(backend);
        expect(FakeEngine.instances).toHaveLength(0);
        expect(ask).not.toHaveBeenCalled();

        vi.mocked(ask).mockResolvedValueOnce(false);
        await service.runStartupOffer();
        await flush();
        expect(ask).toHaveBeenCalledTimes(1);
        const [message, options] = asked();
        expect(message).toContain(`You're using ${browser}'s speech recognition`);
        expect(message).toContain(browser === "Chrome" ? "versus about 21 for Chrome" : "versus about 10 for Edge");
        expect(options).toMatchObject({ title: "Switch to Sigil's built-in speech recognition?", okLabel: "Switch to built-in", cancelLabel: keep });
      });

      it(`"${keep}" keeps the engine, records the answer, and never asks again`, async () => {
        const { service, data } = makeService(profile());
        await service.init();
        vi.mocked(ask).mockResolvedValueOnce(false);
        await service.runStartupOffer();
        await flush();

        expect(data.backend).toBe(backend);
        expect(data.localOfferShown).toBe(true);
        expect(data.desktopDefaultMigrated).toBe(true);
        expect(toast.info).not.toHaveBeenCalled(); // nothing was migrated away from the engine that works

        // The auto-start that waited for the answer happens now: captions are on for this launch too.
        expect(FakeEngine.instances).toHaveLength(1);

        // Later launches: no dialog, and auto-start works as before.
        await new Service_STT(fakeMap).init();
        await flush();
        expect(ask).toHaveBeenCalledTimes(1);
        expect(FakeEngine.instances).toHaveLength(2);
      });

      it(`"${keep}" starts nothing when auto-start is off`, async () => {
        const { service, data } = makeService(profile({ autoStart: false }));
        await service.init();
        vi.mocked(ask).mockResolvedValueOnce(false);
        await service.runStartupOffer();
        await flush();
        expect(data.backend).toBe(backend);
        expect(FakeEngine.instances).toHaveLength(0);
      });

      it("a dialog that fails to open keeps the engine and still starts it", async () => {
        const { service, data } = makeService(profile());
        await service.init();
        vi.mocked(ask).mockRejectedValueOnce(new Error("no dialog"));
        const logged = vi.spyOn(console, "error").mockImplementation(() => {});
        await service.runStartupOffer();
        await flush();
        expect(logged).toHaveBeenCalled();
        expect(data.backend).toBe(backend);
        expect(FakeEngine.instances).toHaveLength(1);
      });

      it("accepting switches to the built-in engine, nothing starts or downloads", async () => {
        const { service, data } = makeService(profile());
        await service.init();
        vi.mocked(ask).mockResolvedValueOnce(true);
        await service.runStartupOffer();
        await flush();

        expect(data.backend).toBe(STT_Backends.local);
        expect(data.localOfferShown).toBe(true);
        expect(data.desktopDefaultMigrated).toBe(true);
        expect(FakeEngine.instances).toHaveLength(0); // never starts by itself: the first Start asks before downloading
        // The language and microphone of the browser page live in the browser, not in the host: the built-in
        // engine keeps its own (automatic language, system default microphone).
        expect(data.local.language).toBe("auto");
        expect(data.local.device).toBe("");
      });

      it("does not ask a profile that already answered", async () => {
        const { service, data } = makeService(profile({ localOfferShown: true }));
        await service.init();
        await flush();
        await service.runStartupOffer();
        expect(ask).not.toHaveBeenCalled();
        expect(data.backend).toBe(backend);
        expect(FakeEngine.instances).toHaveLength(1); // auto-start as usual
      });

      it("does not ask a new install, nor the web build", async () => {
        const fresh = makeService(profile(), { onboardingComplete: false });
        await fresh.service.init();
        await fresh.service.runStartupOffer();
        expect(ask).not.toHaveBeenCalled();
        expect(fresh.data.backend).toBe(backend);

        isApp = false;
        const web = makeService(profile());
        await web.service.init();
        await web.service.runStartupOffer();
        expect(ask).not.toHaveBeenCalled();
        expect(web.data.backend).toBe(backend);
      });
    });
  }

  describe("a profile that chose the built-in engine and later picked Chrome or Edge by hand", () => {
    for (const backend of [STT_Backends.chrome, STT_Backends.edge]) {
      it(`is not asked at the next launch when it was born on the built-in engine (a new install) and moved to ${backend}`, async () => {
        // First launch of a new install: the migration puts it on local; the wizard finishes and the user picks the browser engine.
        const { service, data } = makeService({}, { onboardingComplete: false });
        await service.init();
        await flush();
        expect(data.backend).toBe(STT_Backends.local);
        expect(data.localOfferShown).toBe(true);
        data.backend = backend;
        await flush();

        // Next launch: onboarding is complete, so the profile is not "fresh" any more.
        const next = makeService({ ...JSON.parse(JSON.stringify(data)), autoStart: false });
        await next.service.init();
        await next.service.runStartupOffer();
        expect(ask).not.toHaveBeenCalled();
        expect(next.data.backend).toBe(backend);
      });

      it(`is not asked when it goes to the built-in engine and then to ${backend} in one session`, async () => {
        const { service, data } = makeService({ backend: STT_Backends.whisper, desktopDefaultMigrated: true });
        await service.init();
        expect(data.localOfferShown).toBe(false); // on Whisper it has not chosen the built-in engine yet
        data.backend = STT_Backends.local; // the engine list (or the wizard, or the offer) puts it there
        await flush();
        expect(data.localOfferShown).toBe(true);
        data.backend = backend;
        await flush();

        const next = makeService({ ...JSON.parse(JSON.stringify(data)), autoStart: false });
        await next.service.init();
        await next.service.runStartupOffer();
        expect(ask).not.toHaveBeenCalled();
      });
    }

    it("a profile that never chose it is still offered it", async () => {
      const { service, data } = makeService({ backend: STT_Backends.chrome, desktopDefaultMigrated: true });
      await service.init();
      await flush();
      expect(data.localOfferShown).toBe(false);
      vi.mocked(ask).mockResolvedValueOnce(false);
      await service.runStartupOffer();
      expect(ask).toHaveBeenCalledTimes(1);
    });
  });

  describe("a profile on native Web Speech", () => {
    it("asks with the native wording; Not now moves it to Whisper as before", async () => {
      const { service, data } = makeService({ autoStart: true });
      await service.init();
      await flush();
      vi.mocked(ask).mockResolvedValueOnce(false);
      await service.runStartupOffer();
      await flush();

      const [message, options] = asked();
      expect(message.startsWith("Your current speech recognition doesn't work in the desktop app.")).toBe(true);
      expect(options).toMatchObject({ okLabel: "Switch to built-in", cancelLabel: "Not now" });
      expect(data.backend).toBe(STT_Backends.whisper);
      expect(data.localOfferShown).toBe(true);
      expect(data.desktopDefaultMigrated).toBe(true);
    });

    it("Switch to built-in carries the language over", async () => {
      const { service, data } = makeService({ native: { ...defaults.native, language: "ja-JP" } });
      await service.init();
      vi.mocked(ask).mockResolvedValueOnce(true);
      await service.runStartupOffer();
      expect(data.backend).toBe(STT_Backends.local);
      expect(data.local.language).toBe("ja");
    });

    it("an offer already shown goes straight to Whisper without a dialog", async () => {
      const { service, data } = makeService({ localOfferShown: true });
      await service.init();
      await service.runStartupOffer();
      expect(ask).not.toHaveBeenCalled();
      expect(data.backend).toBe(STT_Backends.whisper);
    });
  });

  it("carries over what Sigil's settings hold: nothing from the Chrome / Edge page, the language from Web Speech", () => {
    // The microphone page keeps its language and microphone in the browser, not in the host's settings.
    for (const backend of [STT_Backends.chrome, STT_Backends.edge])
      expect(localBackendCarryOver({ ...defaults, backend, native: { ...defaults.native, language: "ja-JP" } })).toEqual({ language: null, device: null });
    expect(localBackendCarryOver({ ...defaults, backend: STT_Backends.native, native: { ...defaults.native, language: "ja-JP" } })).toEqual({ language: "ja", device: null });
  });

  it("a profile on another engine is never asked", async () => {
    for (const backend of [STT_Backends.whisper, STT_Backends.vosk, STT_Backends.azure, STT_Backends.local]) {
      const { service } = makeService({ backend, desktopDefaultMigrated: true });
      await service.init();
      await service.runStartupOffer();
    }
    expect(ask).not.toHaveBeenCalled();
  });
});
