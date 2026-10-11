import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageOffer, RESULT_VISIBLE_MS, ROUTE_CHECK_TIMEOUT_MS, type LanguageOfferDeps } from "../languageOffer";
import { KEEP_SNOOZE_MS, STREAK_FINALS, type LockOutcome } from "../languageSuggestion";
import { Service_STT_Schema, STT_Backends, type STT_State } from "../schema";
import type { SttTextMeta } from "../types";

const FRENCH = "Je voudrais essayer ce nouveau micro ce soir";
const ENGLISH = "I would like to try this new microphone tonight";
const START = 1_800_000_000_000;

interface Rig {
  offer: LanguageOffer;
  data: STT_State;
  restart: ReturnType<typeof vi.fn>;
  log: ReturnType<typeof vi.fn>;
  routeReady: ReturnType<typeof vi.fn>;
  state: { running: boolean; now: number };
}

function rig(overrides: Partial<STT_State> = {}, routeReady: boolean | null | Error = true): Rig {
  const data = proxy<STT_State>({ ...Service_STT_Schema.parse({}), backend: STT_Backends.local, ...overrides });
  const state = { running: true, now: START };
  const restart = vi.fn();
  const log = vi.fn();
  const ready = vi.fn(async (_language: string) => {
    if (routeReady instanceof Error) throw routeReady;
    return routeReady;
  });
  const deps: LanguageOfferDeps = {
    data: () => data,
    patch: (fn) => fn(data),
    now: () => state.now,
    running: () => state.running,
    routeReady: ready,
    restart,
    log,
  };
  return { offer: new LanguageOffer(deps), data, restart, log, routeReady: ready, state };
}

let segment = 0;
const meta = (lang: string | undefined, extra: Partial<SttTextMeta> = {}): SttTextMeta => ({
  segmentId: `local-x:1:${++segment}`,
  revision: 3,
  ...(lang ? { lang } : {}),
  ...extra,
});

/** Feeds `count` finals in `lang`. */
function hear(r: Rig, lang: string, count: number, text = FRENCH): void {
  for (let i = 0; i < count; i++) r.offer.noteFinal(text, meta(lang));
}

beforeEach(() => {
  vi.useFakeTimers();
  segment = 0;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("LanguageOffer: when it asks", () => {
  it("offers the detected language after the third consistent final of the session", () => {
    const r = rig();
    hear(r, "fr", STREAK_FINALS - 1);
    expect(r.offer.view.language).toBeNull();
    hear(r, "fr", 1);
    expect(r.offer.view).toMatchObject({ language: "fr", source: "detected", busy: false, result: null });
  });

  it("does not offer on a mixed session, and never sets the language by itself", () => {
    const r = rig();
    r.offer.noteFinal(FRENCH, meta("fr"));
    r.offer.noteFinal(ENGLISH, meta("en"));
    r.offer.noteFinal(FRENCH, meta("fr"));
    r.offer.noteFinal(ENGLISH, meta("en"));
    expect(r.offer.view.language).toBeNull();
    hear(r, "fr", 5);
    expect(r.offer.view.language).toBe("fr");
    expect(r.data.local.language).toBe("auto");
    expect(r.data.local.languageDecision.outcome).toBe("none");
    expect(r.restart).not.toHaveBeenCalled();
  });

  it("ignores short finals, finals without a language and finals of another backend or a set language", () => {
    const r = rig();
    for (let i = 0; i < 6; i++) r.offer.noteFinal("yeah", meta("fr"));
    for (let i = 0; i < 6; i++) r.offer.noteFinal(FRENCH, meta(undefined));
    r.offer.noteFinal(FRENCH);
    expect(r.offer.view.language).toBeNull();

    const set = rig({ local: { ...Service_STT_Schema.parse({}).local, language: "fr" } });
    hear(set, "fr", 6);
    expect(set.offer.view.language).toBeNull();

    const other = rig({ backend: STT_Backends.whisper });
    hear(other, "fr", 6);
    expect(other.offer.view.language).toBeNull();
  });

  it("offers one suggestion per session; a new session counts from zero and may offer again", () => {
    const r = rig();
    hear(r, "fr", 3);
    expect(r.offer.view.language).toBe("fr");
    r.offer.keepAutomatic();
    hear(r, "fr", 6);
    expect(r.offer.view.language).toBeNull();
    // Kept: silent for a week, even in a new session.
    r.offer.sessionStarted();
    hear(r, "fr", 3);
    expect(r.offer.view.language).toBeNull();
    // A week later it asks again.
    r.state.now += KEEP_SNOOZE_MS;
    r.offer.sessionStarted();
    hear(r, "fr", 2);
    expect(r.offer.view.language).toBeNull();
    hear(r, "fr", 1);
    expect(r.offer.view.language).toBe("fr");
  });

  it("starts the count over with each session", () => {
    const r = rig();
    hear(r, "fr", 2);
    r.offer.sessionStarted();
    hear(r, "fr", 1);
    expect(r.offer.view.language).toBeNull();
  });

  it("does not offer a language the panel cannot lock to", () => {
    const r = rig();
    hear(r, "xx", 6);
    expect(r.offer.view.language).toBeNull();
  });
});

describe("LanguageOffer: an offer is only as old as its evidence", () => {
  it("takes an unanswered offer down when a counting final in another language arrives, and stores no answer", () => {
    const r = rig();
    hear(r, "fr", 3);
    expect(r.offer.view.language).toBe("fr");
    // A guest spoke French three times; the streamer goes on in English.
    r.offer.noteFinal(ENGLISH, meta("en"));
    expect(r.offer.view).toMatchObject({ language: null, source: null });
    expect(r.data.local.languageDecision.outcome).toBe("none");
    expect(r.data.local.language).toBe("auto");
    // One suggestion per session: English, however long it goes on, is not offered in the same session ...
    hear(r, "en", 6, ENGLISH);
    expect(r.offer.view.language).toBeNull();
    // ... and the next session counts from zero and asks about what it hears.
    r.offer.sessionStarted();
    hear(r, "en", 3, ENGLISH);
    expect(r.offer.view.language).toBe("en");
  });

  it("keeps the offer while the same language goes on, and when other finals are too short to count", () => {
    const r = rig();
    hear(r, "fr", 3);
    hear(r, "fr", 4);
    r.offer.noteFinal("yeah", meta("en"));
    r.offer.noteFinal(ENGLISH, meta(undefined));
    expect(r.offer.view.language).toBe("fr");
  });

  it("takes an unanswered offer down when the next session starts, so a click cannot lock what nobody said since", async () => {
    const r = rig();
    hear(r, "fr", 3);
    expect(r.offer.view.language).toBe("fr");
    r.offer.sessionStarted();
    expect(r.offer.view).toMatchObject({ language: null, source: null });
    expect(r.data.local.languageDecision.outcome).toBe("none");
    // The new session offers again only from its own speech.
    hear(r, "fr", 2);
    expect(r.offer.view.language).toBeNull();
    hear(r, "fr", 1);
    expect(r.offer.view.language).toBe("fr");
  });

  it("a lock's own restart does not remove the feedback line it just showed", async () => {
    const r = rig();
    hear(r, "fr", 3);
    expect(await r.offer.lock("fr")).toBe("restarted");
    r.offer.sessionStarted(); // what the restart does
    expect(r.offer.view.result).toEqual({ language: "fr", outcome: "restarted" });
    expect(r.offer.view.language).toBeNull();
  });
});

describe("LanguageOffer: the answers", () => {
  it("Keep automatic is stored with the offered language and withdraws the card", () => {
    const r = rig();
    hear(r, "fr", 3);
    r.offer.keepAutomatic();
    expect(r.offer.view.language).toBeNull();
    expect(r.data.local.languageDecision).toEqual({ outcome: "kept", language: "fr", at: START });
    expect(r.data.local.language).toBe("auto");
  });

  it("Don't ask again is stored and holds forever", () => {
    const r = rig();
    hear(r, "ru", 3, "Я хотел бы попробовать этот новый микрофон");
    expect(r.offer.view.language).toBe("ru");
    r.offer.neverAsk();
    expect(r.data.local.languageDecision).toEqual({ outcome: "never", language: "ru", at: START });
    r.state.now += 100 * KEEP_SNOOZE_MS;
    r.offer.sessionStarted();
    hear(r, "ru", 9, "Я хотел бы попробовать этот новый микрофон");
    expect(r.offer.view.language).toBeNull();
  });

  it("an answer from the panel's Windows-language hint carries its own language", () => {
    const r = rig();
    r.offer.keepAutomatic("fr-FR");
    expect(r.data.local.languageDecision).toEqual({ outcome: "kept", language: "fr", at: START });
    r.offer.neverAsk();
    expect(r.data.local.languageDecision).toEqual({ outcome: "never", language: "", at: START });
  });

  it("an offer is withdrawn when the profile leaves auto or the automatic backend by itself", () => {
    const r = rig();
    hear(r, "fr", 3);
    r.data.local.language = "de";
    r.offer.settingsChanged();
    expect(r.offer.view.language).toBeNull();

    const s = rig();
    hear(s, "fr", 3);
    s.data.backend = STT_Backends.whisper;
    s.offer.settingsChanged();
    expect(s.offer.view.language).toBeNull();

    const t = rig();
    hear(t, "fr", 3);
    t.offer.settingsChanged();
    expect(t.offer.view.language).toBe("fr");
  });
});

describe("LanguageOffer: Lock", () => {
  it("restarts running captions at once when every model of the new route is installed", async () => {
    const r = rig();
    hear(r, "fr", 3);
    const outcome = await r.offer.lock("fr");
    expect(outcome).toBe("restarted");
    expect(r.routeReady).toHaveBeenCalledWith("fr");
    expect(r.data.local.language).toBe("fr");
    expect(r.data.local.languageDecision).toEqual({ outcome: "locked", language: "fr", at: START });
    expect(r.restart).toHaveBeenCalledTimes(1);
    expect(r.log).toHaveBeenCalledWith("fr", "restarted");
    expect(r.offer.view).toMatchObject({ language: null, source: null, busy: false, result: { language: "fr", outcome: "restarted" } });
  });

  it("never restarts a running session into a download: the language applies at the next Start", async () => {
    const r = rig({}, false);
    hear(r, "fr", 3);
    expect(await r.offer.lock("fr")).toBe("download");
    expect(r.data.local.language).toBe("fr");
    expect(r.restart).not.toHaveBeenCalled();
    expect(r.offer.view.result).toEqual({ language: "fr", outcome: "download" });
  });

  it("only stores the language when nothing is running", async () => {
    const r = rig();
    r.state.running = false;
    expect(await r.offer.lock("fr")).toBe("next_start");
    expect(r.restart).not.toHaveBeenCalled();
    expect(r.data.local.language).toBe("fr");
  });

  it("does not restart when the router cannot be asked, but still locks", async () => {
    for (const failure of [new Error("ipc"), null]) {
      const r = rig({}, failure);
      expect(await r.offer.lock("fr")).toBe("next_start");
      expect(r.restart).not.toHaveBeenCalled();
      expect(r.data.local.language).toBe("fr");
    }
  });

  it("does not wait forever for the router: after a few seconds the language applies at the next Start", async () => {
    const r = rig();
    r.routeReady.mockImplementation(() => new Promise<boolean>(() => {}));
    const locking = r.offer.lock("fr");
    expect(r.offer.view.busy).toBe(true);
    await vi.advanceTimersByTimeAsync(ROUTE_CHECK_TIMEOUT_MS);
    expect(await locking).toBe("next_start");
    expect(r.data.local.language).toBe("fr");
    expect(r.restart).not.toHaveBeenCalled();
    expect(r.offer.view.busy).toBe(false);
  });

  it("reads whether the session is still running after the router answered", async () => {
    const r = rig();
    r.routeReady.mockImplementation(async () => {
      r.state.running = false;
      return true;
    });
    expect(await r.offer.lock("fr")).toBe("next_start");
    expect(r.restart).not.toHaveBeenCalled();
  });

  it("locks to a tag as its lockable code, and refuses what is not a language", async () => {
    const r = rig();
    expect(await r.offer.lock("pt-BR")).toBe("restarted");
    expect(r.data.local.language).toBe("pt");
    expect(await r.offer.lock("auto")).toBeNull();
    expect(await r.offer.lock("")).toBeNull();
    expect(r.data.local.language).toBe("pt");
  });

  it("applies one lock at a time", async () => {
    const r = rig();
    const first = r.offer.lock("fr");
    const second = r.offer.lock("de");
    expect(await second).toBeNull();
    expect(await first).toBe("restarted");
    expect(r.data.local.language).toBe("fr");
    expect(r.restart).toHaveBeenCalledTimes(1);
  });

  it("a lock after Keep automatic is allowed (the user changed their mind)", async () => {
    const r = rig();
    hear(r, "fr", 3);
    r.offer.keepAutomatic();
    await r.offer.lock("fr");
    expect(r.data.local.languageDecision.outcome).toBe("locked");
  });

  it("clears its feedback line after a while, or when it is closed", async () => {
    const r = rig();
    await r.offer.lock("fr");
    expect(r.offer.view.result).not.toBeNull();
    vi.advanceTimersByTime(RESULT_VISIBLE_MS - 1);
    expect(r.offer.view.result).not.toBeNull();
    vi.advanceTimersByTime(1);
    expect(r.offer.view.result).toBeNull();

    await r.offer.lock("de");
    expect(r.offer.view.result).toMatchObject({ language: "de" });
    r.offer.dismissResult();
    expect(r.offer.view.result).toBeNull();
    vi.advanceTimersByTime(2 * RESULT_VISIBLE_MS);
    expect(r.offer.view.result).toBeNull();
  });

  it("covers every outcome the UI has a line for", async () => {
    const seen = new Set<LockOutcome>();
    for (const [running, ready] of [[true, true], [true, false], [false, true], [false, false], [true, null]] as const) {
      const r = rig({}, ready);
      r.state.running = running;
      const outcome = await r.offer.lock("fr");
      if (outcome) seen.add(outcome);
    }
    expect([...seen].sort()).toEqual(["download", "next_start", "restarted"]);
  });
});
