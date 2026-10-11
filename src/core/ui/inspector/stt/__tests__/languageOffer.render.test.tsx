/**
 * Server-side render tests of the "Lock captions to your language?" card (`LanguageOffer.tsx`), its toast, and the
 * language question of the setup wizard's STT step (`core/ui/Onboarding.tsx`). Real English strings from
 * `public/i18n/en/translation.json`, so a missing key shows up as the key.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import en from "../../../../../../public/i18n/en/translation.json";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => undefined) }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { isActive: vi.fn(() => false), dismiss: vi.fn() }) }));

import { LanguageOffer } from "@/core/services/stt/languageOffer";
import { decisionFor } from "@/core/services/stt/languageSuggestion";
import { Service_STT_Schema, STT_Backends, type STT_State } from "@/core/services/stt/schema";
import { SttStep } from "@/core/ui/Onboarding";
import {
  LanguageOfferCard,
  LanguageOfferPanel,
  LanguageOfferResult,
  createOfferToastSync,
  type OfferToastApi,
} from "../LanguageOffer";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  await i18n.use(initReactI18next).init({
    lng: "en",
    resources: { en: { translation: en } },
    interpolation: { escapeValue: false },
  });
});

const noop = () => {};

describe("LanguageOfferCard", () => {
  it("asks about the detected language, says how a lock takes effect, and has the three answers", () => {
    const out = html(<LanguageOfferCard source="detected" language="French" busy={false} onLock={noop} onKeep={noop} onNever={noop} />);
    expect(out).toContain("Detected French. Lock captions to French?");
    expect(out).toContain("Setting the language usually makes captions more accurate (most of all for European languages) and uses less memory.");
    expect(out).toContain("restarts running captions at once when the French models are already installed");
    expect(out).toContain("otherwise it applies the next time you press Start");
    expect(out).toContain(">Lock to French<");
    expect(out).toContain(">Keep automatic<");
    expect(out).toContain(">Don't ask again<");
    expect(out).not.toMatch(/stt\.lang_offer_/);
  });

  it("names the Windows language when that is the reason", () => {
    const out = html(<LanguageOfferCard source="system" language="French" busy={false} onLock={noop} onKeep={noop} onNever={noop} />);
    expect(out).toContain("Your Windows language is French. Lock captions to French?");
    expect(out).not.toContain("Detected French");
  });

  it("disables the answers while a lock is being applied", () => {
    const out = html(<LanguageOfferCard source="detected" language="French" busy onLock={noop} onKeep={noop} onNever={noop} />);
    expect(out).toContain("Locking...");
    expect(out.match(/disabled=""/g)).toHaveLength(3);
  });
});

describe("LanguageOfferResult", () => {
  it("says what happened for each way a lock can take effect", () => {
    const line = (outcome: "restarted" | "next_start" | "download") => html(<LanguageOfferResult language="French" outcome={outcome} onClose={noop} />);
    expect(line("restarted")).toContain("Locked to French. Captions restarted with the French models.");
    expect(line("next_start")).toContain("Locked to French. It applies the next time you press Start.");
    expect(line("download")).toContain("The next Start downloads the French speech models first, and asks before it does.");
    // Captions that start by themselves never ask and never download: they keep detecting the language until then.
    expect(line("download")).toContain("Until then, captions that start by themselves keep detecting the language automatically.");
    expect(line("download")).not.toMatch(/stt\.lang_offer_/);
  });
});

describe("LanguageOfferPanel (STT inspector)", () => {
  let data: STT_State;
  let offer: LanguageOffer;

  function mount(overrides: Partial<STT_State> = {}, languages = ["fr-FR", "en-US"]) {
    data = proxy<STT_State>({ ...Service_STT_Schema.parse({}), backend: STT_Backends.local, ...overrides });
    offer = new LanguageOffer({
      data: () => data,
      patch: (fn) => fn(data),
      now: () => Date.now(),
      running: () => false,
      routeReady: async () => true,
      restart: noop,
      log: noop,
    });
    vi.stubGlobal("navigator", { language: languages[0], languages });
    vi.stubGlobal("window", { ApiServer: { state: { services: { stt: { data } } }, stt: { languageOffer: offer } } });
  }

  beforeEach(() => mount());
  afterEach(() => vi.unstubAllGlobals());

  it("shows the Windows language to a profile on automatic that never answered", () => {
    const out = html(<LanguageOfferPanel />);
    expect(out).toContain("Your Windows language is French. Lock captions to French?");
    expect(out).toContain(">Lock to French<");
  });

  it("does not turn an English Windows into a one-click Lock to English", () => {
    // English is the display language of many PCs whose owner speaks something else: nothing is offered from it.
    mount({}, ["en-US"]);
    expect(html(<LanguageOfferPanel />)).toBe("");
    // The first routable language after the English display language is a better prior, and is still only a question.
    mount({}, ["en-US", "pt-BR"]);
    expect(html(<LanguageOfferPanel />)).toContain("Your Windows language is Portuguese. Lock captions to Portuguese?");
  });

  it("prefers what the session heard over the Windows language", () => {
    offer.view.language = "de";
    offer.view.source = "detected";
    const out = html(<LanguageOfferPanel />);
    expect(out).toContain("Detected German. Lock captions to German?");
    expect(out).not.toContain("Windows language");
  });

  it("does not show an offer for a profile that has since left the automatic language or backend", () => {
    offer.view.language = "de";
    offer.view.source = "detected";
    data.local.language = "de";
    expect(html(<LanguageOfferPanel />)).toBe("");
    data.local.language = "auto";
    data.backend = STT_Backends.whisper;
    expect(html(<LanguageOfferPanel />)).toBe("");
  });

  it("shows the feedback line after a lock", () => {
    offer.view.result = { language: "fr", outcome: "download" };
    expect(html(<LanguageOfferPanel />)).toContain("Locked to French.");
  });

  it("stays out of the way once answered, when a language is set, on another backend, or without a routable Windows language", () => {
    for (const outcome of ["locked", "kept", "never"] as const) {
      mount({ local: { ...Service_STT_Schema.parse({}).local, languageDecision: decisionFor(outcome, "fr", Date.now()) } });
      expect(html(<LanguageOfferPanel />)).toBe("");
    }
    mount({ local: { ...Service_STT_Schema.parse({}).local, language: "de" } });
    expect(html(<LanguageOfferPanel />)).toBe("");
    mount({ backend: STT_Backends.whisper });
    expect(html(<LanguageOfferPanel />)).toBe("");
    mount({}, ["xx-XX"]);
    expect(html(<LanguageOfferPanel />)).toBe("");
  });
});

describe("the offer toast", () => {
  function setup() {
    const state = new Map<string, () => void>();
    const api: OfferToastApi = {
      isActive: vi.fn((id: string) => state.has(id)),
      show: vi.fn((_content, options) => void state.set(options.toastId, options.onClose)),
      dismiss: vi.fn((id: string) => {
        const onClose = state.get(id);
        state.delete(id);
        onClose?.();
      }),
    };
    const controller = {
      view: proxy({ language: null as string | null, source: null, busy: false, result: null }) as LanguageOffer["view"],
      keepAutomatic: vi.fn(),
      dismissResult: vi.fn(),
    };
    const sync = createOfferToastSync(api, controller, () => "body");
    return { api, controller, sync, state };
  }

  it("is a persistent toast that closes neither by itself nor on a click, and is shown once", () => {
    const { api, sync } = setup();
    sync(true);
    sync(true);
    expect(api.show).toHaveBeenCalledTimes(1);
    expect(api.show).toHaveBeenCalledWith("body", expect.objectContaining({ autoClose: false, closeOnClick: false, draggable: false, icon: false }));
  });

  it("closing it with the X answers Keep automatic for an offer and dismisses the feedback line otherwise", () => {
    const offered = setup();
    offered.controller.view.language = "fr";
    offered.sync(true);
    offered.state.get("stt-language-offer")?.();
    expect(offered.controller.keepAutomatic).toHaveBeenCalledTimes(1);

    const result = setup();
    result.sync(true);
    result.state.get("stt-language-offer")?.();
    expect(result.controller.dismissResult).toHaveBeenCalledTimes(1);
    expect(result.controller.keepAutomatic).not.toHaveBeenCalled();
  });

  it("taking it down because the panel opened or the question was answered is not an answer", () => {
    const { api, controller, sync } = setup();
    controller.view.language = "fr";
    sync(true);
    sync(false);
    expect(api.dismiss).toHaveBeenCalledTimes(1);
    expect(controller.keepAutomatic).not.toHaveBeenCalled();
    expect(controller.dismissResult).not.toHaveBeenCalled();
    // And a later toast works again, including its X.
    sync(true);
    expect(api.show).toHaveBeenCalledTimes(2);
  });

  it("shows the toast again when it is wanted while the one taken down is still leaving", () => {
    vi.useFakeTimers();
    try {
      const closers = new Map<string, () => void>();
      const api: OfferToastApi = {
        isActive: vi.fn((id: string) => closers.has(id)),
        show: vi.fn((_content, options) => void closers.set(options.toastId, options.onClose)),
        // The library reports the close (and forgets the toast) only after its exit animation.
        dismiss: vi.fn((id: string) => {
          const onClose = closers.get(id);
          setTimeout(() => {
            closers.delete(id);
            onClose?.();
          }, 500);
        }),
      };
      const controller = { view: proxy({ language: "fr" as string | null, source: null, busy: false, result: null }) as LanguageOffer["view"], keepAutomatic: vi.fn(), dismissResult: vi.fn() };
      const sync = createOfferToastSync(api, controller, () => "body");
      sync(true);
      sync(false);
      sync(true); // wanted again before the exit animation ended
      expect(api.show).toHaveBeenCalledTimes(1);
      vi.advanceTimersByTime(600);
      expect(api.show).toHaveBeenCalledTimes(2);
      expect(controller.keepAutomatic).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("does nothing when there is nothing to take down", () => {
    const { api, sync } = setup();
    sync(false);
    expect(api.dismiss).not.toHaveBeenCalled();
  });

  it("survives a toast library that throws", () => {
    const { api, sync } = setup();
    vi.mocked(api.show).mockImplementation(() => {
      throw new Error("no container");
    });
    expect(() => sync(true)).not.toThrow();
  });
});

describe("onboarding: What language do you speak?", () => {
  afterEach(() => vi.unstubAllGlobals());

  const step = (languages: string[]) => {
    vi.stubGlobal("navigator", { language: languages[0], languages });
    return html(<SttStep onNext={noop} onBack={noop} />);
  };
  const selected = (out: string) => out.match(/<option value="([^"]*)" selected="">/)?.[1];

  it("asks the question with the one-line reason", () => {
    const out = step(["en-US"]);
    expect(out).toContain("What language do you speak?");
    expect(out).toContain("Setting your language is the biggest step for accurate captions.");
  });

  it("preselects the Windows display language when Sigil can route it", () => {
    const out = step(["fr-FR", "en-US"]);
    expect(selected(out)).toBe("fr");
    expect(out).toContain("Chosen from your Windows language.");
    expect(selected(step(["ru-RU"]))).toBe("ru");
  });

  it("does not preselect English because Windows is in English", () => {
    const english = step(["en-US"]);
    expect(selected(english)).toBe("auto");
    expect(english).not.toContain("Chosen from your Windows language.");
    // A second language the webview lists is a better prior than the English display language.
    expect(selected(step(["en-US", "pt-BR"]))).toBe("pt");
  });

  it("falls back to an input language, and to Detect automatically when none is routable", () => {
    expect(selected(step(["xx-XX", "de-DE"]))).toBe("de");
    const none = step(["xx-XX"]);
    expect(selected(none)).toBe("auto");
    expect(none).not.toContain("Chosen from your Windows language.");
  });

  it("keeps Detect automatically as a choice next to the languages", () => {
    const out = step(["fr-FR"]);
    expect(out).toContain('<option value="auto">Detect automatically</option>');
    expect(out).toContain('<option value="ru">Russian</option>');
    expect(out).toContain('<option value="ja">Japanese</option>');
  });

  it("is unchanged for the other engines in the list", () => {
    const out = step(["fr-FR"]);
    expect(out).toContain("On this PC (automatic)");
    expect(out).toContain("Whisper");
    expect(out).toContain("Vosk");
  });
});
