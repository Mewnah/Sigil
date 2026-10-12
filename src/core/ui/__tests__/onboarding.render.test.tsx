/**
 * Render tests of the first-run setup wizard (`Onboarding.tsx`): five steps, real English strings plus the staged
 * ones (`sttTestI18n`), so a hard-coded or missing string shows up. The wizard's decisions (who sees it) are tested in
 * `services/state/__tests__/firstRun.test.ts`; here is what each step shows and what finishing and skipping write.
 */
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => []) }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import { planDesktopDefaultMigration, planRetiredBackendMigration } from "@/core/services/stt/migrate";
import { Service_STT_Schema } from "@/core/services/stt/schema";
import { Services } from "@/services-registry";
import Onboarding, {
  AudioStep,
  DoneStep,
  INITIAL_STT_ANSWERS,
  ONBOARDING_AUTO_START_DEFAULT,
  ONBOARDING_STEPS,
  SendStep,
  SttAutoStartOption,
  SttStep,
  WIZARD_ROOT_CLASS,
  WelcomeStep,
  applyOnboardingSttSave,
  finishOnboarding,
  planOnboardingAudioSave,
  saveOnboardingStt,
  skipOnboarding,
} from "../Onboarding";
import { planOnboardingSttSave } from "@/core/services/stt/languageSuggestion";

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

beforeAll(async () => {
  i18n.use(initReactI18next); // components read the default instance through useTranslation
  await initSttTestI18n();
});

afterEach(() => vi.unstubAllGlobals());

const noop = () => {};
const rawKey = /\b(?:onboarding|stt)\.[a-z_]+\b/;

/** A host state with just what the wizard writes. */
function stubHost(state: Record<string, unknown> = {}) {
  const stt = {
    showActionButton: false,
    data: { backend: "local", desktopDefaultMigrated: false, localOfferShown: false, autoStart: false, local: { language: "auto" } },
  };
  const host = {
    state: { onboardingComplete: false, audioInputDeviceNative: "", audioOutputDevice: "", audioInputDeviceNativeMigrated: false, ...state },
    patchService: vi.fn((_key: "stt", fn: (s: typeof stt) => void) => fn(stt)),
    changeTab: vi.fn(),
  };
  vi.stubGlobal("window", { ApiServer: host });
  return { host, stt };
}

describe("wizard steps", () => {
  it("welcomes the user in one sentence and offers one button", () => {
    const out = html(<WelcomeStep onNext={noop} />);
    expect(out).toContain("Welcome to Sigil");
    expect(out).toContain("Live captions for your VRChat streams");
    expect(out).toContain("Get started");
    expect(out).not.toMatch(rawKey);
  });

  it("asks for the microphone and speakers, each defaulting to the system default", () => {
    stubHost();
    const out = html(<AudioStep onNext={noop} onBack={noop} />);
    expect(out).toContain("Microphone and speakers");
    expect(out).toContain("Settings → Audio");
    expect(out).toMatch(/<label[^>]*for="onboarding-microphone"[^>]*><span[^>]*>Microphone</);
    expect(out).toMatch(/<label[^>]*for="onboarding-speakers"[^>]*><span[^>]*>Speakers</);
    expect(out.match(/<option value="" selected="">System default<\/option>/g)).toHaveLength(2); // preselected
    expect(out).toContain("Back");
    expect(out).toContain("Continue");
    expect(out).not.toMatch(rawKey);
  });

  it("keeps a saved device selectable when it is not plugged in", () => {
    stubHost({ audioInputDeviceNative: "USB Mic (unplugged)" });
    const out = html(<AudioStep onNext={noop} onBack={noop} />);
    expect(out).toContain('<option value="USB Mic (unplugged)" selected="">USB Mic (unplugged)</option>');
  });

  it("shows the built-in engine as the only choice, with the language question and a pointer to the other engines", () => {
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    const out = html(<SttStep onNext={noop} onBack={noop} />);
    expect(out).toContain("Speech to Text");
    expect(out).toContain("Sigil recognizes your speech on this PC. Models download once, after you agree.");
    expect(out).toContain("Built-in (on your PC)");
    expect(out).toContain("What language do you speak?");
    expect(out).toContain("Online engines and others are in the Speech to Text engine list.");
    expect(out).not.toMatch(/Whisper|Vosk|Moonshine|automatic\)/);
    expect(out.match(/data-testid="onboarding-engine"/g)).toHaveLength(1);
    expect(out).not.toMatch(rawKey);
  });

  it("offers to start captions when Sigil opens: a labelled checkbox, on by default, under the language", () => {
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    const out = html(<SttStep onNext={noop} onBack={noop} />);
    expect(ONBOARDING_AUTO_START_DEFAULT).toBe(true);
    const box = out.match(/<input[^>]*id="onboarding-stt-autostart"[^>]*>/)?.[0] ?? "";
    expect(box).toContain('type="checkbox"');
    expect(box).toContain('checked=""'); // on until the user turns it off
    expect(box).toContain('aria-describedby="onboarding-stt-autostart-hint"');
    expect(box).not.toMatch(/tabindex|disabled/); // a native checkbox: Tab reaches it and Space toggles it
    expect(box).toMatch(/class="[^"]*\bshrink-0\b/); // a long translation wraps the words, it does not squeeze the square
    // The words name the input (a click on them toggles it); the hint describes it.
    expect(out).toMatch(/<label[^>]*for="onboarding-stt-autostart"[^>]*>.*Start captions when Sigil opens<\/span><\/label>/);
    expect(out).toMatch(/<label[^>]*class="[^"]*\bw-fit\b[^"]*\bmax-w-full\b[^"]*"[^>]*for="onboarding-stt-autostart"/); // as wide as its words, not the step
    // The hint promises only what happens: it starts at every launch, and the first Start downloads the models.
    expect(out).toMatch(/<p id="onboarding-stt-autostart-hint"[^>]*>Captions start by themselves each time you open Sigil\. Press Start once first to download the speech models\.<\/p>/);
    expect(out).not.toMatch(/crash|restart/i);
    const at = (s: string) => out.indexOf(s);
    expect(at('data-testid="onboarding-language"')).toBeGreaterThan(-1);
    expect(at('data-testid="onboarding-language"')).toBeLessThan(at('data-testid="onboarding-autostart"'));
    expect(at('data-testid="onboarding-autostart"')).toBeLessThan(at("Online engines and others"));
    expect(at("Online engines and others")).toBeLessThan(at("Continue"));
    expect(out.match(/type="checkbox"/g)).toHaveLength(1);
    expect(out).not.toMatch(rawKey);
  });

  it("shows the checkbox off when it is off, and lines the hint up with the label for a right-to-left layout", () => {
    const off = html(<SttAutoStartOption checked={false} onChange={noop} />);
    expect(off).toContain('type="checkbox"');
    expect(off).not.toContain('checked=""');
    expect(off).toContain("ms-9"); // checkbox (size-6) + gap-3, on the start side
    expect(off).not.toMatch(/\b(?:ml|mr|pl|pr)-\d/);
    expect(html(<SttAutoStartOption checked onChange={noop} />)).toContain('checked=""');
  });

  it("shows what the wizard holds: a box left unchecked and a language picked stay so when the step comes back", () => {
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    const out = html(<SttStep onNext={noop} onBack={noop} answers={{ language: "ja", autoStart: false }} onAnswers={noop} />);
    expect(out.match(/<input[^>]*id="onboarding-stt-autostart"[^>]*>/)?.[0]).not.toContain('checked=""');
    expect(out).toMatch(/<option value="ja" selected="">/); // not the Windows language the step opens on
    expect(out).not.toMatch(/<option value="en" selected="">/);
    // Nothing picked yet: the Windows language, and the box on.
    const fresh = html(<SttStep onNext={noop} onBack={noop} answers={INITIAL_STT_ANSWERS} onAnswers={noop} />);
    expect(fresh).toMatch(/<option value="en" selected="">/);
    expect(fresh.match(/<input[^>]*id="onboarding-stt-autostart"[^>]*>/)?.[0]).toContain('checked=""');
  });

  it("asks where captions should go: VRChat, then OBS Studio, then Twitch, Kick and Discord as small buttons", () => {
    const out = html(<SendStep onNext={noop} onBack={noop} onPick={noop} />);
    expect(out).toContain("Where should captions go?");
    expect(out).toContain("Pick one to set it up now, or finish and do it later under Integrations.");
    const at = (s: string) => out.indexOf(s);
    expect(at("VRChat")).toBeGreaterThan(-1);
    expect(at("VRChat")).toBeLessThan(at("OBS Studio"));
    expect(at("OBS Studio")).toBeLessThan(at("Twitch"));
    expect(at("Twitch")).toBeLessThan(at("Kick"));
    expect(at("Kick")).toBeLessThan(at("Discord"));
    expect(out).toContain("Show captions in the chatbox");
    expect(out).toContain("Captions on your stream");
    expect(out).toContain("Finish setup");
    expect(out).toContain("btn-sm"); // the three small buttons
    expect(out).not.toMatch(/Whisper|Vosk/);
    expect(out).not.toMatch(rawKey);
  });

  it("ends on a done step that says what to press", () => {
    const out = html(<DoneStep onOpen={noop} />);
    expect(out).toContain("You're all set");
    expect(out).toContain("Press Start in the bottom bar and speak.");
    expect(out).toContain("Translation, Text to Speech and Voice Changer are under Speech in the left menu.");
    expect(out).toContain("Open Sigil");
    expect(out).not.toMatch(/Whisper|Vosk/);
    expect(out).not.toMatch(rawKey);
  });
});

describe("wizard shell", () => {
  it("is a labelled modal dialog above the app, with five progress dots", () => {
    const out = html(<Onboarding />);
    expect(ONBOARDING_STEPS).toHaveLength(5);
    expect(out).toContain('role="dialog"');
    expect(out).toContain('aria-modal="true"');
    expect(out).toContain('aria-label="Sigil setup"');
    expect(out).toContain("z-[90]");
    expect(out).toContain('aria-label="Setup, step 1 of 5"');
    expect(out.match(/rounded-full transition-colors/g)).toHaveLength(5);
    expect(out).toContain("Welcome to Sigil");
  });

  it("offers Skip on the first four steps and not on the last", () => {
    stubHost();
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    for (const step of [0, 1, 2, 3]) {
      const out = html(<Onboarding initialStep={step} />);
      expect(out, `step ${step + 1}`).toContain("Skip setup");
      expect(out).toContain(`aria-label="Setup, step ${step + 1} of 5"`);
    }
    const last = html(<Onboarding initialStep={4} />);
    expect(last).not.toContain("Skip setup");
    expect(last).toContain('aria-label="Setup, step 5 of 5"');
    expect(last).toContain("You're all set");
  });

  it("has no accessibility step and no leftover hard-coded English", () => {
    stubHost();
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    const all = [0, 1, 2, 3, 4].map((step) => html(<Onboarding initialStep={step} />)).join("\n");
    expect(all).not.toMatch(/High Contrast|Accessibility|Launch Dashboard|Connect Integrations|Audio Setup|Speech Recognition/);
  });
});

describe("wizard shell: title bar, right-to-left layout and assistive technology", () => {
  const everyStep = () => {
    stubHost();
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    return [0, 1, 2, 3, 4].map((step) => html(<Onboarding initialStep={step} />));
  };

  it("leaves the title bar uncovered, so the window can still be moved, minimised and closed", () => {
    // The window has no system decorations: SigilHeader (h-10) holds the drag area and the window buttons.
    expect(WIZARD_ROOT_CLASS).toContain("top-10");
    expect(WIZARD_ROOT_CLASS).not.toMatch(/\binset-0\b|\btop-0\b/);
    expect(WIZARD_ROOT_CLASS).toContain("z-[90]");
    expect(html(<Onboarding />)).toContain(WIZARD_ROOT_CLASS);
  });

  it("is mirrored for right-to-left languages: start-aligned text and flipped arrows", () => {
    const all = everyStep().join("\n");
    expect(all).not.toMatch(/\btext-left\b|\btext-right\b|\b(?:ml|mr|pl|pr)-\d/);
    expect(all).toContain("text-start");
    const arrows = (node: ReactNode) => html(node).match(/<svg[^>]*rtl:-scale-x-100/g)?.length ?? 0;
    stubHost();
    expect(arrows(<WelcomeStep onNext={noop} />)).toBe(1); // Get started
    expect(arrows(<AudioStep onNext={noop} onBack={noop} />)).toBe(2); // Back, Continue
    expect(arrows(<SttStep onNext={noop} onBack={noop} />)).toBe(2);
    expect(arrows(<SendStep onNext={noop} onBack={noop} onPick={noop} />)).toBe(3); // Back, the two cards (Finish setup has a rocket)
    expect(arrows(<DoneStep onOpen={noop} />)).toBe(1); // Open Sigil
  });

  it("announces the step as a progress bar, not a live status", () => {
    stubHost();
    vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
    const out = html(<Onboarding initialStep={2} />);
    expect(out).toContain('role="progressbar"');
    expect(out).toContain('aria-valuenow="3"');
    expect(out).toContain('aria-valuemin="1"');
    expect(out).toContain('aria-valuemax="5"');
    expect(out).toContain('aria-valuetext="Setup, step 3 of 5"');
    expect(out).not.toContain('role="status"');
  });

  it("gives every step a heading that can take focus, so a new step starts there", () => {
    for (const out of everyStep()) expect(out).toMatch(/<h2 tabindex="-1" class="[^"]*outline-none"/);
  });
});

describe("what the speech step writes", () => {
  // A saved profile on Web Speech that has not seen the offer, as the schema gives it.
  const profile = () => Service_STT_Schema.parse({});

  it("puts the profile on the built-in engine, marks the migration done and the offer as answered", () => {
    const data = profile();
    applyOnboardingSttSave(data, planOnboardingSttSave("local", "auto", 1000), true);
    expect(data.backend).toBe("local");
    expect(data.desktopDefaultMigrated).toBe(true);
    // The built-in engine was chosen here: a later switch to Chrome or Edge is that user's choice, not an offer.
    expect(data.localOfferShown).toBe(true);
    expect(data.local.language).toBe("auto");
  });

  it("stores the language the user answered", () => {
    const data = profile();
    applyOnboardingSttSave(data, planOnboardingSttSave("local", "ja", 1000), true);
    expect(data.local.language).toBe("ja");
    expect(data.local.languageDecision).toMatchObject({ outcome: "locked", language: "ja" });
  });

  it("stores the answer to \"Start captions when Sigil opens\", on or off", () => {
    expect(profile().autoStart).toBe(false); // the schema default, which a skipped wizard keeps
    const on = profile();
    applyOnboardingSttSave(on, planOnboardingSttSave("local", "auto", 1000), true);
    expect(on.autoStart).toBe(true);
    const off = profile();
    off.autoStart = true;
    applyOnboardingSttSave(off, planOnboardingSttSave("local", "auto", 1000), false);
    expect(off.autoStart).toBe(false);
  });

  it("leaves a profile that the next launch starts by itself, with nothing holding the start back", () => {
    const data = profile();
    applyOnboardingSttSave(data, planOnboardingSttSave("local", "en", 1000), ONBOARDING_AUTO_START_DEFAULT);
    expect(data.autoStart).toBe(true);
    // `Service_STT.init` starts when auto-start is on and neither migration claims this launch. Whether the wizard is
    // done or shows again (quit in the middle), the migration leaves the profile alone: what holds the start back while
    // the wizard shows is `Service_STT.init` itself (`stt/__tests__/service.test.ts`), not a migration.
    for (const freshInstall of [false, true]) {
      const migration = planDesktopDefaultMigration({
        isDesktop: true,
        alreadyMigrated: data.desktopDefaultMigrated,
        backend: data.backend,
        freshInstall,
        localOfferShown: data.localOfferShown,
      });
      expect(migration, `freshInstall ${freshInstall}`).toEqual({ action: "none" });
    }
    expect(planRetiredBackendMigration({ isDesktop: true, backend: data.backend })).toEqual({ action: "none" });
  });
});

describe("Continue on the speech step", () => {
  it("starts captions at launch by default, and saves the engine and the language with it", () => {
    const { host, stt } = stubHost();
    saveOnboardingStt("fr", ONBOARDING_AUTO_START_DEFAULT, 1000);
    expect(host.patchService).toHaveBeenCalledTimes(1);
    expect(host.patchService).toHaveBeenCalledWith("stt", expect.any(Function));
    expect(stt.data.autoStart).toBe(true);
    expect(stt.data.backend).toBe("local");
    expect(stt.data.local.language).toBe("fr");
  });

  it("does not start captions at launch when the box was unchecked", () => {
    const { stt } = stubHost();
    stt.data.autoStart = true;
    saveOnboardingStt("auto", false, 1000);
    expect(stt.data.autoStart).toBe(false);
    expect(stt.data.desktopDefaultMigrated).toBe(true); // the rest of the step is saved all the same
  });
});

describe("finishing and skipping", () => {
  it("skip only marks the wizard done", () => {
    const { host, stt } = stubHost();
    skipOnboarding();
    expect(host.state.onboardingComplete).toBe(true);
    expect(stt.showActionButton).toBe(false);
    expect(stt.data.autoStart).toBe(false); // a skipped wizard keeps the schema default: captions start by hand
    expect(host.patchService).not.toHaveBeenCalled();
    expect(host.changeTab).not.toHaveBeenCalled();
  });

  it("finish without Continue on the speech step does not turn on auto-start either", () => {
    const { stt } = stubHost();
    finishOnboarding();
    expect(stt.data.autoStart).toBe(false);
  });

  it("finish puts Speech to Text in the bottom bar and marks the wizard done", () => {
    const { host, stt } = stubHost();
    finishOnboarding();
    expect(stt.showActionButton).toBe(true);
    expect(host.state.onboardingComplete).toBe(true);
    expect(host.changeTab).not.toHaveBeenCalled();
  });

  it("finish with a destination opens its panel after the wizard is done", () => {
    for (const tab of [Services.vrc, Services.obs, Services.twitch, Services.kick, Services.discord]) {
      const { host } = stubHost();
      host.changeTab.mockImplementation(() => expect(host.state.onboardingComplete).toBe(true));
      finishOnboarding(tab);
      expect(host.changeTab).toHaveBeenCalledWith({ tab });
    }
  });
});

describe("the audio step never writes a device the user did not choose", () => {
  const saved = { input: "", output: "" };

  it("writes nothing when nothing changed (also when the device lists never loaded)", () => {
    expect(planOnboardingAudioSave(saved, { input: "", output: "" })).toEqual({});
    expect(planOnboardingAudioSave({ input: "Mic A", output: "Spk A" }, { input: "Mic A", output: "Spk A" })).toEqual({});
  });

  it("writes only the device that changed, and system default is a choice too", () => {
    expect(planOnboardingAudioSave(saved, { input: "Mic B", output: "" })).toEqual({ input: "Mic B" });
    expect(planOnboardingAudioSave(saved, { input: "", output: "Spk B" })).toEqual({ output: "Spk B" });
    expect(planOnboardingAudioSave({ input: "Mic A", output: "Spk A" }, { input: "", output: "Spk A" })).toEqual({ input: "" });
  });
});
