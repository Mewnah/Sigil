/**
 * What the speech step of the setup wizard does with a click, and what the wizard keeps between steps. The repo has no
 * DOM environment, so the components run outside React's renderer: `withHooks` gives them hook state (kept between calls
 * when the same slots are passed, effects never run) and `expand` replaces each function component in a tree with what
 * it renders. The handlers on the resulting `input`, `select` and `button` elements are the ones a click would call.
 * What the steps show is in `onboarding.render.test.tsx`.
 */
import * as React from "react";
import type { ReactElement, ReactNode } from "react";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn(async () => []) }));

import { initSttTestI18n } from "@/core/services/stt/__tests__/sttTestI18n";
import Onboarding, { AudioStep, INITIAL_STT_ANSWERS, SttStep } from "../Onboarding";

type AnyElement = ReactElement<Record<string, any>>;

const internals = (React as unknown as { __SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED: { ReactCurrentDispatcher: { current: unknown } } })
  .__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED;

/** Runs `run` with the hooks of one component instance: `slots` is its state, and survives as long as the caller keeps it. */
function withHooks<T>(slots: unknown[], run: () => T): T {
  let at = 0;
  const dispatcher = {
    useState: (initial: unknown) => {
      const i = at++;
      if (!(i in slots)) slots[i] = typeof initial === "function" ? (initial as () => unknown)() : initial;
      const set = (value: unknown) => { slots[i] = typeof value === "function" ? (value as (v: unknown) => unknown)(slots[i]) : value; };
      return [slots[i], set];
    },
    useRef: (initial: unknown) => {
      const i = at++;
      if (!(i in slots)) slots[i] = { current: initial };
      return slots[i];
    },
    useMemo: (make: () => unknown) => make(),
    useCallback: (fn: unknown) => fn,
    useContext: (context: { _currentValue: unknown }) => context._currentValue,
    useEffect: () => {},
    useLayoutEffect: () => {},
  };
  const previous = internals.ReactCurrentDispatcher.current;
  internals.ReactCurrentDispatcher.current = dispatcher;
  try {
    return run();
  } finally {
    internals.ReactCurrentDispatcher.current = previous;
  }
}

/** Replaces every function component in `node` by what it renders (fresh hook state each), down to the DOM elements. */
function expand(node: ReactNode): ReactNode {
  if (Array.isArray(node)) return node.map(expand);
  if (!React.isValidElement(node)) return node;
  const el = node as AnyElement;
  if (typeof el.type === "function") return expand(withHooks([], () => (el.type as (props: unknown) => ReactNode)(el.props)));
  return { ...el, props: { ...el.props, children: expand(el.props.children) } } as AnyElement;
}

/** The first element of `node` (depth first) that `match` accepts. */
function find(node: ReactNode, match: (el: AnyElement) => boolean): AnyElement {
  const found = search(node, match);
  if (!found) throw new Error("no element matches");
  return found;
}

function search(node: ReactNode, match: (el: AnyElement) => boolean): AnyElement | undefined {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = search(child, match);
      if (found) return found;
    }
    return undefined;
  }
  if (!React.isValidElement(node)) return undefined;
  const el = node as AnyElement;
  return match(el) ? el : search(el.props.children, match);
}

const isCheckbox = (el: AnyElement) => el.type === "input" && el.props.type === "checkbox";
const isLanguageSelect = (el: AnyElement) => el.type === "select" && el.props.id === "onboarding-stt-language";
const isContinue = (el: AnyElement) => el.type === "button" && /\bbtn-primary\b/.test(String(el.props.className));
const isBack = (el: AnyElement) => el.type === "button" && /\bbtn-ghost\b/.test(String(el.props.className));

beforeAll(async () => {
  i18n.use(initReactI18next);
  await initSttTestI18n();
});

afterEach(() => vi.unstubAllGlobals());

const noop = () => {};

/** A host state with just what the speech step writes. */
function stubHost() {
  const stt = { data: { backend: "native", desktopDefaultMigrated: false, localOfferShown: false, autoStart: false, local: { language: "auto" } } };
  const patchService = vi.fn((_key: "stt", fn: (s: typeof stt) => void) => fn(stt));
  vi.stubGlobal("window", { ApiServer: { state: { onboardingComplete: false }, patchService } });
  vi.stubGlobal("navigator", { language: "en-US", languages: ["en-US"] });
  return { stt, patchService };
}

describe("the speech step hands a click to the wizard", () => {
  it("the box shows the answer it was given, and a click hands over the other one", () => {
    const onAnswers = vi.fn();
    const off = find(expand(<SttStep onNext={noop} onBack={noop} answers={{ autoStart: false }} onAnswers={onAnswers} />), isCheckbox);
    expect(off.props.checked).toBe(false);
    off.props.onChange({ target: { checked: true } });
    expect(onAnswers).toHaveBeenLastCalledWith({ autoStart: true });

    const on = find(expand(<SttStep onNext={noop} onBack={noop} answers={{ language: "ja", autoStart: true }} onAnswers={onAnswers} />), isCheckbox);
    expect(on.props.checked).toBe(true);
    on.props.onChange({ target: { checked: false } });
    expect(onAnswers).toHaveBeenLastCalledWith({ language: "ja", autoStart: false }); // the language it already had stays
  });

  it("a picked language is handed over with the box as it is", () => {
    stubHost();
    const onAnswers = vi.fn();
    const tree = expand(<SttStep onNext={noop} onBack={noop} answers={{ autoStart: false }} onAnswers={onAnswers} />);
    const select = find(tree, isLanguageSelect);
    expect(select.props.value).toBe("en"); // the Windows language until one is picked
    select.props.onChange({ target: { value: "fr" } });
    expect(onAnswers).toHaveBeenCalledWith({ language: "fr", autoStart: false });
  });
});

describe("Continue on the speech step saves what the wizard holds", () => {
  it("an unchecked box saves auto-start off, with the language that was picked", () => {
    const { stt, patchService } = stubHost();
    const onNext = vi.fn();
    find(expand(<SttStep onNext={onNext} onBack={noop} answers={{ language: "ja", autoStart: false }} onAnswers={noop} />), isContinue).props.onClick();
    expect(patchService).toHaveBeenCalledTimes(1);
    expect(stt.data.autoStart).toBe(false);
    expect(stt.data.local.language).toBe("ja");
    expect(stt.data.backend).toBe("local");
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it("a checked box saves auto-start on, and an untouched language saves the Windows language", () => {
    const { stt } = stubHost();
    find(expand(<SttStep onNext={noop} onBack={noop} answers={INITIAL_STT_ANSWERS} onAnswers={noop} />), isContinue).props.onClick();
    expect(stt.data.autoStart).toBe(true);
    expect(stt.data.local.language).toBe("en");
  });

  it("a step shown on its own saves its own defaults", () => {
    const { stt } = stubHost();
    find(expand(<SttStep onNext={noop} onBack={noop} />), isContinue).props.onClick();
    expect(stt.data.autoStart).toBe(true);
    expect(stt.data.backend).toBe("local");
  });
});

describe("the wizard keeps the answers of the speech step while the user goes back", () => {
  // `Onboarding` is memo(function): the function is `.type`. One set of slots is one mounted wizard.
  const wizard = () => {
    const slots: unknown[] = [];
    const run = (Onboarding as unknown as { type: (props: { initialStep: number }) => ReactNode }).type;
    return () => withHooks(slots, () => run({ initialStep: 2 }));
  };
  const speech = (tree: ReactNode) => find(tree, (el) => el.type === SttStep);

  it("starts with the box on and no language picked", () => {
    stubHost();
    expect(speech(wizard()()).props.answers).toEqual({ autoStart: true });
  });

  it("hands the step what it was given when it comes back after Back and Continue", () => {
    stubHost();
    const render = wizard();
    // The user unchecks the box and picks Japanese (clicks on the step), then goes back to the microphone step...
    let step = speech(render());
    step.props.onAnswers({ language: "ja", autoStart: false });
    step = speech(render());
    expect(step.props.answers).toEqual({ language: "ja", autoStart: false });
    step.props.onBack();
    const audio = find(render(), (el) => el.type === AudioStep);
    // ...and forward again: the step is mounted anew with the same answers, not the defaults.
    audio.props.onNext();
    expect(speech(render()).props.answers).toEqual({ language: "ja", autoStart: false });
  });

  it("Continue saves the held answer, not a default, after a trip back", () => {
    const { stt } = stubHost();
    const render = wizard();
    speech(render()).props.onAnswers({ language: "de", autoStart: false });
    speech(render()).props.onBack();
    find(render(), (el) => el.type === AudioStep).props.onNext();
    const again = expand(speech(render()));
    find(again, isBack); // the step is on screen with its buttons
    find(again, isContinue).props.onClick();
    expect(stt.data.autoStart).toBe(false);
    expect(stt.data.local.language).toBe("de");
  });
});
