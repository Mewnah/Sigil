import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

/**
 * What the page shows when start-up fails (src/index.tsx hands the failure to handleStartupFailure).
 * A /client mirror in OBS stays transparent and reloads itself when the failure may pass; a failure
 * that no reload fixes, and any failure in an ordinary browser, keeps the explanatory panel.
 */

vi.mock("i18next", () => ({ default: { t: (key: string) => key, language: "en" } }));

import { MirrorRefusedError, MirrorStartError } from "@/shared/services/peer/errors";
import { classifyStartupFailure, handleStartupFailure, isObsBrowserSource, StartupErrorView, type StartupFailureEnv } from "../clientStartup";

const NETWORK = { host: "192.168.1.20", port: "3030" };

function fakeEnv(overrides: Partial<StartupFailureEnv> = {}) {
  const rendered: ReactNode[] = [];
  const reasons: string[] = [];
  const env: StartupFailureEnv = {
    isClient: true,
    clientNetwork: NETWORK,
    obs: false,
    renderView: (view) => void rendered.push(view),
    scheduleReload: (reason) => {
      reasons.push(reason);
      return 4_000;
    },
    ...overrides,
  };
  return { env, rendered, reasons };
}

const html = (node: ReactNode) => renderToStaticMarkup(node as never);

describe("isObsBrowserSource", () => {
  it("is true only where OBS's browser source injected window.obsstudio", () => {
    expect(isObsBrowserSource({ obsstudio: { pluginVersion: "3.0.0" } })).toBe(true);
    expect(isObsBrowserSource({})).toBe(false);
    expect(isObsBrowserSource({ obsstudio: undefined })).toBe(false);
  });
});

describe("classifyStartupFailure", () => {
  it("tells what a reload can and cannot fix", () => {
    expect(classifyStartupFailure(new MirrorRefusedError(), true)).toBe("refused");
    expect(classifyStartupFailure(new MirrorStartError(new Error("no WebRTC")), true)).toBe("final");
    expect(classifyStartupFailure(new Error("Failed to fetch dynamically imported module"), true)).toBe("maybe-transient");
    expect(classifyStartupFailure("odd", true)).toBe("maybe-transient");
    // The host window never reloads itself.
    expect(classifyStartupFailure(new Error("x"), false)).toBe("host");
    expect(classifyStartupFailure(new MirrorRefusedError(), false)).toBe("host");
  });
});

describe("handleStartupFailure", () => {
  it("shows the panel in the host window and never reloads it", () => {
    const { env, rendered, reasons } = fakeEnv({ isClient: false, clientNetwork: undefined });
    expect(handleStartupFailure(new Error("settings broken"), env)).toBe("panel");
    expect(reasons).toEqual([]);
    expect(rendered).toHaveLength(1);
    expect(html(rendered[0])).toContain("Application failed to start");
    expect(html(rendered[0])).toContain("settings broken");
    expect(html(rendered[0])).not.toContain("OBS browser source");
  });

  it("keeps the panel, and reloads nothing, when the host refused the link, in OBS too", () => {
    for (const obs of [true, false]) {
      const { env, rendered, reasons } = fakeEnv({ obs });
      expect(handleStartupFailure(new MirrorRefusedError(), env)).toBe("panel");
      expect(reasons).toEqual([]);
      expect(html(rendered[0])).toContain("client.peer_invalid_key");
      expect(html(rendered[0])).toContain("192.168.1.20:3030");
    }
  });

  it("keeps the panel for a failure no reload can fix (no WebRTC, a bad host or port), in OBS too", () => {
    for (const obs of [true, false]) {
      const { env, rendered, reasons } = fakeEnv({ obs });
      const error = new MirrorStartError(new Error("Your browser does not support some WebRTC features"));
      expect(handleStartupFailure(error, env)).toBe("panel");
      expect(reasons).toEqual([]);
      expect(rendered).toHaveLength(1);
      const shown = html(rendered[0]);
      expect(shown).toContain("Your browser does not support some WebRTC features");
      expect(shown).toContain("check that nothing blocks WebSockets");
      expect(shown).not.toContain("reloads itself");
    }
  });

  it("stays transparent and reloads itself in OBS when the failure may pass", () => {
    const { env, rendered, reasons } = fakeEnv({ obs: true });
    expect(handleStartupFailure(new Error("Failed to fetch dynamically imported module"), env)).toBe("blank");
    expect(rendered).toEqual([null]);
    expect(reasons).toEqual(["the mirror failed to start"]);
  });

  it("keeps the panel in an ordinary browser while it reloads itself, and says when", () => {
    const { env, rendered, reasons } = fakeEnv({ obs: false });
    expect(handleStartupFailure(new Error("Failed to fetch dynamically imported module"), env)).toBe("panel");
    expect(reasons).toEqual(["the mirror failed to start"]);
    const shown = html(rendered[0]);
    expect(shown).toContain("Failed to fetch dynamically imported module");
    expect(shown).toContain("tries again in 4 s");

    // A reload that is already pending has its own panel: no promise of a second one.
    const pending = fakeEnv({ scheduleReload: () => null });
    handleStartupFailure(new Error("again"), pending.env);
    expect(html(pending.rendered[0])).not.toContain("tries again");
  });
});

describe("StartupErrorView", () => {
  it("renders every dynamic value as text", () => {
    const shown = html(
      createElement(StartupErrorView, {
        error: new Error("<img src=x onerror=alert(1)>"),
        client: { host: "<b>evil</b>", port: "3030" },
      }),
    );
    expect(shown).not.toContain("<img");
    expect(shown).not.toContain("<b>evil");
    expect(shown).toContain("&lt;img");
  });
});
