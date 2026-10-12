import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * The /client loading view stays up for as long as the first connection takes (the page retries
 * inside itself), so after a few failed attempts it says what may be wrong instead of pulsing
 * "Connecting" with no clue. The background stays transparent either way.
 */

vi.mock("@/core/ui/logo", () => ({ default: () => null }));
vi.mock("../loading.module.css", () => ({ default: { screen: "screen" } }));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string>) => (options ? `${key}(${Object.values(options).join("|")})` : key),
  }),
}));

import ClientLoadingView, { LOADING_HINT_AFTER_FAILURES } from "../view_loading";

function render(state: { phase: string; lastError: string; failures: number }) {
  vi.stubGlobal("window", {
    ApiShared: { peer: { clientMirrorState: proxy(state) } },
    Config: { clientNetwork: { host: "192.168.1.20", port: "3030" } },
  });
  return renderToStaticMarkup(createElement(ClientLoadingView));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("ClientLoadingView", () => {
  it("only says it is connecting while the first attempts run", () => {
    const shown = render({ phase: "connecting", lastError: "", failures: LOADING_HINT_AFTER_FAILURES - 1 });
    expect(shown).toContain("client.loading_connecting");
    expect(shown).not.toContain("loading_still_trying");
  });

  it("names the address and the last error once the attempts keep failing", () => {
    const shown = render({ phase: "reconnecting", lastError: "network", failures: LOADING_HINT_AFTER_FAILURES });
    expect(shown).toContain("client.loading_still_trying(192.168.1.20:3030)");
    expect(shown).toContain("client.peer_error_detail(network)");
    // Still the transparent source: nothing paints an opaque background over the stream.
    expect(shown).toContain("background-color:transparent");
  });

  it("does not make up an error it does not have", () => {
    const shown = render({ phase: "reconnecting", lastError: "", failures: 9 });
    expect(shown).toContain("loading_still_trying");
    expect(shown).not.toContain("peer_error_detail");
  });
});
