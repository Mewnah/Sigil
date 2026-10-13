import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The Google Fonts list comes from the Rust `google-fonts` plugin (the API key is compiled into the app and
 * never in this bundle). Only the host can ask; the list is kept for the life of the page.
 */

const m = vi.hoisted(() => ({
  fetchGoogleFontsList: vi.fn(),
  toastError: vi.fn(),
  isServer: vi.fn(() => true),
  fetch: vi.fn(),
}));

vi.mock("@/utils/googleFonts", () => ({ fetchGoogleFontsList: m.fetchGoogleFontsList }));
vi.mock("react-toastify", () => ({ toast: { error: m.toastError, promise: vi.fn() } }));
vi.mock("browser-fs-access", () => ({ fileOpen: vi.fn() }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ save: vi.fn(), open: vi.fn() }));
vi.mock("@tauri-apps/plugin-fs", () => ({ readFile: vi.fn() }));

import Service_Files from "../index";

let service: Service_Files;

beforeEach(() => {
  vi.stubGlobal("window", { Config: { isServer: () => m.isServer() } });
  vi.stubGlobal("fetch", m.fetch);
  m.isServer.mockReturnValue(true);
  service = new Service_Files();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Service_Files font list", () => {
  it("loads the list through Rust once and reuses it", async () => {
    m.fetchGoogleFontsList.mockResolvedValue([{ family: "Roboto", variants: ["regular"], files: {} }]);
    await service.installFont("Not A Real Family");
    await service.installFont("Still Not Real");
    expect(m.fetchGoogleFontsList).toHaveBeenCalledTimes(1);
    expect(m.toastError).not.toHaveBeenCalled();
  });

  it("never calls the Google Fonts API from the page (no key in the bundle to send)", async () => {
    m.fetchGoogleFontsList.mockResolvedValue([]);
    await service.installFont("Roboto");
    expect(m.fetch).not.toHaveBeenCalled();
  });

  it("tells the user why the list could not be loaded, and retries on the next try", async () => {
    m.fetchGoogleFontsList.mockRejectedValueOnce("The Google Fonts list is not available in this build");
    await service.installFont("Roboto");
    expect(m.toastError).toHaveBeenCalledWith(
      "Could not load the Google Fonts list: The Google Fonts list is not available in this build",
    );

    m.toastError.mockClear();
    m.fetchGoogleFontsList.mockResolvedValue([{ family: "Roboto", variants: ["regular"], files: {} }]);
    await service.installFont("Not A Real Family");
    expect(m.fetchGoogleFontsList).toHaveBeenCalledTimes(2);
    expect(m.toastError).not.toHaveBeenCalled();
  });

  it("a client (no Tauri, no key) does not try", async () => {
    m.isServer.mockReturnValue(false);
    await service.installFont("Roboto");
    expect(m.fetchGoogleFontsList).not.toHaveBeenCalled();
    expect(m.toastError).toHaveBeenCalledWith(expect.stringContaining("Sigil app"));
  });
});
