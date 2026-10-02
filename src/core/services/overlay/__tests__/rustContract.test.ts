import { describe, expect, it } from "vitest";
import settingsRs from "../../../../../src-tauri/src/services/vr_overlay/settings.rs?raw";
import typesRs from "../../../../../src-tauri/src/services/vr_overlay/types.rs?raw";
import captionWindowRs from "../../../../../src-tauri/src/services/caption_window.rs?raw";
import vrCommands from "../../../../../src-tauri/plugin-commands/vr-overlay.txt?raw";
import windowCommands from "../../../../../src-tauri/plugin-commands/caption-window.txt?raw";
import overlayIndexTs from "../index.ts?raw";
import overlayDriversTs from "../drivers.ts?raw";
import { CAPTION_WINDOW, VR_OVERLAY_STATE_EVENT } from "@/types";
import { VRC_OVERLAY_ATTACH } from "../../vrc/schema";
import { VR_DEFAULTS, VR_LIMITS, VR_OVERLAY_REASONS } from "../config";

/** Every source file of the overlay service that calls the plugins. */
const overlayServiceTs = overlayIndexTs + overlayDriversTs;

/** The TS side of the overlay must agree with Rust on ranges, defaults, names and commands (P6-B). */

function rustPair(name: string): [number, number] {
  const m = settingsRs.match(new RegExp(`const ${name}: \\((?:f32|u32), (?:f32|u32)\\) = \\(([-\\d.]+), ([-\\d.]+)\\);`));
  expect(m, `settings.rs ${name}`).not.toBeNull();
  return [Number(m![1]), Number(m![2])];
}

function rustStr(src: string, name: string): string {
  const m = src.match(new RegExp(`pub const ${name}: &str = "([^"]*)";`));
  expect(m, name).not.toBeNull();
  return m![1];
}

function camel(variant: string): string {
  return variant[0].toLowerCase() + variant.slice(1);
}

function enumVariants(src: string, name: string): string[] {
  const body = src.match(new RegExp(`pub enum ${name} \\{([\\s\\S]*?)\\n\\}`));
  expect(body, name).not.toBeNull();
  return [...body![1].matchAll(/^\s+([A-Z][A-Za-z]+),?$/gm)].map((m) => camel(m[1]));
}

function commands(txt: string): string[] {
  return txt.split(/\r?\n/).map((l) => l.replace(/#.*/, "").trim()).filter(Boolean);
}

describe("vr-overlay contract", () => {
  it("clamping ranges match vr_overlay::settings", () => {
    expect(rustPair("FONT_PX")).toEqual([...VR_LIMITS.fontSizePx]);
    expect(rustPair("MAX_LINES")).toEqual([...VR_LIMITS.maxLines]);
    expect(rustPair("WIDTH_M")).toEqual([...VR_LIMITS.widthM]);
    expect(rustPair("DISTANCE_M")).toEqual([...VR_LIMITS.distanceM]);
    expect(rustPair("OFFSET_Y_M")).toEqual([...VR_LIMITS.offsetYM]);
    expect(rustPair("OPACITY")).toEqual([...VR_LIMITS.opacity]);
  });

  it("defaults match VrOverlayConfig::default()", () => {
    const body = typesRs.match(/impl Default for VrOverlayConfig \{[\s\S]*?Self \{([\s\S]*?)\n\s+\}/)![1];
    const num = (rust: string) => Number(body.match(new RegExp(`${rust}: ([-\\d.]+),`))![1]);
    const str = (rust: string) => body.match(new RegExp(`${rust}: "([^"]*)"\\.into\\(\\),`))![1];
    expect(num("width_m")).toBe(VR_DEFAULTS.widthM);
    expect(num("distance_m")).toBe(VR_DEFAULTS.distanceM);
    expect(num("offset_y_m")).toBe(VR_DEFAULTS.offsetYM);
    expect(num("font_size_px")).toBe(VR_DEFAULTS.fontSizePx);
    expect(num("opacity")).toBe(VR_DEFAULTS.opacity);
    expect(num("max_lines")).toBe(VR_DEFAULTS.maxLines);
    expect(str("text_color")).toBe(VR_DEFAULTS.textColor);
    expect(str("background_color")).toBe(VR_DEFAULTS.backgroundColor);
    expect(body).toContain("attach: VrOverlayAttach::Head,");
    expect(VR_DEFAULTS.attach).toBe("head");
  });

  it("enums match their Rust wire names", () => {
    expect(enumVariants(typesRs, "VrOverlayReason")).toEqual([...VR_OVERLAY_REASONS]);
    expect(enumVariants(typesRs, "VrOverlayAttach")).toEqual([...VRC_OVERLAY_ATTACH]);
    expect(enumVariants(typesRs, "VrOverlayState")).toEqual(["off", "waiting", "running", "unavailable", "error"]);
  });

  it("names and every command the service invokes are registered", () => {
    expect(VR_OVERLAY_STATE_EVENT).toBe("vr_overlay:state");
    const vr = commands(vrCommands);
    const win = commands(windowCommands);
    const invoked = [...overlayServiceTs.matchAll(/plugin:(vr-overlay|caption-window)\|([a-z_]+)/g)].map((m) => [m[1], m[2]] as const);
    // Commands built from a name (`#call("caption_window_open", ...)`).
    const called = [...overlayServiceTs.matchAll(/#(?:call|status)\("([a-z_]+)"/g)].map((m) => m[1]);
    expect(invoked.length + called.length).toBeGreaterThan(5);
    for (const [plugin, cmd] of invoked) expect(plugin === "vr-overlay" ? vr : win).toContain(cmd);
    for (const cmd of called) expect([...vr, ...win]).toContain(cmd);
  });
});

describe("caption-window contract", () => {
  it("label, route and event match caption_window.rs", () => {
    expect(rustStr(captionWindowRs, "CAPTIONS_WINDOW_LABEL")).toBe(CAPTION_WINDOW.label);
    expect(rustStr(captionWindowRs, "CAPTIONS_ROUTE")).toBe(CAPTION_WINDOW.route);
    expect(rustStr(captionWindowRs, "STATE_EVENT")).toBe(CAPTION_WINDOW.stateEvent);
  });
});
