import { beforeEach, describe, expect, it, vi } from "vitest";

import Service_Scenes from "..";

/**
 * Deleting the scene that is on air switches the host back to "main" locally. A /client mirror
 * showing the deleted scene would draw nothing (no element has that scene any more): the switch is
 * published so the mirrors follow.
 */

let publish: ReturnType<typeof vi.fn>;
let patched: { scenes: Record<string, unknown>; elements: Record<string, { scenes: Record<string, unknown> }> };

beforeEach(() => {
  publish = vi.fn();
  patched = { scenes: { main: {}, intro: {} }, elements: { text1: { scenes: { main: {}, intro: {} } } } };
  (globalThis as any).window = {
    ApiShared: { pubsub: { publish } },
    ApiClient: { document: { patch: (fn: (state: typeof patched) => void) => fn(patched) } },
  };
});

describe("Service_Scenes.deleteScene", () => {
  it("tells the mirrors when the scene on air is deleted", () => {
    const scenes = new Service_Scenes();
    scenes.state.activeScene = "intro";
    scenes.deleteScene("intro");
    expect(scenes.state.activeScene).toBe("main");
    expect(publish).toHaveBeenCalledExactlyOnceWith("scenes:change", "main");
    expect(patched.scenes).toEqual({ main: {} });
    expect(patched.elements.text1!.scenes).toEqual({ main: {} });
  });

  it("publishes nothing when another scene is deleted", () => {
    const scenes = new Service_Scenes();
    scenes.state.activeScene = "main";
    scenes.deleteScene("intro");
    expect(scenes.state.activeScene).toBe("main");
    expect(publish).not.toHaveBeenCalled();
    expect(patched.scenes).toEqual({ main: {} });
  });
});
