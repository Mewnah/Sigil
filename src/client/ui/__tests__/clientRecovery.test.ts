import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { ErrorBoundary } from "react-error-boundary";

/**
 * The /client page (an OBS browser source nobody refreshes) shows nothing and reloads itself when
 * its React tree throws, and reloads are rate limited so a failure that survives a reload cannot
 * spin. A reload only happens while the page's own address answers: otherwise it would land on the
 * browser's error page, which has no script left to try again.
 */

import {
  ClientErrorBoundary,
  RELOAD_BASE_DELAY_MS,
  RELOAD_MAX_DELAY_MS,
  RELOAD_PROBE_RETRY_MS,
  RELOAD_UNCOUNTED_DELAY_MS,
  RELOAD_WINDOW_MS,
  reloadDelayMs,
  scheduleClientReload,
  type ReloadEnv,
} from "../clientRecovery";

class MemoryStorage {
  data = new Map<string, string>();
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.data.set(key, value);
  }
}

type Timer = { id?: number; fn: () => void; ms: number };

/** `window.setTimeout` / `clearTimeout` that only record the timers (the probe also sets its own deadline). */
function fakeTimers(timers: Timer[]) {
  let next = 1;
  return {
    setTimeout: (fn: () => void, ms: number) => {
      const id = next++;
      timers.push({ id, fn, ms });
      return id;
    },
    clearTimeout: (id: number) => {
      const index = timers.findIndex((t) => t.id === id);
      if (index >= 0) timers.splice(index, 1);
    },
  };
}

/** Let the promise chains of the probe settle. */
const flush = () => new Promise<void>((resolve) => setImmediate(resolve));

function env(overrides: Partial<ReloadEnv> = {}) {
  const timers: Timer[] = [];
  const storage = new MemoryStorage();
  let now = 1_000_000;
  const reload = vi.fn();
  const reachable = vi.fn(async () => true);
  const base: ReloadEnv = {
    now: () => now,
    storage,
    setTimeout: (fn: () => void, ms: number) => timers.push({ fn, ms }),
    reload,
    reachable,
  };
  return {
    ...base,
    ...overrides,
    timers,
    /** The in-memory storage (the env's own `storage` may be overridden). */
    memory: storage,
    reload,
    reachable,
    advance: (ms: number) => void (now += ms),
    /** The oldest pending timer fires and the probe it starts settles. */
    fire: async () => {
      timers.shift()!.fn();
      await flush();
    },
  };
}

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => {});
});

describe("reloadDelayMs", () => {
  it("doubles for every reload in the window and is capped", () => {
    expect([0, 1, 2, 3, 4, 5, 6, 40].map(reloadDelayMs)).toEqual([4_000, 8_000, 16_000, 32_000, 64_000, 120_000, 120_000, 120_000]);
    expect(RELOAD_BASE_DELAY_MS).toBe(4_000);
    expect(RELOAD_MAX_DELAY_MS).toBe(120_000);
  });

  it("is slow, not instant, when the reloads cannot be counted", () => {
    expect(reloadDelayMs(null)).toBe(RELOAD_UNCOUNTED_DELAY_MS);
  });
});

describe("scheduleClientReload", () => {
  it("reloads after a few seconds, and each further reload in the window waits longer", async () => {
    const e = env();
    const delays: (number | null)[] = [];
    for (let i = 0; i < 4; i++) {
      delays.push(scheduleClientReload("test", new Error("boom"), e));
      expect(e.reload).toHaveBeenCalledTimes(i);
      e.advance(delays[i]!);
      await e.fire();
      expect(e.reload).toHaveBeenCalledTimes(i + 1);
    }
    expect(delays).toEqual([4_000, 8_000, 16_000, 32_000]);
  });

  it("forgets reloads older than the window", async () => {
    const e = env();
    for (let i = 0; i < 3; i++) {
      scheduleClientReload("test", undefined, e);
      await e.fire();
    }
    e.advance(RELOAD_WINDOW_MS);
    expect(scheduleClientReload("test", undefined, e)).toBe(4_000);
    await e.fire();
  });

  it("schedules one reload at a time", async () => {
    const e = env();
    expect(scheduleClientReload("first", undefined, e)).toBe(4_000);
    expect(scheduleClientReload("second", undefined, e)).toBeNull();
    expect(e.timers).toHaveLength(1);
    await e.fire();
    expect(scheduleClientReload("third", undefined, e)).toBe(8_000);
    await e.fire();
  });

  it("waits a fixed, slow time when session storage is refused or damaged", async () => {
    const refused = env({ storage: null });
    expect(scheduleClientReload("test", undefined, refused)).toBe(RELOAD_UNCOUNTED_DELAY_MS);
    await refused.fire();

    const broken = env();
    broken.memory.setItem("sigil.client.reloads", "{not json");
    expect(scheduleClientReload("test", undefined, broken)).toBe(RELOAD_UNCOUNTED_DELAY_MS);
    await broken.fire();

    const throws = env({
      storage: {
        getItem: () => {
          throw new Error("blocked");
        },
        setItem: () => {},
      },
    });
    expect(scheduleClientReload("test", undefined, throws)).toBe(RELOAD_UNCOUNTED_DELAY_MS);
    await throws.fire();

    // Nonsense entries (future times, wrong types) do not count.
    const odd = env();
    odd.memory.setItem("sigil.client.reloads", JSON.stringify([2_000_000, "x", null, 999_000]));
    expect(scheduleClientReload("test", undefined, odd)).toBe(8_000);
    await odd.fire();
  });

  it("is just as slow when reads work but writes are refused (a full quota), however often it loads", async () => {
    // Every load of the page would read "no reloads yet" and wait the shortest time again, for ever.
    const setItem = vi.fn(() => {
      throw new DOMException("quota", "QuotaExceededError");
    });
    const e = env({ storage: { getItem: () => null, setItem } });
    for (let i = 0; i < 3; i++) {
      expect(scheduleClientReload("test", undefined, e)).toBe(RELOAD_UNCOUNTED_DELAY_MS);
      await e.fire();
    }
    expect(setItem).toHaveBeenCalledTimes(3);

    // Never faster than what the counted reloads already asked for.
    const many = env({
      storage: {
        getItem: () => JSON.stringify([999_000, 999_100, 999_200, 999_300, 999_400]),
        setItem,
      },
    });
    expect(scheduleClientReload("test", undefined, many)).toBe(RELOAD_MAX_DELAY_MS);
    await many.fire();
  });

  it("reloads only once the page's own address answers, and keeps probing until then", async () => {
    // Sigil is restarting when the timer fires: a reload now would land on the browser's error page.
    const e = env();
    e.reachable.mockResolvedValueOnce(false).mockRejectedValueOnce(new Error("offline")).mockResolvedValue(true);
    scheduleClientReload("test", undefined, e);
    await e.fire();
    expect(e.reload).not.toHaveBeenCalled();
    expect(e.timers.map((t) => t.ms)).toEqual([RELOAD_PROBE_RETRY_MS]);

    await e.fire();
    expect(e.reload).not.toHaveBeenCalled();
    expect(e.timers.map((t) => t.ms)).toEqual([RELOAD_PROBE_RETRY_MS]);

    await e.fire();
    expect(e.reachable).toHaveBeenCalledTimes(3);
    expect(e.reload).toHaveBeenCalledTimes(1);
    expect(e.timers).toHaveLength(0);

    // While it waited, no second reload could be queued; afterwards the next one can.
    expect(scheduleClientReload("again", undefined, e)).toBe(8_000);
    await e.fire();
  });

  it("holds the pending flag while it waits for the address", async () => {
    const e = env();
    e.reachable.mockResolvedValue(false);
    scheduleClientReload("test", undefined, e);
    await e.fire();
    expect(scheduleClientReload("second", undefined, e)).toBeNull();
    e.reachable.mockResolvedValue(true);
    await e.fire();
    expect(e.reload).toHaveBeenCalledTimes(1);
  });
});

describe("ClientErrorBoundary", () => {
  /**
   * There is no DOM or test renderer here, so React's own part is played by hand: the boundary
   * is the real react-error-boundary class, built from the props `ClientErrorBoundary` gives it,
   * and the lifecycle React runs when a child throws (getDerivedStateFromError, then
   * componentDidCatch, then render with the new state) is called in that order.
   */
  function crash(element: ReactElement, error: Error) {
    const Boundary = element.type as typeof ErrorBoundary;
    const boundary = new Boundary(element.props);
    expect(boundary.render()).toBe(element.props.children); // healthy: it renders its children
    boundary.state = { ...boundary.state, ...Boundary.getDerivedStateFromError(error) };
    boundary.componentDidCatch(error, { componentStack: "at Element (src/client/ui/element-instance.tsx)" });
    return boundary.render();
  }

  it("is the real react-error-boundary", () => {
    const child = { type: "span", props: {} } as unknown as ReactElement;
    const element = ClientErrorBoundary({ children: child }) as ReactElement;
    expect(element.type).toBe(ErrorBoundary);
  });

  it("renders nothing on stream and reloads the page when the tree throws", async () => {
    const reload = vi.fn();
    const timers: Timer[] = [];
    const fetched: string[] = [];
    vi.stubGlobal("window", {
      sessionStorage: new MemoryStorage(),
      ...fakeTimers(timers),
      location: { reload, href: "http://localhost:3030/client?token=t" },
    });
    vi.stubGlobal("fetch", async (url: string) => {
      fetched.push(url);
      return { ok: true };
    });
    try {
      const child = { type: "span", props: {} } as unknown as ReactElement;
      const element = ClientErrorBoundary({ children: child }) as ReactElement;
      // No panel, no text: the source stays transparent while it waits.
      const rendered = crash(element, new Error("Rendered more hooks than during the previous render."));
      expect(rendered).toBeNull();
      expect(timers).toHaveLength(1);
      expect(timers[0]!.ms).toBe(4_000);
      expect(reload).not.toHaveBeenCalled();

      timers[0]!.fn();
      await flush();
      // It looked at its own address (what a reload would load) first, then reloaded.
      expect(fetched).toEqual(["http://localhost:3030/client?token=t"]);
      expect(reload).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("does not reload while the address does not answer", async () => {
    const reload = vi.fn();
    const timers: Timer[] = [];
    vi.stubGlobal("window", {
      sessionStorage: new MemoryStorage(),
      ...fakeTimers(timers),
      location: { reload, href: "http://localhost:3030/client" },
    });
    let up = false;
    vi.stubGlobal("fetch", async () => {
      if (!up) throw new TypeError("Failed to fetch");
      return { ok: true };
    });
    try {
      const child = { type: "span", props: {} } as unknown as ReactElement;
      crash(ClientErrorBoundary({ children: child }) as ReactElement, new Error("boom"));
      timers.shift()!.fn();
      await flush();
      expect(reload).not.toHaveBeenCalled();
      // It tries again a few seconds later instead of giving up, and reloads once Sigil answers.
      expect(timers.map((t) => t.ms)).toEqual([RELOAD_PROBE_RETRY_MS]);
      up = true;
      timers.shift()!.fn();
      await flush();
      expect(reload).toHaveBeenCalledTimes(1);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
