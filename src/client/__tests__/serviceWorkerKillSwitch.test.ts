import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { afterEach, describe, expect, it, vi } from "vitest";

/**
 * `public/sw.js` is a kill switch for the caching service worker that Curses and early Sigil builds
 * registered at `/sw.js` (an OBS browser source keeps it between runs and then keeps showing the old
 * page). It must retire that worker, and nothing in Sigil may register it. The web server serves it
 * as JavaScript (`src-tauri/src/services/web/assets.rs`, tested there).
 *
 * The worker runs here in a fake ServiceWorkerGlobalScope that follows two rules a real browser
 * enforces and the first version of this test did not model:
 *  - `clients.matchAll()` lists only the windows this registration controls unless
 *    `includeUncontrolled` is set, and `WindowClient.navigate()` rejects (TypeError) for a window
 *    the calling worker does not control;
 *  - an event stays open until every promise given to `waitUntil` settles, so a promise that never
 *    settles (a stalled navigation, a stuck Cache Storage call) would hold install/activate open.
 * What a browser does with the real Curses page and worker is checked by
 * `scripts/check-sw-kill-switch.mjs` (`pnpm check:sw-kill`, opt-in: it needs the Playwright Chromium).
 */

const ROOT = path.resolve(__dirname, "../../..");
const SW_SOURCE = fs.readFileSync(path.join(ROOT, "public/sw.js"), "utf8");
/** The worker without its comments (the header explains what it deliberately does not call). */
const SW_CODE = SW_SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

type Listener = (event: { waitUntil(promise: Promise<unknown>): void }) => void;

interface FakeWindowClient {
  url: string;
  /** Controlled by the old worker's registration, so by this worker once it is active. */
  controlled: boolean;
  navigate: (url: string) => Promise<unknown>;
}

interface WorkerOptions {
  caches?: string[];
  clients?: Array<{ url: string; controlled: boolean }>;
  deleteFails?: boolean;
  navigateFails?: "reject" | "throw";
  navigateHangs?: boolean;
  keysHang?: boolean;
  matchAllHangs?: boolean;
  unregisterHangs?: boolean;
}

const NEVER = new Promise<never>(() => {});

/** Run `public/sw.js` in a fake ServiceWorkerGlobalScope and keep the order of every call it makes. */
function runWorker(options: WorkerOptions = {}, source: string = SW_SOURCE) {
  const calls: string[] = [];
  const listeners: Record<string, Listener> = {};
  const cacheNames = options.caches ?? ["workbox-precache-v2-http://localhost:3030/", "runtime"];
  const windows: FakeWindowClient[] = (
    options.clients ?? [
      { url: "http://localhost:3030/client?host=localhost&port=3030", controlled: true },
      { url: "http://localhost:3030/mic.html", controlled: true },
      { url: "http://localhost:3030/obs-captions.html", controlled: false },
    ]
  ).map((c) => ({
    ...c,
    navigate: (url: string): Promise<unknown> => {
      calls.push(`navigate ${url}`);
      // A client whose navigate() throws before it returns a promise.
      if (options.navigateFails === "throw") throw new TypeError("navigate failed (sync)");
      if (!c.controlled) return Promise.reject(new TypeError("only a client controlled by this worker can be navigated"));
      if (options.navigateFails === "reject") return Promise.reject(new TypeError("navigate failed"));
      if (options.navigateHangs) return NEVER;
      return Promise.resolve(undefined);
    },
  }));
  const matchAllArgs: unknown[] = [];
  const scope = {
    addEventListener: (type: string, listener: Listener) => {
      listeners[type] = listener;
    },
    skipWaiting: async () => void calls.push("skipWaiting"),
    clients: {
      claim: async () => void calls.push("claim"),
      matchAll: (args?: { type?: string; includeUncontrolled?: boolean }) => {
        matchAllArgs.push(args);
        calls.push("matchAll");
        if (options.matchAllHangs) return NEVER;
        return Promise.resolve(windows.filter((c) => c.controlled || args?.includeUncontrolled));
      },
    },
    registration: {
      unregister: () => {
        calls.push("unregister");
        return options.unregisterHangs ? NEVER : Promise.resolve(true);
      },
    },
  };
  const cachesApi = {
    keys: () => {
      calls.push("caches.keys");
      return options.keysHang ? NEVER : Promise.resolve(cacheNames);
    },
    delete: async (name: string) => {
      calls.push(`caches.delete ${name}`);
      if (options.deleteFails) throw new Error("delete failed");
      return true;
    },
  };
  // The worker's timeouts must not keep the test process alive, and must follow vi.useFakeTimers().
  const setTimeoutSandbox = (fn: () => void, ms: number) => {
    const handle = setTimeout(fn, ms) as unknown as { unref?: () => void };
    handle?.unref?.();
    return handle;
  };
  vm.runInNewContext(source, { self: scope, caches: cachesApi, Promise, setTimeout: setTimeoutSandbox });

  /** Dispatch an event and resolve when everything given to `waitUntil` has settled. */
  function fire(type: string): Promise<void> {
    const pending: Promise<unknown>[] = [];
    listeners[type]!({ waitUntil: (p) => void pending.push(p) });
    return Promise.all(pending).then(() => undefined);
  }
  return {
    calls,
    listeners,
    fire,
    navigations: () => calls.filter((c) => c.startsWith("navigate ")),
    matchAllArgs: () => matchAllArgs,
  };
}

const CLIENT_URL = "http://localhost:3030/client?host=localhost&port=3030";
const MIC_URL = "http://localhost:3030/mic.html";
const CAPTIONS_URL = "http://localhost:3030/obs-captions.html";

afterEach(() => {
  vi.useRealTimers();
});

describe("public/sw.js (kill switch)", () => {
  it("only listens for install and activate: it has no fetch handler and answers no request", () => {
    const { listeners } = runWorker();
    expect(Object.keys(listeners).sort()).toEqual(["activate", "install"]);
    expect(SW_SOURCE).not.toMatch(/["']fetch["']/);
    expect(SW_SOURCE).not.toMatch(/\bonfetch\b/);
    expect(SW_SOURCE).not.toMatch(/respondWith|importScripts/);
  });

  it("is plain JavaScript the resolver types by extension (it must not start like HTML or a shell script)", () => {
    expect(SW_SOURCE.trimStart().startsWith("/*") || SW_SOURCE.trimStart().startsWith("//")).toBe(true);
    expect(SW_SOURCE).not.toMatch(/^\s*(<|#!)/);
  });

  describe("install", () => {
    it("takes over at once and deletes every cache, so the old worker falls back to the network", async () => {
      const worker = runWorker();
      await worker.fire("install");
      expect([...worker.calls].sort()).toEqual(
        ["caches.delete runtime", "caches.delete workbox-precache-v2-http://localhost:3030/", "caches.keys", "skipWaiting"].sort(),
      );
    });

    it("never unregisters, claims or navigates (unregistering before activation would stop the takeover)", async () => {
      const worker = runWorker();
      await worker.fire("install");
      for (const forbidden of ["unregister", "claim", "matchAll"]) expect(worker.calls).not.toContain(forbidden);
      expect(worker.navigations()).toEqual([]);
    });

    it("still takes over when a cache cannot be deleted", async () => {
      const worker = runWorker({ deleteFails: true });
      await expect(worker.fire("install")).resolves.toBeUndefined();
      expect(worker.calls).toContain("skipWaiting");
    });

    it("is not held open by a Cache Storage call that never returns", async () => {
      vi.useFakeTimers();
      const worker = runWorker({ keysHang: true });
      let settled = false;
      const done = worker.fire("install").then(() => void (settled = true));
      await vi.advanceTimersByTimeAsync(1900);
      expect(settled).toBe(false);
      await vi.advanceTimersByTimeAsync(200);
      await done;
      expect(settled).toBe(true);
      expect(worker.calls).toContain("skipWaiting");
    });
  });

  describe("activate", () => {
    it("unregisters, then reloads each window it controls from the network, then sweeps the caches", async () => {
      const worker = runWorker({ caches: ["late-cache"] });
      await worker.fire("activate");
      expect(worker.calls).toEqual([
        "matchAll",
        "unregister",
        `navigate ${CLIENT_URL}`,
        `navigate ${MIC_URL}`,
        "caches.keys",
        "caches.delete late-cache",
      ]);
    });

    it("never touches a window the old worker did not control (no claim, no includeUncontrolled)", async () => {
      const worker = runWorker();
      await worker.fire("activate");
      expect(worker.navigations()).not.toContain(`navigate ${CAPTIONS_URL}`);
      expect(worker.calls).not.toContain("claim");
      expect(worker.matchAllArgs()).toEqual([{ type: "window" }]);
      expect(SW_CODE).not.toMatch(/includeUncontrolled|\.claim\s*\(/);
    });

    it("reloads each window to its own URL, so the query (host, port, token) is kept", async () => {
      const worker = runWorker();
      await worker.fire("activate");
      expect(worker.navigations()).toEqual([`navigate ${CLIENT_URL}`, `navigate ${MIC_URL}`]);
    });

    it("is not held open by a navigation that never completes", async () => {
      const worker = runWorker({ navigateHangs: true });
      await expect(worker.fire("activate")).resolves.toBeUndefined();
      expect(worker.navigations()).toHaveLength(2);
    });

    it("does not fail when a window cannot be navigated, and still reloads the others", async () => {
      for (const navigateFails of ["reject", "throw"] as const) {
        const worker = runWorker({ navigateFails });
        await expect(worker.fire("activate")).resolves.toBeUndefined();
        expect(worker.calls).toContain("unregister");
        expect(worker.navigations()).toHaveLength(2);
      }
    });

    it("unregisters and reloads before the cache sweep, which cannot hold them back", async () => {
      const worker = runWorker({ deleteFails: true, caches: ["a", "b"] });
      await worker.fire("activate");
      const at = (call: string) => worker.calls.findIndex((c) => c.startsWith(call));
      expect(at("unregister")).toBeLessThan(at("navigate"));
      expect(at("navigate")).toBeLessThan(at("caches.keys"));
      expect(worker.calls).toContain("caches.delete a");
    });

    it("is not held open by a Cache Storage sweep that never returns", async () => {
      vi.useFakeTimers();
      const worker = runWorker({ keysHang: true });
      const done = worker.fire("activate");
      await vi.advanceTimersByTimeAsync(2100);
      await done;
      expect(worker.calls).toContain("unregister");
      expect(worker.navigations()).toHaveLength(2);
    });

    it("still reloads the windows when unregister never returns", async () => {
      vi.useFakeTimers();
      const worker = runWorker({ unregisterHangs: true });
      const done = worker.fire("activate");
      await vi.advanceTimersByTimeAsync(4200);
      await done;
      expect(worker.navigations()).toHaveLength(2);
    });

    it("still unregisters when the window list never returns", async () => {
      vi.useFakeTimers();
      const worker = runWorker({ matchAllHangs: true });
      const done = worker.fire("activate");
      await vi.advanceTimersByTimeAsync(4200);
      await done;
      expect(worker.calls).toContain("unregister");
      expect(worker.navigations()).toEqual([]);
    });

    it("works with no cache and no open page", async () => {
      const worker = runWorker({ caches: [], clients: [] });
      await worker.fire("activate");
      expect(worker.calls).toEqual(["matchAll", "unregister", "caches.keys"]);
    });
  });
});

/** The first version of the worker (claim, includeUncontrolled, navigations awaited inside waitUntil). */
const FIRST_VERSION = `
self.addEventListener("install", function (event) { event.waitUntil(self.skipWaiting()); });
self.addEventListener("activate", function (event) {
  event.waitUntil(
    self.clients.claim()
      .then(function () { return caches.keys(); })
      .then(function (keys) { return Promise.all(keys.map(function (key) { return caches.delete(key); })); })
      .catch(function () {})
      .then(function () { return self.registration.unregister(); })
      .catch(function () {})
      .then(function () { return self.clients.matchAll({ type: "window", includeUncontrolled: true }); })
      .then(function (windows) {
        return Promise.all(windows.map(function (client) { return client.navigate(client.url).catch(function () {}); }));
      })
      .catch(function () {}),
  );
});
`;

describe("the fake worker scope has teeth", () => {
  it("catches the first version of the worker: it reloads a window it never controlled", async () => {
    const worker = runWorker({}, FIRST_VERSION);
    await worker.fire("activate");
    expect(worker.calls).toContain("claim");
    expect(worker.navigations()).toContain(`navigate ${CAPTIONS_URL}`);
  });

  it("catches the first version of the worker: one stalled navigation holds the activate event open", async () => {
    const worker = runWorker({ navigateHangs: true }, FIRST_VERSION);
    let settled = false;
    void worker.fire("activate").then(() => void (settled = true));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);
  });
});

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__tests__" || entry.name === "node_modules") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) sourceFiles(full, out);
    else if (/\.(ts|tsx|js|jsx|mjs|html)$/.test(entry.name)) out.push(full);
  }
  return out;
}

describe("nothing in Sigil registers or loads the kill switch", () => {
  const files = [
    ...sourceFiles(path.join(ROOT, "src")),
    ...fs
      .readdirSync(path.join(ROOT, "public"))
      .filter((name) => /\.(html|js)$/.test(name) && name !== "sw.js")
      .map((name) => path.join(ROOT, "public", name)),
    path.join(ROOT, "index.html"),
    path.join(ROOT, "vite.config.ts"),
  ];

  it("scans the host app, /client and every page we ship", () => {
    expect(files.length).toBeGreaterThan(50);
    expect(files.some((f) => f.endsWith(path.join("src", "index.tsx")))).toBe(true);
  });

  it("never calls serviceWorker.register", () => {
    const offenders = files.filter((f) => /serviceWorker\s*\??\.\s*register\s*\(/.test(fs.readFileSync(f, "utf8")));
    expect(offenders).toEqual([]);
  });

  it("never names /sw.js as a script, a registration or an import", () => {
    const offenders = files.filter((f) => /["'`(]\/?sw\.js\b/.test(fs.readFileSync(f, "utf8")));
    expect(offenders).toEqual([]);
  });

  it("does not build with a PWA plugin that would generate a worker of its own", () => {
    expect(fs.readFileSync(path.join(ROOT, "vite.config.ts"), "utf8")).not.toMatch(/vite-plugin-pwa|VitePWA|workbox/i);
    const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, "package.json"), "utf8"));
    const deps = { ...pkg.dependencies, ...pkg.devDependencies };
    expect(Object.keys(deps).filter((name) => /pwa|workbox/i.test(name))).toEqual([]);
  });

  it("keeps the in-page cleanup on /client (it unregisters, never registers)", () => {
    const index = fs.readFileSync(path.join(ROOT, "src/index.tsx"), "utf8");
    expect(index).toContain("serviceWorker.getRegistrations()");
    expect(index).toContain("reg.unregister()");
  });
});
