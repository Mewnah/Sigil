import { test, expect } from "@playwright/test";

/**
 * Validates `pnpm run build` output via `vite preview`.
 * Does not run Tauri; full host flows need the desktop app.
 */
const routes = [
  ["index", "/"],
  ["client shell", "/client"],
] as const;

test.describe("preview smoke", () => {
  for (const [label, path] of routes) {
    test(`${label} mounts #root`, async ({ page }) => {
      await page.goto(path, { waitUntil: "domcontentloaded" });
      await expect(page.locator("#root")).toBeAttached();
    });
  }
});

/** RTL locale bundles must be served (regression guard for `ar` / `ur`). */
test.describe("i18n RTL assets", () => {
  for (const code of ["ar", "ur"] as const) {
    test(`${code} translation.json is served`, async ({ request }) => {
      const res = await request.get(`/i18n/${code}/translation.json`);
      expect(res.ok()).toBeTruthy();
      const json = await res.json();
      expect(typeof json).toBe("object");
    });
  }
});

/** Minimal layout sanity when `dir="rtl"` (mirroring smoke; not a substitute for full manual RTL QA). */
test("host shell still mounts under forced rtl on html", async ({ page }) => {
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => {
    document.documentElement.setAttribute("dir", "rtl");
    document.documentElement.setAttribute("lang", "ar");
  });
  await expect(page.locator("#root")).toBeAttached();
  const dir = await page.evaluate(() => document.documentElement.getAttribute("dir"));
  expect(dir).toBe("rtl");
});

/**
 * `/client` connects PeerJS only to the server that served it (or loopback): a website cannot open
 * `/client?host=evil.example` and feed the mirror from its own server (LAN security stage 1).
 * WebSocket is stubbed to record the URLs PeerJS opens; there is no host server in preview.
 */
declare global {
  interface Window {
    __wsUrls: string[];
  }
}

async function recordSockets(page: import("@playwright/test").Page) {
  await page.addInitScript(() => {
    window.__wsUrls = [];
    const Real = window.WebSocket;
    const Stub = function (url: string | URL, protocols?: string | string[]) {
      window.__wsUrls.push(String(url));
      return new Real("ws://127.0.0.1:1/blocked-by-test", protocols);
    } as unknown as typeof WebSocket;
    Object.assign(Stub, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
    Stub.prototype = Real.prototype;
    window.WebSocket = Stub;
  });
}

async function socketUrls(page: import("@playwright/test").Page): Promise<string[]> {
  await page.waitForFunction(() => window.__wsUrls.length > 0, null, { timeout: 20_000 });
  return page.evaluate(() => window.__wsUrls);
}

test.describe("client host pinning", () => {
  test("/client?host=<foreign> never connects to the foreign host", async ({ page }) => {
    await recordSockets(page);
    const requests: string[] = [];
    page.on("request", (r) => requests.push(r.url()));
    await page.goto("/client?host=evil.example&port=80&id=abc", { waitUntil: "domcontentloaded" });
    const servingPort = new URL(page.url()).port;
    const urls = await socketUrls(page);
    for (const u of urls) {
      expect(u).not.toContain("evil.example");
      // The PeerJS provider rewrites localhost to 127.0.0.1.
      expect(u).toMatch(new RegExp(`^wss?://(localhost|127\.0\.0\.1):${servingPort}/`));
    }
    expect(requests.filter((u) => new URL(u).hostname.endsWith("evil.example"))).toEqual([]);
    // The foreign parameters are gone from the URL; other parameters stay.
    expect(new URL(page.url()).search).toBe("?id=abc");
  });

  test("/client keeps a loopback ?host= (old Curses links, dev server)", async ({ page }) => {
    await recordSockets(page);
    await page.goto("/client?host=127.0.0.1&port=3030", { waitUntil: "domcontentloaded" });
    const urls = await socketUrls(page);
    expect(urls[0]).toMatch(/^wss?:\/\/127\.0\.0\.1:3030\//);
    expect(new URL(page.url()).search).toBe("?host=127.0.0.1&port=3030");
  });

  test("/client ignores a port that would inject a URL authority", async ({ page }) => {
    await recordSockets(page);
    await page.goto(`/client?port=${encodeURIComponent("3030@evil.example")}`, { waitUntil: "domcontentloaded" });
    const urls = await socketUrls(page);
    for (const u of urls) expect(u).not.toContain("evil.example");
  });
});

/**
 * P6-B early route: outside Tauri `/captions` renders nothing and never boots the host (no settings, no PeerJS), so the
 * caption window's URL served over the LAN is inert.
 */
test("captions route renders nothing and boots no host", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("/captions", { waitUntil: "load" });
  await expect(page.locator("html")).toHaveAttribute("data-sigil-captions", "1");
  await page.waitForTimeout(1500);
  expect(await page.title()).toBe("Sigil captions");
  expect(await page.locator("#root").innerHTML()).toBe("");
  const booted = await page.evaluate(() => {
    const w = window as unknown as Record<string, unknown>;
    return {
      server: typeof w.ApiServer,
      shared: typeof w.ApiShared,
      config: typeof w.Config,
      theme: document.documentElement.getAttribute("data-theme"),
      bg: getComputedStyle(document.body).backgroundColor,
    };
  });
  expect(booted).toEqual({ server: "undefined", shared: "undefined", config: "undefined", theme: null, bg: "rgba(0, 0, 0, 0)" });
  expect(errors).toEqual([]);
});
