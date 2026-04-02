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
