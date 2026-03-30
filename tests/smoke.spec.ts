import { test, expect } from "@playwright/test";

/**
 * Validates `npm run build` output via `vite preview`.
 * Does not run Tauri; full host flows need the desktop app.
 */
const routes = [
  ["index", "/"],
  ["client shell", "/client?host=127.0.0.1&port=59999&id=server"],
] as const;

test.describe("preview smoke", () => {
  for (const [label, path] of routes) {
    test(`${label} mounts #root`, async ({ page }) => {
      await page.goto(path, { waitUntil: "domcontentloaded" });
      await expect(page.locator("#root")).toBeAttached();
    });
  }
});
