import { test, expect } from "@playwright/test";

/**
 * Ensures the production bundle is served and the HTML shell mounts #root.
 * Full host UI requires Tauri (invoke); this smoke only validates the Vite build + preview pipeline.
 */
test.describe("preview smoke", () => {
  test("index serves and contains root mount", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("#root")).toBeAttached();
  });
});
