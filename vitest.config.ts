import { defineConfig } from "vitest/config";
import * as path from "path";

/**
 * Unit tests for pure logic (`pnpm test:unit`). Separate from vite.config.ts so tests never load the
 * app's build plugins. Playwright smoke tests live in tests/ and are not collected here.
 *
 * Conventions:
 * - Put tests next to the code as `__tests__/<name>.test.ts` (TS under src/) or `*.test.mjs` for
 *   Node scripts under scripts/.
 * - Default environment is `node`: keep tested modules free of `window`, Tauri `invoke` and DOM, or
 *   opt a single file in with a `// @vitest-environment jsdom` pragma after adding jsdom.
 * - Tests never touch the network or real models.
 */
export default defineConfig({
  resolve: {
    alias: [{ find: "@", replacement: path.resolve(__dirname, "src") }],
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.{ts,tsx}", "scripts/**/*.test.{mjs,ts}"],
    exclude: ["node_modules/**", "dist/**", "tests/**", "src-tauri/**"],
    passWithNoTests: true,
    restoreMocks: true,
  },
});
