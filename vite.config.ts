import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import * as path from "path";
import { visualizer } from "rollup-plugin-visualizer";
import tailwindcss from "@tailwindcss/vite";
import { loadEnvClassification, viteEnvPrefix } from "./scripts/release/env-classification.mjs";

const analyzeBundle = process.env.ANALYZE === "true";

export default defineConfig({
  plugins: [
    tailwindcss(),
    ...(analyzeBundle
      ? [
          visualizer({
            gzipSize: true,
            template: "treemap",
            filename: "stats/rollup-stats.html",
          }) as any,
        ]
      : []),
    react({
      // jsxImportSource: '@welldone-software/why-did-you-render'
    }),
  ],

  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  /*
   * Only the `public` names of scripts/release/env-classification.json reach import.meta.env (OAuth client ids,
   * redirect overrides, TAURI_DEBUG; SIGIL_* canonical, CURSES_* legacy). They are complete variable names, so
   * Vite's prefix match is an exact match. This used to be the prefixes VITE_, TAURI_, SIGIL_ and CURSES_, which
   * put every variable with such a name (the Kick client secret, the Google Fonts API key, any SIGIL_* path or
   * flag in the shell, TAURI_SIGNING_PRIVATE_KEY) into the bundle that the embedded web server hands to the LAN.
   * Secrets are compiled into Sigil.exe by src-tauri/build.rs instead. viteEnvPrefix() throws on a classification
   * that could expose a secret; scripts/release/check-no-secrets.mjs checks the finished build.
   */
  envPrefix: viteEnvPrefix(loadEnvClassification(__dirname)),
  resolve: {
    alias: [{ find: "@", replacement: path.resolve(__dirname, "src") }],
  },
  build: {
    // Tauri WebView2 stays current; OBS browser source uses older CEF — keep output compatible with both.
    target: ["es2020", "chrome95", "safari15"],
    // don't minify for debug builds
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    // produce sourcemaps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,
    // vosk-browser ships ~6MB WASM; it is lazy-loaded (dynamic import) but still one Rollup chunk.
    chunkSizeWarningLimit: 7000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Vite's empty stand-ins for Node built-ins (e.g. `crypto`, required by a crypto-js-style fallback) get
          // their own chunk. Otherwise Rollup folds them into the first manual chunk that imports them (speech-sdk),
          // and the host bundle statically imports the Azure SDK's CommonJS setup code at boot.
          if (id.includes("__vite-browser-external")) return "browser-external";
          if (!id.includes("node_modules")) return;
          if (id.includes("@twurple")) return "twurple";
          if (id.includes("microsoft-cognitiveservices-speech")) return "speech-sdk";
          if (id.includes("vosk-browser")) return "vosk-browser";
          if (id.includes("framer-motion")) return "framer-motion";
          if (id.includes("peerjs") || id.includes("/yjs/") || id.includes("y-protocols"))
            return "collab";
          if (id.includes("react-dom") || id.includes("react/")) return "react-vendor";
        },
      },
    },
  },
});
