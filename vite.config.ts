import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import * as path from "path";
import { visualizer } from "rollup-plugin-visualizer";
import tailwindcss from "@tailwindcss/vite";

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
  /* SIGIL_* is canonical; CURSES_* still exposed for legacy .env files from the Curses fork. */
  envPrefix: ["VITE_", "TAURI_", "SIGIL_", "CURSES_"],
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
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          if (id.includes("@twurple")) return "twurple";
          if (id.includes("microsoft-cognitiveservices-speech")) return "speech-sdk";
          if (id.includes("openai")) return "openai";
          if (id.includes("framer-motion")) return "framer-motion";
          if (id.includes("peerjs") || id.includes("/yjs/") || id.includes("y-protocols"))
            return "collab";
          if (id.includes("react-dom") || id.includes("react/")) return "react-vendor";
        },
      },
    },
  },
});
