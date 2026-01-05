import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import * as path from "path";
import { VitePWA } from "vite-plugin-pwa";
import { visualizer } from "rollup-plugin-visualizer";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    tailwindcss(),
    visualizer({
      gzipSize: true,
      template: 'treemap',
      filename: 'stats/rollup-stats.html',
    }) as any,
    react({
      // jsxImportSource: '@welldone-software/why-did-you-render'
    }),
    VitePWA({
      workbox: {
        skipWaiting: true
      },
      injectRegister: null,
      devOptions: {
        enabled: false
      },
    }),
  ],

  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_", "SIGIL_", "CURSES_"],
  resolve: {
    alias: [{ find: "@", replacement: path.resolve(__dirname, "src") }],
  },
  build: {
    // Tauri uses modern Chromium - chrome120+ supports OKLCH for DaisyUI v5
    target: ["es2021", "chrome120", "safari15"],
    // don't minify for debug builds
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    // produce sourcemaps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});
