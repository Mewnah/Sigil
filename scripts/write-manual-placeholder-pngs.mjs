/**
 * Writes a minimal valid 1×1 PNG into each filename used by docs/manual SHOTLIST.
 * Replace these with real captures when ready (same filenames).
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const assetsDir = path.join(__dirname, "..", "docs", "manual", "assets");

/** Smallest valid PNG (1×1 transparent). */
const TINY_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

const NAMES = [
  "dashboard.png",
  "layout-overview.png",
  "stt-inspector.png",
  "tts-inspector.png",
  "translation-inspector.png",
  "transform-inspector.png",
  "voice-changer-inspector.png",
  "canvas-scenes.png",
  "canvas-properties-text.png",
  "files-inspector.png",
  "twitch-inspector.png",
  "kick-inspector.png",
  "discord-inspector.png",
  "obs-inspector.png",
  "vrc-inspector.png",
  "settings-inspector.png",
  "project-inspector.png",
  "control-bar.png",
  "client-mirror.png",
];

fs.mkdirSync(assetsDir, { recursive: true });
for (const name of NAMES) {
  const p = path.join(assetsDir, name);
  fs.writeFileSync(p, TINY_PNG);
  console.log("[write-manual-placeholder-pngs]", p);
}
