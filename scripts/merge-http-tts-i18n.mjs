/**
 * Merges HTTP TTS inspector keys (Kokoro, Melo, Chatterbox, Fish Speech) into non-en locale files.
 * Uses English strings as placeholders until translators update each file.
 *
 * Run from repo root: node scripts/merge-http-tts-i18n.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const i18nRoot = path.join(__dirname, "..", "public", "i18n");

/** Must match public/i18n/en/translation.json tts.* keys for HTTP backends (and shared helpers). */
const httpTtsKeys = {
  backend_kokoro: "Kokoro (HTTP)",
  backend_melo: "MeloTTS (HTTP)",
  backend_chatterbox: "Chatterbox (HTTP)",
  backend_fish_speech: "Fish Speech (HTTP)",
  http_refresh_list: "Refresh list",
  http_list_fetch_failed:
    "Could not load voices from the server. Check the URL and that the service is running.",
  kokoro_title: "Kokoro",
  kokoro_desc:
    "Kokoro FastAPI–compatible server (OpenAI-style /v1/audio/speech). Example: docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu",
  kokoro_endpoint: "Server URL",
  kokoro_speed: "Speed",
  melo_title: "MeloTTS",
  melo_desc: "MeloTTS HTTP API. Example: docker run -p 8888:8888 myshell-ai/melotts",
  melo_endpoint: "Server URL",
  melo_speaker: "Speaker",
  melo_speed: "Speed",
  chatterbox_title: "Chatterbox",
  chatterbox_desc:
    "Chatterbox TTS HTTP server (see your stack’s compose or docs). Default URL is often port 5555.",
  chatterbox_endpoint: "Server URL",
  chatterbox_speed: "Speed",
  chatterbox_exaggeration: "Exaggeration",
  fish_title: "Fish Speech",
  fish_desc: "Fish Speech API. Example: docker run -p 8080:8080 fishaudio/fish-speech",
  fish_endpoint: "Server URL",
  fish_reference_id: "Reference / voice id",
};

for (const dirent of fs.readdirSync(i18nRoot, { withFileTypes: true })) {
  if (!dirent.isDirectory() || dirent.name === "en") continue;
  const file = path.join(i18nRoot, dirent.name, "translation.json");
  if (!fs.existsSync(file)) continue;
  const raw = fs.readFileSync(file, "utf8");
  const j = JSON.parse(raw);
  j.tts = { ...j.tts, ...httpTtsKeys };
  fs.writeFileSync(file, JSON.stringify(j, null, 2) + "\n", "utf8");
  console.log("updated", dirent.name);
}
