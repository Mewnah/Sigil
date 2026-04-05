/**
 * Merges neural TTS + voice_changer keys into non-en locale files (English placeholders).
 * Run from repo root: node scripts/merge-neural-i18n.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const i18nRoot = path.join(__dirname, "..", "public", "i18n");

const neuralTtsKeys = {
  neural_onnx_title: "Neural ONNX (KittenTTS)",
  neural_onnx_desc:
    "On-device neural TTS (small ONNX model). The first run downloads weights into the Hugging Face cache.",
  neural_onnx_hub: "Hugging Face repo id",
  neural_onnx_voice: "Voice",
  neural_onnx_speed: "Speed",
  neural_onnx_ep: "Execution provider hint",
  neural_sidecar_title: "Neural sidecar (HTTP)",
  neural_sidecar_desc:
    "Uses your own server: GET /health and POST /speech with JSON { text, voice }. Example script: scripts/neural-sidecar/tts_sidecar.py (port 8766).",
  neural_sidecar_endpoint: "Base URL",
  neural_route_through_vc: "Mix into voice changer input (start VC first)",
  neural_web_audio_output_note:
    "Playback uses the default Web Audio output (same as Native TTS). The device field is kept for future routing.",
};

const voiceChanger = {
  mode: "Engine",
  mode_classic: "Classic (phase vocoder)",
  mode_sidecar: "Neural sidecar (WebSocket)",
  mode_onnx: "ONNX in-process (not available yet)",
  sidecar_ws_url: "Sidecar WebSocket URL",
  latency_preset: "Output buffer",
  latency_low: "Low latency",
  latency_stable: "Stable (more delay, fewer glitches)",
  vram_note:
    "Voice conversion competes for GPU time with VRChat and OBS. Prefer Sidecar on another GPU, DirectML on integrated graphics, or CPU. See scripts/neural-sidecar/README.md.",
};

for (const dirent of fs.readdirSync(i18nRoot, { withFileTypes: true })) {
  if (!dirent.isDirectory() || dirent.name === "en") continue;
  const file = path.join(i18nRoot, dirent.name, "translation.json");
  if (!fs.existsSync(file)) continue;
  const raw = fs.readFileSync(file, "utf8");
  const j = JSON.parse(raw);
  j.tts = { ...j.tts, ...neuralTtsKeys };
  j.voice_changer = { ...voiceChanger };
  fs.writeFileSync(file, JSON.stringify(j, null, 2) + "\n", "utf8");
  console.log("updated", dirent.name);
}
