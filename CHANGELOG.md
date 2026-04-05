# Changelog

All notable changes to **Sigil** are documented here. Sigil continues from the **Curses / Curses+** lineage; compare **0.3.0** to **Curses+ 0.2.2** for the largest delta.

## [Unreleased]

### Speech-to-text (Whisper / Vosk)

- **Outbound HTTP** — Shared `reqwest` client uses a **Sigil/version User-Agent**, **Accept** defaults, longer **whole-request timeout** for large model downloads, **`http_download_get`**, and **HTTP error context** (status, `Content-Type`, short body preview) on failed Whisper/Vosk fetches.
- **Whisper** — Hugging Face resolve URLs append **`?download=true`**. **`ensure_dependencies`** accepts an optional **local `.bin` path** (copy + validate instead of download). New command **`import_whisper_model`** copies a user-picked file into app data for the selected catalog model. **Start/stop:** `onStart` runs only after **`start_recording`** succeeds; **Stop** cancels during download/setup so STT does not stay stuck in **Connecting…**.
- **Vosk** — **`download_vosk_model`** accepts an optional **local zip path**; **`import_vosk_model_zip`** copies a user-picked zip into **`vosk-models/`**. **Stop** tears down work in progress (download / WASM load / capture); **`createModel`** is wrapped with a **timeout** so a hung load cannot strand the UI.
- **Inspector / state** — Checkboxes and paths for **use local Whisper .bin** / **use local Vosk zip**, browse buttons, and **Copy into Sigil (import)** actions. New persisted fields on STT schema; English i18n keys under **`stt.*`**.

### Voice changer (inspector polish)

- Sectioned layout; **(i)** help popovers clamp to the inspector / SimpleBar scroll region so they stay readable while scrolling.
- **Monitor** gain row uses full-width layout.
- Removed **quick character preset** cards and one-shot **latency** shortcut buttons in favor of direct numeric fields and buffer presets (**low_latency** / **stable** / **custom**).

### Copy and internationalization

- **ONNX execution provider** — Voice changer and neural ONNX TTS use the label **Execution provider** (replacing “run the model on this device” style copy). Help strings describe **ONNX execution providers** (CPU vs GPU backends). Shared option strings **`tts.ep_*_plain`** drop the redundant **“This PC —”** prefix. Non-English locales use the same **Execution provider** label (stub **“(hint)”** suffix removed where it appeared).

### UI

- **In-app manual** — **`HelpManualModal`** in the header to read synced Markdown under **`public/manual/`** (pages such as voice changer, Discord, VRChat, backends).
- **Service panels** — Inspector components for each host service are registered in **`servicePanels.tsx`** with **lazy** imports to trim initial bundle work.

### Documentation

- **`docs/VOICE_CHANGER.md`** — Parity table and **chunk vs playback buffer** section aligned with the expanded chunk allowlist and pitch-in-all-modes behavior (see **0.4.0** voice changer notes for runtime scope).

---

## [0.4.0] - 2026-04-04

Release focused on VRChat/Discord hardening, neural speech and voice-changer depth, translation backends, performance work, and first-class HTTP TTS options in the UI.

### Performance and internals

- **Voice changer** — Hot DSP parameters batched under one lock (`VoiceChangerHotParams`); CPAL capture feeds an **SPSC `rtrb` ring** instead of a contended `Mutex<Vec<f32>>`; output queue trimming uses **`VecDeque::drain`** instead of repeated `pop_front`. **`push_aux_wav_bytes`** centralizes WAV→PCM mix for aux/TTS.
- **Neural TTS → voice changer** — **`neural_onnx_tts_speak_into_voice_changer`** and **`neural_sidecar_tts_speak_into_voice_changer`** synthesize or fetch audio in Rust and mix in one step (no base64 round-trip through the webview). Playback path returns **raw WAV bytes** (`Vec<u8>` / `number[]`), consistent with Kokoro-style HTTP TTS.
- **Neural DSP** — Classic phase vocoder path reuses an interleaved output buffer in **`chunk_processor`** to cut allocations.
- **PubSub / peer** — One **`JSON.stringify`** per publish where possible; **`broadcastPubSubParts`**; ring-backed transcript history (cap 120, no per-line **`shift()`**); **`#fanOutPubSubRemote`** for host relay after **`publishLocally`**.
- **STT** — Whisper noise-marker regex at module scope; Moonshine **`Float32Array`** pool for `audio:chunk`; **`base64ToUint8Array`** prefers **`Uint8Array.fromBase64`** when the runtime supports it.

### Text-to-speech (HTTP backends in UI)

- **Kokoro**, **MeloTTS**, **Chatterbox**, and **Fish Speech** — Already implemented as Tauri plugins and TS services; they are now **selectable in the TTS inspector** with endpoint fields, speed/options where applicable, and **Refresh list** for voices/speakers (Chatterbox remains voice name as text). See **`docs/BACKENDS.md`** for URLs and docker hints.

### VRChat (OSC)

- **Connection:** Configurable **OSC host** and **port** (default `127.0.0.1:9000`); UDP socket binds an **ephemeral** local port instead of a fixed port that could collide.
- **Diagnostics:** `bind_status` IPC + system log on bind failure; throttled system logs on failed OSC sends; clearer `invoke` error strings (string / `Error` / `{ message }`).
- **Chatbox:** Enforce **144** character limit per VRChat OSC rules; clear **typing** indicator on finals when the indicator is enabled; optional **chat notification sound** (third `/chatbox/input` bool per VRChat docs).
- **KillFrenzy:** Clear `hideTimer` after `clearTimeout` so visibility logic stays correct; `syncPoints` clamped to 1–4 with safe queue handling (`try`/`catch`, re-queue, drain after `finally`); inspector hide-after field **max** aligned with milliseconds.
- **UX:** **Send test** respects the selected target (Chatbox vs KillFrenzy).

### Discord (incoming webhooks)

- **API alignment:** Message **content** split into **2000**-character chunks with spacing between posts; **`allowed_mentions`: `{ parse: [] }`**; JSON body omits empty **`avatar_url`** and drops unnecessary **`embeds`/`attachments` nulls**; **`wait=true`** on execute for reliable errors.
- **Webhook URL:** Normalize/validate **`discord.com` / `discordapp.com`** `…/webhooks/…` URLs; trim and warn on invalid paste.
- **Username override:** Length clamp and rejection of disallowed substrings (**`clyde`**, **`discord`**) per Discord rules.
- **Rate limits:** Retry on **HTTP 429** using **`Retry-After`** (header or JSON body), capped wait.
- **Forum / media channels:** Optional **`thread_id`** (query) and **`thread_name`** (JSON) in service state and inspector.
- **Live gate:** **`postOnlyWhenLive`** (migrates persisted **`postWithTwitchLive`**) — when enabled, posts only if **Twitch or Kick** reports live; recording alert chip matches the same rule.
- **UX / docs:** **Send test** in inspector; **4xx/5xx** response bodies logged (truncated); onboarding copy describes **webhook** captions; **manual** updated (incl. independence from VRChat OSC).

### Text-to-speech (neural backends)

- **Short backend names** — Dropdown labels **ONNX (local)** and **HTTP server**; ONNX subsection title matches. **Service** (backend) is listed **before** text source in the speech provider block.
- **ONNX (local)** — `neural-onnx-tts`: load from **Hugging Face repo** or a local **speech package** folder (`speech_package_v1`; see `docs/LOCAL_SPEECH_PACKAGE.md`). User-facing copy and errors avoid vendor-specific naming; implementation still uses the `kittentts` crate for that package format. IPC: `neural_onnx_tts_set_model_source`, `neural_onnx_tts_set_local_package_format`. Persisted: `neuralOnnx.modelSource`, `neuralOnnx.localModelDir`, `neuralOnnx.localPackageFormat`. Folder browse (desktop). **Future:** extra local formats (e.g. Piper-style voices) are not in this release.
- **HTTP server** — `neural-sidecar-tts` plugin unchanged; inspector subsection title shortened to match.
- **IPC (0.4)** — **`neural_onnx_tts_speak`** / **`neural_sidecar_tts_speak`** return **raw WAV bytes** for playback; **`neural_onnx_tts_speak_into_voice_changer`** / **`neural_sidecar_tts_speak_into_voice_changer`** mix in one Rust hop when routing through the voice changer. Regenerate **`src-tauri/gen/schemas/`** with **`cargo build`** after **`build.rs`** edits.
- **Inspector / UX** — Shared **execution provider** option strings (`tts.ep_*_plain`); Web Audio output note unchanged in meaning.

### Voice changer (neural integration) — VCClient-class controls and streamer routing

Reference stack: [w-okada/voice-changer](https://github.com/w-okada/voice-changer) / VCClient (chunk, gains, buffer, truncation, buf/res-style metrics, monitor). Sigil stays on **Rust + cpal** and does **not** embed the Python/Socket.IO server; the **WebSocket** mode uses **raw binary PCM** via [`scripts/neural-sidecar/README.md`](scripts/neural-sidecar/README.md), documented as non–drop-in vs VCClient’s REST/Socket.IO server.

- **Processing chunk size (VC “CHUNK” equivalent)** — Runtime **`chunk_frames`** clamped to an allowlist from **160** through **48000** frames per channel (coarser steps at large sizes; ~**1 s** @ 48 kHz at the top), threaded through **`ClassicPhaseVocoderProcessor`** / `neural_audio::chunk_processor`, the voice-changer loop drain size, **ONNX** (`OnnxVoiceBridge`), and **WebSocket** sidecar chunk size. Changing chunk **restarts** the engine while running so DSP/tensor shapes stay consistent.
- **Pitch and formant (built-in + post-process)** — Rust clamp **±24** semitones. Inspector: **numeric pitch** + step buttons **±0.1 / ±0.5 / ±1 / ±12** semitones, range slider, reset to **0** (= no shift). **Formant** (brightness tilt **[−1, 1]**) with number field and **±0.05 / ±0.1** steps + slider. **Built-in, WebSocket, and ONNX** paths apply pitch where the DSP chain runs (including post-process on model/sidecar output when not using the classic mic-only phase vocoder).
- **Gains** — **`input_gain` / `output_gain`** (0–4) applied in Rust on the capture path and before main playback push; inspector numeric fields with **0.1** steps.
- **Playback buffer** — User-visible **`output_prefill_ms`** and **`output_max_queue_ms`** (clamped in Rust). **`latency_preset`**: **low_latency** / **stable** templates fill those ms fields; **custom** when the user edits ms manually. **`applyLatencyPreset`** syncs and **restarts** if the voice changer is already running. Changing buffer ms on blur also **restarts** when running.
- **Silence truncation (worklet-style)** — Optional **RMS threshold** + **consecutive silent chunks** trim the main playback deque toward the prefill floor (VCClient `volTrancate*` analogue).
- **Metrics (buf / res)** — IPC **`get_voice_changer_metrics`**: **buffer_ms** from queued samples, **process_ms** from an **exponential moving average** of per-chunk processing time (smoother than a single spike). Inspector polls when Tauri + running (~5 Hz).
- **Monitor output (streamers)** — Optional **second cpal output**: **`monitor_output_device`** (empty = off, **`default`**, or enumerated device), **`monitor_gain`** (0–4) on **pre–main-output-gain** PCM. **Channel up/down-mix** and **linear resample** when the monitor device’s rate or layout differs from the main output. Changing main or monitor device **restarts** while running.
- **Phase vocoder** — **`vocoder_window_ms`** (30–60) with number + slider; **`vocoder_oversample`** **1 | 2 | 4 | 8 | 16 | 32** (**1** = lowest CPU; library does not support “zero” oversampling). Section divider clarifies post-process / resample applies across modes where relevant.
- **Modes and UX** — Labels **Built-in**, **WebSocket**, **ONNX (local)**. ONNX: desktop **Browse** for `.onnx`, **execution provider** selection (**CPU / CUDA / DML**), float32 in/out with **adaptive chunk shape** vs mic; identity test model via `scripts/neural-sidecar/create_identity_vc_onnx.py`. Tips mention **no in-app AEC/noise suppression** (use OBS/OS/mic stack).
- **Rust / plugins** — `OnnxVoiceBridge` + **`ort`** + **`ndarray`**; plain-English validation on start. **`get_voice_changer_metrics`** and extended **`set_voice_changer_params`** fields wired in **`build.rs`** / capabilities (regenerate `src-tauri/gen/schemas/` after manifest edits).
- **TTS → VC** — **`push_aux_wav_bytes`** in Rust; UI prefers **`*_speak_into_voice_changer`** for neural TTS when VC is running; **`voice_changer_push_aux_wav`** / aux PCM still accept **`wavBytes`** / **`wavBytesB64`**.
- **Documentation** — [docs/VOICE_CHANGER.md](docs/VOICE_CHANGER.md) feature parity table (chunk allowlist, pitch, formant, gains, buffer, truncation, metrics, monitor, noise gap, WebSocket vs Socket.IO); [docs/SERVICE_CONNECTIVITY_AUDIT.md](docs/SERVICE_CONNECTIVITY_AUDIT.md) voice-changer row and summary; [scripts/neural-sidecar/README.md](scripts/neural-sidecar/README.md) updated for configurable processing chunk sizes vs fixed 512.
- **Internationalization** — New and updated **`voice_changer.*`** strings (chunk, latency **custom**, buffer fields, gains, truncation, metrics, pitch/formant steps, **monitor** section, stream/buffer copy); non-`en` locales synced with **`pnpm i18n:sync`**.

### Translation

- **LibreTranslate** — Inspector now lists **LibreTranslate (local server)** (previously only Azure in the UI): base URL, optional **API key**, from/to languages (dropdowns when `GET /languages` succeeds, otherwise free-text codes), auto-detect, interim STT handling; `plugin:translate|set_api_key` and JSON/query `api_key` on translate/detect/languages where applicable.
- **On-device** — **On-device (WebView Translator API)** backend: feature-detects `Translator` / `LanguageDetector`, BCP-47 normalization, cached translator instances; clear errors when the WebView runtime does not expose the API.
- **Docs** — `scripts/translation-sidecar/README.md` (Docker quick start, API contract, API key); `docs/SERVICE_CONNECTIVITY_AUDIT.md` translation row and backend wiring table.

### Internationalization

- **`toasts.stt_external_mic_timeout`** added to all non-English locale files so **`pnpm i18n:check`** passes (parity with `en`).
- New strings for Discord thread fields, send test, updated “post only when live” copy, and VRChat chatbox notification sound (`vrc.textbox_play_notification_sound`).
- **Neural TTS, voice changer, translation** — Expanded `tts.*` (provider subheader, backend labels, `ep_*_plain`), full `voice_changer.*` for the VC-style panel (see **Voice changer** bullets above for scope), and `transl.*`; non-`en` locales updated via **`pnpm i18n:sync`** (and related merge scripts where used).
- **HTTP TTS inspector keys** — **`scripts/merge-http-tts-i18n.mjs`** merges Kokoro / Melo / Chatterbox / Fish Speech **`tts.*`** strings into all non-**`en`** locales (English placeholders until translated).

### Other

- **Core schema:** `zodServiceSchemaFactory` accepts **`ZodTypeAny`** so service data schemas may use **`z.preprocess`** (Discord migration).
- **Tauri plugins / ACL** — `build.rs` permission manifests include `neural-onnx-tts`, `neural-sidecar-tts`, extended `voice-changer` commands (incl. aux WAV), and `translate` `set_api_key`; `capabilities/migrated.json` aligned. Run **`cargo build`** after `build.rs` edits so `src-tauri/gen/schemas/` stays in sync.

---

## [0.3.1] - 2026-03-31

### Internationalization

- **New locale bundles:** Arabic, Bengali, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese (Brazil), Russian, Thai, Turkish, Urdu, and Vietnamese (`public/i18n/<code>/translation.json`), plus refreshed Simplified / Traditional Chinese (`zh`, `zh_cn`, `zh_tw`).
- **Tooling:** `pnpm i18n:check` (key parity vs English + registered codes), `pnpm i18n:sync` (merge missing keys from `en`), `pnpm i18n:report` (keys still identical to English), `pnpm i18n:translate` → `scripts/build-translations.py` (optional Google MT refresh).
- **RTL:** `ar` and `ur` set `dir` / `lang` on `<html>`; shared UI chrome uses logical Tailwind (`ms`/`me`, `ps`/`pe`, `start`/`end`, etc.) where mirroring matters.
- **i18next plural forms** for multi-element duplicate actions and toasts (`elements.duplicate_n_*`, `elements.toast_duplicated_n_*`).
- **Client mirror** respects host **UI language** after peer init.

### Fixes and polish

- **Tauri CSP:** allow `blob:` for release WebView image rendering.
- **0.3.0 follow-ups:** image blob handling, file library modal, and delete confirmation flow.

### CI and quality

- **GitHub Actions:** `ci.yml` (pnpm, Node 20, `i18n:check`, typecheck); Playwright workflow aligned to **pnpm** + **Node 20** with `i18n:check` before tests; release workflow runs locale checks before build.
- **Playwright:** preview server uses `pnpm`; smoke tests verify `ar` / `ur` translation assets and basic RTL `dir` sanity.
- **Docs:** `ARCHITECTURE.md` — translation review checklist, RTL smoke checklist, i18n script summary.

### Native (Rust)

- Pre-webview **MessageBox** strings remain English; inline note defers optional localized OS dialogs to a future bridge.

### Documentation

- README updates from the community (synced on `master` before this release).

---

## [0.3.0] - 2026-03-30

### Highlights

- **Rebrand** to **Sigil** (app name, icons, bundle id `com.sigil.app`, Windows binary **Sigil.exe**).
- **Tauri 2** migration: plugins, capabilities/ACL, `@tauri-apps/api` v2, fs/dialog/shell/process/global-shortcut plugins.
- **pnpm** as the standard package manager for the repo (npm lockfile removed).

### Speech-to-text

- **Whisper (Rust)** — Expanded model options (additional quantized and Large-v3–class models); download and pipeline hardening across the upgrade branch.
- **Moonshine** — HTTP integration (Useful Sensors–style server: health + transcribe).
- **Vosk** — Rust-side catalog and on-disk model zip caching; load via Tauri FS + browser runtime.
- **OpenAI-style local HTTP** — `POST …/v1/audio/transcriptions` for local sidecars (e.g. speaches, LocalAI-style stacks).
- **PubSub** — Binary audio path can feed the Whisper pipeline for linked/distributed setups.
- **Stream alignment** — Options to stop STT/TTS with Twitch stream state; inspector and locale strings updated.

### Text-to-speech

### Translation

### Voice, audio, and VRChat

- **Voice changer** — Real-time pitch (semitones), formant control, presets, input/output device selection (Tauri plugin).
- **Global audio devices** — Shared input/output selection with Rust audio capture for multiple services.
- **OSC** — Remains available from the Tauri host for VRChat-oriented workflows.

### Streaming and integrations

- **Twitch** — OAuth for public and confidential apps; loopback and env via `SIGIL_*` (legacy `CURSES_*` still supported).
- **Kick** — OAuth and chat with the same env conventions.
- **OBS** — Browser source and obs-websocket workflows; fixes for browser source captions and related behavior.
- **Discord** — Webhook outbound paths unchanged in scope; see README.

### Editor and UI

- **Sigil Studio** — Dashboard/header refresh; **Outfit** for key chrome typography.
- **Layout** — Restored modern host layout, bottom panel, themes, property inspector placement.
- **System logs** — In-app diagnostics for services.
- **Legacy templates** — Old audio visualizer elements stripped on load (`legacyDocument` migration).

### Internationalization

- English plus Chinese (`zh`, `zh_cn`, `zh_tw`) updated for new service keys (sidebar, STT/TTS Twitch options, Azure secondary language, transform title, chat delay, etc.).

### Documentation and build

- README, **ARCHITECTURE.md**, **docs/BACKENDS.md**, **docs/MANUAL-TAURI-CHECKLIST.md**, and related upgrade notes.
- GitHub **release** workflow: Node 20, `Sigil.exe` + MSI artifacts, build-time secrets for Twitch, Kick, Google Fonts, optional Twitch client secret.

### Upgrade notes (from Curses+ 0.2.2)

1. Installers and shortcuts use **Sigil**; refresh OBS browser URLs if they still reference Curses.
2. Prefer **`SIGIL_*`** env vars in `.env`; **`CURSES_*`** remains as fallback where documented.
3. Custom automation targeting **Tauri 1** APIs must move to **Tauri 2**.
4. Use **`pnpm install`** / **`pnpm tauri build`** for this repo.

### Credits

Sigil is a fork of **[Curses](https://github.com/mmpneo/curses)** by **mmpneo**. Thanks to the original author and all contributors.

---

## [0.2.2] - (Curses+)

Last pre-Sigil milestone in this fork’s history: **Whisper FFI refactor** for high-performance local STT. For full detail, see git history at tag/commit `Release v0.2.2: Whisper FFI Refactor`.

[0.3.0]: https://github.com/Mewnah/Sigil/releases/tag/v0.3.0
[0.3.1]: https://github.com/Mewnah/Sigil/releases/tag/v0.3.1
[0.4.0]: https://github.com/Mewnah/Sigil/releases/tag/v0.4.0
