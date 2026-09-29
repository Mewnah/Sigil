## Learned User Preferences

- When implementing attached plans, do not edit the plan file itself; mark existing todos in progress and complete all of them without stopping.
- Prefer consolidated documentation over many scattered markdown files; keep detail but reduce file count.
- Prioritize release-build reliability for STT and core audio paths, not just `pnpm tauri dev` behavior.
- STT services should stop immediately so users can switch engines without waiting for long connecting timeouts.
- Voice changer settings should hot-swap without requiring a full application restart.
- When thermo review findings are accepted, apply the recommended fixes in-tree rather than only documenting them.
- Release timing is flexible; comprehensive multi-part implementation plans are acceptable when not shipping soon.
- Optimize the voice changer for VRChat streamers so it coexists with OBS and VRChat/games on shared CPU and GPU.
- Prefer compute-preserving realtime voice-changer defaults, while still offering higher-quality options for users with headroom.
- When building a release, produce the Tauri installer bundle (NSIS/MSI), not a `--no-bundle` portable `Sigil.exe`.
- Voice changer inspector should show only controls relevant to the selected mode; hide overlapping preset or duplicate options.

## Learned Workspace Facts

- Sigil is a Tauri desktop fork of the curses application; some UI and behavior are migrated from curses.
- Dev: `pnpm tauri dev`. Release: `pnpm install` then `pnpm tauri build` for NSIS/MSI under `src-tauri/target/release/bundle/`; the raw `Sigil.exe` skips the installer WebView layout and can fail on launch. Web-only: `pnpm build` → `dist/`. Typecheck: `pnpm typecheck`.
- NiceModal dialogs need a side-effect import in the inspector module (e.g. `import "../stt-replacements-modal"`) so `NiceModal.register` runs.
- `muteSoundEffects` silences studio SFX only; the OBS `/client` browser mirror is unaffected.
- Template import must call files-service asset reconciliation after immer-yjs replaces the `filesMeta` Y.Array or images/fonts stay blank.
- Packaged Whisper/Vosk STT needs Tauri `assetProtocol` (with `protocol-asset` Cargo feature), scoped to `com.sigil.app`, plus CSP `worker-src`/`connect-src` for `blob:` and `asset:`; Vosk loads models via `convertFileSrc` from AppData.
- Rust-backed STT engines use the inspector Rust input device dropdown via `rustAudioInput.ts`, not browser WebRTC device IDs.
- Voice changer ONNX inference uses the unified `ort` crate (CPU + CUDA + DirectML in one release build); execution provider is selected at runtime in the inspector.
- AI voice changer direction is one RVC engine with profiles (`streamer` / `low_latency` / `quality`), not a separate LLVC CPU default; F0 defaults to fcpe on CPU, with rmvpe/crepe treated as GPU-oriented.
- Voice-changer ORT sessions should disable spinning and cap intra/inter threads so idle inference does not peg a core beside OBS/VRChat.
- Large voice-changer audio changes should stay compatible with the shared TTS/native audio stack (device resolve, ORT session helpers under `neural_audio`).
- Operator docs live in `docs/` and `public/docs/`; the user manual is consolidated into numbered chapters `00`–`04` under `docs/manual/`.
