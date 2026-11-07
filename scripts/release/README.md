# Release tooling (Windows)

What turns the working tree into installers a friend can run, and the checks that say it is safe to hand them out. Nothing here runs by itself, pushes, or publishes behind your back: the installers are handed out by hand, and the one script that uploads (`publish-assets.ps1`, for the `assets-v1` model and runtime release) sends nothing unless you pass `-Execute`.

## The release, in order

```powershell
node scripts/release/check-catalog-pins.mjs          # 1. every published hash pinned, the release set immutable (offline, seconds)
pnpm tauri:release                                    # 2. NSIS (per-user) + MSI, provider DLLs and the Visual C++ runtime staged next to Sigil.exe
pwsh scripts/release/verify-bundle.ps1                # 3. DLLs inside both installers, no new load-time imports, EP probe, no secret value in dist / exe / installers
pwsh scripts/release/packaged-smoke-all.ps1           # 4. the installer's app starts with no GPU and no packs
pwsh scripts/release/packaged-smoke-all.ps1 -Mode Sandbox   # 5. the same in a clean Windows (Windows Sandbox, virtual GPU off)
```

`.github/workflows/release.yml` (manual dispatch) runs 1 to 4 on a hosted Windows runner, which has no GPU and no Sigil data, and creates a **draft** release. Its smoke step passes `-RequireOnline -RequireGui`, so a skipped download or app check stops the release instead of passing silently (see "A skip is not a pass"). Steps 1 to 3 need no network except the first build's ONNX Runtime download.

Build tip on a PC someone is using: `$p = Get-Process -Id $PID; $p.PriorityClass = 'BelowNormal'; $env:CARGO_BUILD_JOBS = 6` before `pnpm tauri:release`; hold `scripts/bench/gpu-lock.ps1` while it compiles if a benchmark may be running.

| File | What it does |
|---|---|
| `tauri.release.conf.json` | Overlay for `tauri build`: maps the staging folder `src-tauri/target/native-bundle` (the ONNX Runtime provider DLLs and the Visual C++ runtime) to the install root, and installs per user. `pnpm tauri:release` applies it. |
| `tauri-release.mjs` | What `pnpm tauri:release` runs: `tauri build` (with the overlay) after remapping the builder's home folder out of the paths the compilers embed (`--remap-path-prefix`, MSVC `/d1trimfile`, `/PDBALTPATH`), so the installers do not carry the build machine's user name. `verify-bundle.ps1` fails if `Sigil.exe` still names it. |
| `stage-native-dlls.mjs` | Copies `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` and `DirectML.dll` from the cargo target, and the Visual C++ runtime (`vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`, `msvcp140_1.dll`) from the Visual Studio redistributable folder, into `src-tauri/target/native-bundle` (hash manifest next to it). See "The Visual C++ runtime". |
| `verify-bundle.ps1`, `expected-imports.txt` | Step 3: staged DLLs match their manifest and sit next to `Sigil.exe` in the MSI and the NSIS setup (the three provider DLLs and the four Visual C++ runtime DLLs); `dumpbin /DEPENDENTS` shows no CUDA, cuDNN, ONNX Runtime, ggml, OpenVR or sherpa load-time import (they load lazily so Sigil starts on any PC) and no Visual C++ runtime import beyond the four shipped; the installed layout runs the headless EP probe. |
| `check-catalog-pins.mjs`, `release-set.json` | Step 1, below. |
| `check-no-secrets.mjs`, `env-classification.json`, `env-classification.mjs` | Part of step 3: no secret value in the frontend bundle, the exe or the extracted installers. See "Secrets". |
| `packaged-smoke-all.ps1` | Steps 4 and 5, below. |
| `publish-assets.ps1` | Uploads the staged Sigil-built assets to the `assets-v1` GitHub release with the GitHub CLI. Prints the plan; uploads only with `-Execute`. See "The assets-v1 release". |
| `exit-probe.ps1` | Diagnostic, not a gate: one minimized launch that reproduces the shutdown stall the exit guard exists for (WM_CLOSE to the event loop's invisible window), see "Closing the window must end the process". |

## Secrets (part of step 3)

Which build-time variables may reach the frontend bundle is written down once, in `env-classification.json` (names only): the `public` list (OAuth client ids, redirect overrides, `TAURI_DEBUG`) is the allow-list `vite.config.ts` gives Vite as `envPrefix`; the `secret` list (the Kick and Twitch client secrets, the Google Fonts API key) is what `src-tauri/build.rs` compiles into `Sigil.exe`, XOR-obfuscated with a per-build key. The obfuscation keeps the values out of a text search of the binary and out of the web bundle that the embedded server hands to the LAN. **It is not protection**: the key is in the same binary, and the secret is in memory and on the wire while the app runs. Kick requires a client secret for its token endpoint (there is no public-client flow), so it has to ship; the real defences are rotating it in the Kick developer portal and that Rust only accepts Sigil's two localhost redirect URIs.

- `env-classification.mjs` loads the file and refuses a combination that could expose a secret (a credential-looking public name, a name in both lists, a public name that is a prefix of a secret name). `vite.config.ts` calls it, so an unsafe edit stops the build.
- `check-no-secrets.mjs [path ...]` collects the values of every variable named like `SECRET`, `API_KEY`, `TOKEN` or `PASSWORD` (and every `secret` name) from `.env`, `.env.local` and the process environment, and fails when one appears as plain text, as UTF-8, UTF-16LE, JSON-escaped or URL-encoded, in `dist/` or in any path given (`Sigil.exe`, an extracted installer folder). It prints variable names and file paths, never a value. `--require-values` fails when there is nothing to compare (CI), `--no-process-env` reads only the files, `--no-dist` scans only the given paths. A compressed installer (NSIS, MSI) can only be searched as the bytes it is: `verify-bundle.ps1` extracts both and passes the folders.
- `verify-bundle.ps1` runs it after the installers are extracted (`-RequireSecretValues` in CI). By hand after `pnpm build`: `node scripts/release/check-no-secrets.mjs`.
- If it ever fails: the build is not safe to hand out; rotate the named secret (Kick developer portal, Google Cloud console), find the reference in `src/` (`src/utils/__tests__/noFrontendSecrets.test.ts` should have caught it) and rebuild.
- After creating `.env` for the first time, Cargo does not know it exists yet (a build script cannot watch a file that is missing without recompiling every time): touch `src-tauri/build.rs` or run `cargo clean -p sigil` once.

## The Visual C++ runtime

`Sigil.exe` load-time imports `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll` and `msvcp140_1.dll`. They are not part of Windows: a PC that never installed a program built with them (a clean install, a new VM, Windows Sandbox) has none, and Sigil exits with `0xC0000135` (`STATUS_DLL_NOT_FOUND`) before it shows anything. A dev PC always has them in System32, so only a clean machine sees it. Both installers and the portable zip therefore carry them app-local, next to `Sigil.exe`:

- `stage-native-dlls.mjs --stage` (the overlay's `beforeBundleCommand`) copies them into the same staging folder as the ONNX Runtime provider DLLs, so the overlay's single mapping (`target/native-bundle` to the install root) puts them in the NSIS setup, the MSI and the portable zip. Nothing is added to the base `tauri.conf.json`.
- Source: the redistributable folder of the installed Visual Studio / Build Tools, `VC\Redist\MSVC\<version>\x64\Microsoft.VC14*.CRT` (vswhere lists every installation; the newest version wins). Microsoft licenses these files for app-local deployment (REDIST.TXT). They are never taken from System32 and never downloaded. If they cannot be found the staging step stops the build; `node scripts/release/stage-native-dlls.mjs --check-vc` shows the folder it would use (CI runs it before the compile), `--vc-redist <folder>` or `SIGIL_VC_REDIST_DIR` names another `Microsoft.VC14*.CRT` folder (a Windows system folder is refused).
- `verify-bundle.ps1` requires the four files in the staging folder (hash manifest) and next to `Sigil.exe` in both installers, byte for byte, and fails when `Sigil.exe` imports another `vcruntime` / `msvcp` DLL. The smoke's `layout` check requires them next to the installed `Sigil.exe` (a plain `-Mode Layout` folder only gets a note).
- The Universal CRT (`api-ms-win-crt-*`) is part of Windows 10 and 11 and is not bundled.

## Catalog pins (step 1)

`src-tauri/model-catalog/*.json` lists everything Sigil can download. `build.rs` already refuses a release build with a published file that has no `sha256` / `git_oid`; `check-catalog-pins.mjs` runs the same rule in seconds, without Cargo, and adds:

- every source URL is immutable: a Hugging Face **commit** (not `main`, not a tag), a release tag that is not `latest` / `nightly`, a raw.githubusercontent.com **commit**, a versioned NVIDIA redistributable, a Mozilla attachment. A host that can swap the file under the same URL (alphacephei.com, for Vosk) is accepted only when `release-set.json` lists it under `tofu` with the reason;
- the **release set** (`release-set.json`: the defaults a new install fetches, and the light options next to them) is published, and so is everything it `requires`; an unpublished requirement must be listed under `allowUnpublished` (Firefox Translations' helper, which is built locally and imported);
- every CPU model, runtime and the language-ID model of `stt-routing.json` is in the defaults, so a routing change without a matching edit of `release-set.json` fails.

Two stronger checks, run by hand (neither runs in CI):

```powershell
# Re-derive size and digest of the defaults from the hosts' own metadata (Hugging Face tree API, GitHub release asset
# digests, NVIDIA's redistrib manifests, GitHub blob ids). Sequential, 2.5 s between requests. --group all adds the
# light options (--group light only those). --hash-by-download hashes a file again when its host publishes no digest
# (one at a time, up to --max-download-mb, default 50; the Firefox Translations models are never downloaded).
node scripts/release/check-catalog-pins.mjs --online
node scripts/release/check-catalog-pins.mjs --online --group light --hash-by-download --max-download-mb 10
# Compare the catalog with the install records of a model store that downloaded the entries (the store verified every
# byte against the pin when it wrote the record).
node scripts/release/check-catalog-pins.mjs --installed "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p7"
```

`--report <file>` writes the result as JSON. Pinning new entries stays `scripts/models/pin-catalog.mjs` (see `src-tauri/model-catalog/README.md`).

What the checks cannot promise:

- **Rolling releases.** The default speech models come from k2-fsa's `asr-models` and `tts-models` GitHub releases, which keep gaining assets and whose old assets may one day be deleted. A replaced asset fails its hash check instead of installing, but a removed one cannot be fetched: copy the defaults' archives to a release of your own and add a second `github-release` source (after the first) to those entries. `check-catalog-pins.mjs` prints these entries as notes.
- **Vosk.** alphacephei.com publishes only md5 digests, so the sha256 of those 11 legacy archives comes from one download each.
- **Sigil-built packs** (`runtime/bergamot`, transcribe.cpp, Qwen3-TTS server, s2, the voice-changer encoders) are `unpublished`: they install by Import or `SIGIL_LOCAL_ASSETS`. When you publish one, build it, pin its size and sha256 in its fragment, set `"status": "published"` and rerun the check.

## Packaged smoke (steps 4 and 5)

`packaged-smoke-all.ps1` needs only Windows PowerShell 5.1 or PowerShell 7 and the installer, so one file runs on the dev PC, in Windows Sandbox or a VM, and on a CI runner. It writes `bench-results/release-smoke/<time>/smoke-result.json` (schema `sigil-packaged-smoke/1`: machine facts, one record per check, the notes).

| Mode | What it does | Where |
|---|---|---|
| `Extract` (default) | Unpacks the newest NSIS setup with 7-Zip and runs it against a scratch Windows profile (USERPROFILE / APPDATA / LOCALAPPDATA redirected; the script checks that the real `%APPDATA%\com.sigil.app` is untouched). Nothing is installed. | the dev PC |
| `Portable` / `Layout` | The portable zip, or any folder with `Sigil.exe`. | the dev PC |
| `Installed` | Runs the NSIS setup silently (per user), tests the installed app, uninstalls. Refuses outside a disposable machine unless `-AllowInstall`. A GitHub-hosted runner (`GITHUB_ACTIONS=true` and `RUNNER_ENVIRONMENT=github-hosted`) and Windows Sandbox (user `WDAGUtilityAccount`) count as disposable; a bare `CI=true` does not (it is set in plenty of developer shells), and neither does a self-hosted runner, which can be someone's own PC. | CI, a VM, Sandbox |
| `Sandbox` | Writes a Windows Sandbox configuration (virtual GPU **off**, 4 GB, networking only with `-Online`), runs `Installed` inside and reads the result back. Only when Windows Sandbox is already enabled; the script never enables Windows features. Exit code 3 and the manual steps otherwise. Windows Sandbox opens its viewer window on your desktop (it cannot be hidden), so start this one when you are away from the PC; automated runs use `Extract`. | the dev PC |

Checks: `build-fresh` (with the repository at hand: no source file that goes into the app is newer than the installer; if one is, the check is a skip that names the newest file, because every other check then describes an older tree), `layout` (the three provider DLLs and the four Visual C++ runtime DLLs next to the exe), `sysinfo`, `no-gpu` (with `-ExpectNoGpu`: no hardware adapter), `catalog` (`fetch --list` shows the whole embedded catalog, nothing installed, the count equals the repository's), `ep-probe`, then the app itself on a fresh profile: `gui-server` (the embedded web server answers `/`, `/client`, the translations), `gui-page` (no "Application failed to start", host services booted; when the WebView never exposes its DevTools page the failure carries a summary and `gui-page-diag.txt` holds the processes, listening sockets and the `/json/version` answer), `gui-models` (Models & Runtimes lists every entry as not installed), `gui-idle` (no file under `models` / `runtimes`, no helper process started by itself), `gui-focus` (the app did not take the foreground, below), `gui-log` (no panic; `-Strict` also fails on ERROR lines), `gui-exit` (closing the window ends the process, no helper outlives it). Optional tiers:

- `-Online`: `base/test-small` installs through the model store; the sherpa-onnx pack (8 MB) installs and loads next to Sigil's static ONNX Runtime. Both prove the catalog pins end to end. `-OnlineModel` adds SenseVoice (163 MB) and decodes English speech.
- `-PackStore <app data folder of a model store>` (read only): the release exe decodes real speech with the default models (Nemotron 560 ms on a French clip, SenseVoice on an English one).

A step that cannot run (no WebView2 runtime, no network, 7-Zip missing) is a SKIP; `-Strict` makes every skip a failure. Exit code: 0 passed, 1 failed, 2 usage, 3 Windows Sandbox unavailable.

### A skip is not a pass

A skip leaves the exit code at 0, so a gate that only tolerates skips cannot catch what the skipped step would have caught. Two things keep the release gate honest:

- **A download that fails because a host answered wrongly is a failure, never a skip.** The online checks read why `Sigil.exe --headless fetch` failed (`Get-FetchFailureKind`): only an unreachable host (DNS, connect, timeout, rate limit) is a skip. A sha256 or size that disagrees with the catalog, HTTP 4xx / 5xx, a different version, or a message the script does not recognize fails the check. So a k2-fsa asset that was replaced or removed after it was pinned fails the smoke even without any switch.
- **`-RequireOnline` and `-RequireGui` turn the skips of their tier into failures** (`online-*` and `gui-*`: no network, no WebView2 runtime, no DevTools page, a build that predates `SIGIL_START_MINIMIZED`). `-RequireOnline` implies `-Online`. `release.yml` passes both. `-Strict` is the blanket version (every skip, plus ERROR lines in the log and an exit that only the exit guard achieved).

### The page check without DevTools (Windows Sandbox)

`gui-page` and `gui-models` normally read the running page over Chrome DevTools: the smoke starts the app with `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=<port>`. In Windows Sandbox (Windows 11 Enterprise 26100, WebView2 155.0.4283.45) that override is **not applied**: the WebView2 browser process starts with Sigil's own arguments only (the host PC's carries `--remote-debugging-port` right after them), no port opens, and the same happens when the variable is passed in an explicit environment block (`ProcessStartInfo`, `UseShellExecute = false`) and with the profile on the guest disk instead of the mapped folder. Edge itself opens the port from its command line in the same guest, so DevTools is not blocked by policy; the override is. The guest account is an administrator without UAC (the smoke records `elevated` in its machine facts and in `gui-page-diag.txt`), which is the lead, not a proven cause. A GitHub-hosted runner is elevated the same way and may behave like the guest.

When the DevTools page never appears, `gui-page` therefore falls back to the app's own log (`Get-FrontendBootEvidence`): the page relays its system log to `sigil.log`, and its first line `[System] Monitoring ...` is written after the host services have initialized, so a page that died earlier has none. That is the failure this fallback exists to catch: with no default route (a PC with no network, Windows Sandbox with networking off) the first IPC call used to fail with "Error retrieving local IP", the front end never started, and the log stopped after the start-up probe. The check passes with a detail and a note that say the page was **not inspected** (an error screen drawn later would not be seen), `gui-page-diag.txt` keeps the evidence (processes and command lines, listening sockets, the `/json/version` answer, elevation), and `gui-models` then rests on the catalog check (`fetch --list`: every entry not installed) and an empty `models` / `runtimes` folder of the fresh profile instead of the page's own list. With DevTools available nothing changes. `-NoDevTools` starts the app without the override to exercise this path anywhere.

### A window must never pop up on the PC the smoke runs on

The app is started with `SIGIL_START_MINIMIZED=1` (`src-tauri/src/main.rs`): the main window is created hidden and unfocused, shown minimized without activation, and the saved-window-position restore (which ends in a forced `set_focus`) is skipped. `gui-focus` proves it on every run: a sampler records the foreground window every 100 ms from the instant the process is created for 15 s, and the check fails if it was ever Sigil's, if the main window does not end minimized, or if any other window shows (a dialog; the profile is seeded so no one-time offer appears). A `Sigil.exe` built before that switch existed is not started on a PC that is not disposable (the GUI checks are skipped, with the reason). `-ShowWindow` starts the app the normal way for a manual run.

### Closing the window must end the process

`gui-exit` posts `WM_CLOSE` to the window titled `Sigil` (what the close button does; again after 10 s) and waits up to `-ExitWaitSec` (35 s) for the process to end. It does **not** use `Process.CloseMainWindow()`: Sigil's event loop (tao) owns an invisible "Tao Thread Event Target" window that Windows counts as a visible, owner-less top-level window, and once the real window is minimized it is the one `CloseMainWindow()` picks. `WM_CLOSE` there destroys the loop's own message window; the real window still closes afterwards, but the process then lingers. That is the "process outlived its closed window" seen in about one launch in seven before the harness was fixed, and `scripts/release/exit-probe.ps1` (post `WM_CLOSE` to that window first, then close the real one) reproduced it in its one run: the real window closed, the process stayed until the guard ended it 20 s later. Tools that post `WM_CLOSE` to every window of a process (`taskkill` without `/F`, installers, the Restart Manager) can do the same to a real install, and tao's window procedure cannot be changed, so `src-tauri/src/exit_guard.rs` makes sure the process cannot linger: 20 s after the first close request it logs one `exit guard:` WARN line (is the main window still there, shown, minimized, is its UI thread unresponsive) and ends the process with `TerminateProcess`. The smoke passes with a note when only the guard ended the app, and fails under `-Strict`; the line in `%LOCALAPPDATA%\com.sigil.app\logs\sigil.log` says which case it was. Never use `Process.CloseMainWindow()` in a harness for this app (`CloseMainWindow` or `taskkill` without `/F` are what a script would reach for first); close the window titled `Sigil`, as `gui-exit` does.

### Clean machine smoke without Windows Sandbox

1. Use a Windows 10/11 x64 virtual machine with no GPU passthrough and no Sigil data (a snapshot is ideal). Copy the NSIS setup and `packaged-smoke-all.ps1` into it.
2. In Windows PowerShell inside the VM: `powershell -ExecutionPolicy Bypass -File .\packaged-smoke-all.ps1 -Mode Installed -AllowInstall -InstallerPath .\Sigil_<version>_x64-setup.exe -ExpectNoGpu -Online -ResultsDir C:\smoke-out`. (`-AllowInstall` says this VM is disposable: the script installs the app for the current user and uninstalls it afterwards, and refuses to do that on a PC it cannot tell is disposable. GitHub-hosted runners and Windows Sandbox are recognized without it.)
3. Every check in `C:\smoke-out\smoke-result.json` should be `pass` (the GUI checks `skip` without the WebView2 runtime; the setup installs it when the VM has internet).
4. By hand, once, with the window open: Settings -> GPU & diagnostics names the processor (or DirectML with the reason) and **Test GPU acceleration** completes; Models & Runtimes shows every entry as not installed; the STT backend **Built-in (on your PC)** with English, **Start**, asks before downloading and shows the size.

## Firefox Translations (Bergamot) pack: manual build

The translator chain is Hy-MT2 1.8B, Firefox Translations, NLLB-200 600M. The middle member needs a Sigil-built helper that no release step builds: without it the chain skips that member and falls through to NLLB (ADR 0008). Building needs MSVC, CMake and seven source archives from GitHub (the root is the whole `mozilla/translations` tree at commit `69455ac`, so its size is unknown until the first fetch):

```powershell
pwsh scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives   # one archive at a time, 3 s apart; records sha256 in helper/UPSTREAM.json
pwsh scripts/runtime-packs/bergamot/build.ps1                           # writes sigil-bergamot-69455ac-s1-cpu.zip and .sha256
```

Import the zip in Models & Runtimes (or put it in a `SIGIL_LOCAL_ASSETS` folder). To publish it later, pin its size and sha256 in `runtime-bergamot.json` and set the status as described above.

## Why whisper-rs stays

The plan retires `whisper-rs` (the one statically linked ggml) only if transcribe.cpp matches it on WER and speed. Not obvious today, so it stays:

- Whisper is still a route. `stt-routing.json` sends the long tail of languages (`other`) and the automatic route's last fallback to `whisper/small-q5_1` and `whisper/base-q5_1`; the legacy Whisper backend and its downloaded models depend on it.
- No parity data exists. The P7 campaign measured Whisper only as the fp16 base model (WER 25.8 on clean English, RTF 1.09 on 4 threads: an upper bound on cost) and did not run Whisper through transcribe.cpp; the transcribe.cpp tier is opt-in, unbenchmarked in P7 and its packs are unpublished.
- Retiring it would move Whisper users onto a pack they would have to import.

Keeping it costs one ggml in `Sigil.exe` and the `GGML_NATIVE=OFF` / `GGML_AVX2=ON` build environment (`src-tauri/.cargo/config.toml`, `release.yml`). Revisit when a published transcribe.cpp pack runs the quantized Whisper models with WER and RTF at parity on the P7 corpus (`scripts/speech-bench`); then drop `whisper-rs` from `Cargo.toml` and the two GGML variables.

## Publishing (by hand)

1. `pwsh scripts/release/packaged-smoke-all.ps1 -Mode Sandbox -Online` (or the VM steps above) on the exact installer you will hand out.
2. Checksums: `Get-FileHash src-tauri\target\release\bundle\nsis\*-setup.exe, src-tauri\target\release\bundle\msi\*.msi, src-tauri\target\release\Sigil_*_x64_portable.zip | ForEach-Object { "$($_.Hash.ToLower())  $(Split-Path -Leaf $_.Path)" } | Set-Content SHA256SUMS.txt -Encoding ascii`.
3. Upload the NSIS setup (recommended, per user), the MSI, the portable zip and `SHA256SUMS.txt` to a GitHub release; the workflow's draft does this for the first three.
4. Optional mirrors for the default models (see "Rolling releases") go to a release such as `assets-v1` of this repository.

## The assets-v1 release (Sigil-built packs, exports and templates)

The `unpublished` entries of `src-tauri/model-catalog/` (the transcribe, Qwen3-TTS and echo runtime packs, the NLLB 1.3B zip, the voice changer's encoders, templates and zero-shot graphs, the FCPE pitch model, the benchmark models, the Magpie tokenizers, ...) are files Sigil builds itself. Their catalog sources are assets of one pre-release, <https://github.com/Mewnah/Sigil-assets/releases/tag/assets-v1> (`https://github.com/Mewnah/Sigil-assets/releases/download/assets-v1/<asset name>`). The Models tab shows a download button only for `published` entries, so these entries get their button in five steps. Each step is a command you run; nothing leaves the PC before step 3 and nothing here ever asks for a credential.

1. **Stage.** One folder (any location; hard links are fine) that holds each file under its exact release asset name. A release has no folders, so the name is the last part of the URL in the catalog. `node scripts/models/assets-v1.mjs plan --staging <dir>` lists what the catalog expects, what is there, what is missing and what no entry names (`--hash` also hashes every file against its pin).
2. **Pin.** `node scripts/models/pin-staged.mjs --staging <dir> --write` hashes the staged files and writes `size` and `sha256` into the catalog fragments in the format `pin-catalog.mjs` writes, keeping each fragment's own layout. For an archive entry the pin is the zip as it is downloaded (the store hashes the download, then extracts). A file that already has a pin is only checked: other bytes are a MISMATCH (exit 1), and `--force` re-pins a rebuilt pack. It also writes `SHA256SUMS.txt` and `MANIFEST.json` into the folder. `pin-catalog.mjs` cannot do this job before the upload: it reads the host's own metadata and the release does not hold the files yet.
3. **Upload.** `pwsh scripts/release/publish-assets.ps1 -Staging <dir>` prints the upload plan (files, sizes, which entries use them, what is blocked) and stops. With `-Execute` it needs the GitHub CLI that you installed and logged in (`gh auth login`; the script never reads or stores a token): it creates the `assets-v1` pre-release when it does not exist, then uploads the files one at a time with a pause between uploads, skipping a file that is already there with the same size. A file that is there with another size stops that file unless you pass `-Replace`. `-Verify` hashes every file against its pin before planning. Files blocked by the plan (no pin, wrong size or hash) are never uploaded.
4. **Prove and flip.** `node scripts/models/publish-entries.mjs --ids all-staged --staging <dir>` (dry run) checks, sequentially with at least 2.5 s between requests, that every release URL answers and its bytes hash to the pin (each file is downloaded to a temp file that is deleted right after; `--digest-only` compares GitHub's own asset digests instead, one request for the whole release). A file that also has an upstream source (the Magpie v2607 checkpoint on Hugging Face) is checked against that host's metadata, and its release copy only when staged. Add `--write` to set `"status": "published"` on the entries that verified; it changes only the status line of each entry, and an entry whose `requires` did not verify stays unpublished. `--ids a/b,c/d` names entries instead of `all-staged`; `--except <id>` drops one.
5. **Check.** `node scripts/release/check-catalog-pins.mjs` (offline) and `pnpm docs:check`, then rebuild: `build.rs` refuses a `published` file without a pin, and the Models tab now shows a download button for the flipped entries ("Import only" is gone). The voice changer's fragments are written by `scripts/assets/vc_catalog.py`; it keeps the status of the committed fragment, so regenerating never takes a published entry back.

A dry run of step 4 needs no release: serve the staging folder on this PC (any static file server, 127.0.0.1 only, files laid out flat) and add `--assets-base http://127.0.0.1:<port>/ --catalog-dir <copy of src-tauri/model-catalog>`.

Entries whose files are not built yet (the Vulkan packs need the Vulkan SDK, Bergamot and the s2 packs a first source fetch) take the same road later: build the zip, put it in the staging folder, run steps 2 to 4 for it (`--only <id prefix>` on `pin-staged.mjs`, `--ids <id>` on `publish-entries.mjs`). Rebuilding a pack changes its bytes: pin again with `--force` before uploading, and upload with `-Replace` if the old file is already on the release, and bump the entry `version` so the Models tab offers the update.

Release assets are limited to 2 GiB each (`mt/nllb-200-distilled-1.3b-int8` is 1.56 GiB).

