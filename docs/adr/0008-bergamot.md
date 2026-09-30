# ADR 0008: Mozilla Bergamot as a JSON-lines helper process (zero-VRAM CPU translation)

- **Status:** design accepted; **no-go for this phase**. The client, the protocol, the model catalog,
  the build script and the protocol shell are done and tested. `sigil-bergamot.exe` itself is not built:
  fetching the upstream sources was refused in the P4-E session, so the real build and the en↔ja
  latency run are a manual step (see Consequences). Until a pack is built and imported, the zero-VRAM
  CPU fallback is **NLLB-200 600M int8 plus Hy-MT2 1.8B on llama-server CPU**. Bergamot stays in the
  router's default chain: without its pack the member fails to start quietly and the chain falls through
  to those engines, and an imported pack is used with no settings change (see "No-go path").
- **Date:** 2026-09-30
- **Workstream:** P4-E
- **Plan decision:** "Translation pipeline placement" (Bergamot runs as a helper exe with JSON lines
  over stdio under `process_host`, so its C++ runtime stays out of the real-time process); deferred item
  "Bergamot on MSVC" (research §3.1 "Mozilla Bergamot", "zero setup (Bergamot on demand)").

## Context

Research §3.1 picks Mozilla's Firefox Translations models (Bergamot: Marian int8 students with intgemm)
as the offline CPU tier:

- 36–70 MB per direction, no graphics memory, tens of milliseconds per line on one core;
- better than NLLB-600M, OPUS-MT and Argos on en↔ja/ko/zh in Mozilla's own evaluations (vendor numbers,
  en-ja COMET22 0.900);
- the automatic fallback when the GPU is busy or the LLM or a cloud engine fails, and the cheapest way to
  re-translate partial captions (the router prefers it for interims).

The engine is C++ (bergamot-translator on Mozilla's marian fork). It raises three problems on Windows:

1. **MSVC fragility.** marian and intgemm are built mostly with GCC and Clang; the MSVC build is the
   least exercised path (the plan's deferred item).
2. **Isolation.** Marian aborts the process on many errors and has global state. In process it would
   share Sigil's address space with the voice changer, a static ORT, whisper-rs's ggml and the audio
   callbacks.
3. **Which upstream.** `browsermt/bergamot-translator` has not moved in years. Mozilla now maintains the
   engine in the `inference/` folder of `mozilla/translations`, the tree its WebAssembly build (the one
   Firefox ships) comes from. The current models (registry versions 2.x) are trained and evaluated for
   that tree.

Models come from Mozilla's registry, the Remote Settings collection `main/translations-models`: 715
records, 111 directions, all to or from English, several versions per direction with filter expressions
per platform and channel.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| **A. `sigil-bergamot.exe` helper under `process_host`, JSON lines over stdio, built from `mozilla/translations` `inference/`** | Crash, memory and CPU isolation (Job Object, memory cap, below-normal priority). No CRT or symbol clash with Sigil. The upstream Firefox uses. | One more build (MSVC risk) and a pipe protocol to maintain. | **Chosen.** Protocol shell built with MSVC and conformance-tested (below). |
| B. In-process DLL through FFI | No IPC | A marian abort kills Sigil (and the voice changer). Global state and allocations next to the real-time threads. `/MT` vs `/MD` and static intgemm symbols. | Rejected by the plan's placement decision. |
| C. The WebAssembly build in a Rust WASM runtime | Upstream artifact | Needs the `mozIntGemm` imports implemented by the host, plus a large new dependency (a Cargo hotspot). Slower than native intgemm. | Rejected. |
| D. translateLocally's Windows binary | Prebuilt | A Qt GUI app with its own model manager and downloads. An unpinned third-party executable. | Rejected. |
| E. Convert the Marian students to ONNX and run them on Sigil's ORT | No C++ | intgemm-packed weights, shortlist and SSRU decoder have no maintained ONNX path. | Rejected for now. |
| F. `browsermt/bergamot-translator` v0.4.5 instead of Mozilla's tree | Smaller archive | Unmaintained. May not load the current 2.x models. | Rejected. |
| G. No Bergamot: NLLB-600M int8 + Hy-MT2 1.8B CPU | Already built (P4-B/D) | NLLB is slower and weaker on CJK; Hy-MT2 on CPU is far slower. | The fallback if A is a no-go. |

## Decision

### Helper: `sigil-bergamot.exe --jsonl [--threads N] [--cache-size N]`

- **Source:** `scripts/runtime-packs/bergamot/helper/`:
  - `src/main.cpp`: protocol shell and scheduler;
  - `src/json.cpp`: strict, depth-limited JSON;
  - `src/engine_bergamot.cpp`: the upstream adapter;
  - `src/engine_fake.cpp`: the protocol-test engine;
  - `CMakeLists.txt` and `UPSTREAM.json`.
- **Location.** The plan named `tools/bergamot-helper/`, but `tools/**` belonged to the concurrently
  running P3, so the source lives under the workstream's own `scripts/runtime-packs/bergamot/`. A later
  integrator may move it.
- **Engine:** Bergamot's `AsyncService` with N workers and a translation cache.
  - Models are loaded from memory bundles read through wide-character paths, so user folders with
    non-ASCII names work (model aligned to 256 bytes, shortlist and vocabularies to 64).
  - Decoder settings are Firefox's: beam 1, `max-length-break` 128, `mini-batch-words` 1024,
    `workspace` 128 MB, `gemm-precision` from the file name.
  - `marian::setThrowExceptionOnAbort(true)`, so a bad model fails its `load` instead of killing the
    helper.
- **Build:** MSVC, static CRT, upstream's WebAssembly-compatible sources (intgemm for int8, the ONNX.js
  SGEMM, no MKL/BLAS), library only.
  - `-Arch avx2` (default; upstream's `x86-64-v3` mapping to `/arch:AVX2`).
  - `sse2` compatibility build: intgemm still picks SSSE3/AVX2/AVX-512 kernels at run time.
  - `avx512`.
  - Never `native`.
- **Stdout isolation:** at startup the CRT's stdout and the Win32 stdout handle are pointed at stderr. The
  protocol writes to a private duplicate of the original stdout, so a library log line can never corrupt
  the stream.

### Protocol 1

One JSON object per line each way, UTF-8. Every request has a client-chosen `id` and an `op`:

| op | fields | result |
| --- | --- | --- |
| `hello` | `protocol` | `{protocol, helper, engine, threads, arch}` |
| `load` | `key` (`[a-z0-9_-]{1,40}`), `model`, `lex` (or null), `vocabs` `[src, trg]`, `gemm` | `{key, ms}` |
| `unload` | `key` | `{key}` |
| `translate` | `keys` (1, or 2 = pivot), `text` (≤ 64 KiB), `priority` `final`/`interim` | `{text, ms}` |
| `cancel` | `target` (a queued request's id) | `{target, cancelled}` |
| `stats` | | `{loaded, queued, running, threads, translations, loads}` |

- **Replies** are `{"id":N,"ok":true,"result":{…}}` or `{"id":N,"ok":false,"error":{"kind","message"}}`.
  - Error kinds: `invalid_request`, `not_loaded`, `load_failed`, `translate_failed`, `cancelled`,
    `busy`, `internal`.
  - A line that is not JSON gets `"id":null`.
  - Replies come in any order.
- **Scheduling:** loads and unloads first, then finals, then interims. At most `threads` translations
  run at once. The queue holds at most 256 requests. `cancel` drops a queued request.
- **Shutdown:** EOF on stdin means quit after the running work (the process host allows 500 ms, then
  ends the job).
- **stderr** carries diagnostics only.

### Runtime pack `runtime/bergamot` (`model-catalog/runtime-bergamot.json`)

- Built by `scripts/runtime-packs/bergamot/build.ps1` and kept **unpublished** (asset hosting is local
  only). Install it by local import or `SIGIL_LOCAL_ASSETS` (`sigil-bergamot-69455ac-s1-cpu.zip`).
- **Pinned sources** (`UPSTREAM.json`):
  - `mozilla/translations` at commit `69455ac` (only `inference/` is extracted);
  - one archive per git submodule the native build needs, at the commits the parent tree records:
    `ssplit-cpp`, `intgemm`, `sentencepiece`, `onnxjs`, `ruy` and `simd_utils`.
- **Fetching:** sequential, at least 3 s apart, into `%LOCALAPPDATA%\sigil-dev-cache\bergamot`. The sha256
  and size of each archive are recorded once with `-PinArchives` and verified on every later fetch.
- **Pack contents:** the exe, `sigil-pack.json`, each component's license file and
  `THIRD-PARTY-NOTICES.md`. `sigil-pack.json` holds the protocol, `arch` and the sha256 of every file.
- **The client checks the pack before every start:**
  - the protocol version;
  - the CPU features (AVX2 + FMA for the default build);
  - the executable's size and sha256 (hashed once per file version).

  A mismatch fails with an actionable message instead of a crash loop.

### Models: `mt-bergamot/<source>_<target>` (`model-catalog/mt-bergamot.json`)

- **Generated** by `scripts/runtime-packs/bergamot/gen-catalog.mjs` from the registry records, using
  Firefox's rule for Windows on the release channel:
  - only records whose filter expression applies (unknown expressions are skipped with a warning);
  - no pre-release versions;
  - per direction, the highest version with a complete file set: an intgemm model, a shortlist, and
    either a shared vocabulary or `srcvocab` + `trgvocab`.
- **Coverage today:** 106 directions, 3.45 GB if every one were installed. The five nightly-only
  directions (az→en, be→en, mt→en, nn→en, en→sq) are excluded.
- **Ids** use Sigil's canonical codes, lower-cased: `zh` is Simplified and `zh-hant` Traditional.
  Traditional Chinese never falls back to the Simplified models.
- **Entry details:**
  - family `mt-bergamot`: its own group in Models & Runtimes;
  - `requires: ["runtime/bergamot"]`;
  - license MPL-2.0, as a catalog fact.
- **Pinning:** sizes and sha256 are the registry's own attachment digests, which Firefox verifies too.
  The files come from Mozilla's attachments CDN, which serves them unchanged and immutable.

### Rust client: `src-tauri/src/services/mt_local/bergamot{.rs,/}`

The commands are the frozen `bergamot_*` set.

- **Routing.** A direct direction, or two legs through English. An unknown source is refused, since
  Bergamot has no language detection (the TS engine guesses the source from the script first). The same
  language on both sides passes the text through.
- **Missing models.** They fail with `not_installed` naming every missing id. The client never
  downloads.
- **Model loading.**
  - Models load on demand, once per helper process.
  - At most `min(6, what fits in 1.2 GB)` directions are loaded, and that is also the peak: a load
    first unloads the least recently used directions that no line is translating with (the unloads go
    out before the load, and the helper handles control messages in order). When the other directions
    are all in use, the load waits for those lines (at most the 90 s load timeout, then `busy`). Each
    direction takes about 64 MB plus a 128 MB workspace per worker, and the recipe caps the job at
    1.5 GB, so going over the limit would get the helper killed mid-load.
  - A direction whose load ends the helper twice is refused, with advice to reinstall it. Only a helper
    that ended on its own counts. A helper Sigil ended (a stop, a restart after a hang, a thread-count
    change) or a load message that never reached the helper does not. The count is keyed by the model
    file (path, size, modification time), so a reinstall lifts the refusal, and a stop clears it.
- **Failures and limits.**
  - A final gets one retry after a helper crash or restart; an interim gets none.
  - More than 3 interims in flight get `busy`.
  - A final that has no answer within 30 s restarts the helper; for an interim the limit is 10 s.
  - A failed helper start fails fast with `busy` for 30 s.
- **Restarts.** After `process_host` restarts a crashed helper, the client re-attaches to it (a new
  `take_stdio`) instead of starting another one.
- **Stop and cancel.** A stop cancels everything in flight and never restarts the helper.
  `bergamot_cancel` fails the call at once and asks the helper to drop the request if it is still
  queued.
  - Every stop starts a new generation. A helper connection belongs to the generation it was opened
    under, and requests only use connections of their own generation. The stop is registered with the
    process host before the generation changes, and it ends only the helpers of earlier generations.
    A pipeline restart sends `bergamot_stop` and then, right away, `bergamot_start` and a warm-up line;
    the new start keeps its own helper and the stop does not report Idle over it.
  - Starting the helper and loading models run on their own tasks. A line cancelled during either
    returns at once (under 500 ms) and frees its slot. The start or the load already sent completes
    for the lines after it, so a dropped start never leaves a helper nobody can attach to. No further
    load starts for a cancelled line.
- **Status** is `mt_local:status` with `engine: "bergamot"`.

### TS engine: `src/core/services/translation/services/bergamot.ts`

- **start**
  - Lists the directions (`bergamot_pairs`).
  - Rejects a pair Bergamot cannot translate.
  - Names missing downloads in `TranslationError.missing`, for the inspector's model requirement.
    Missing direction models also go to the status as a warning, since a router member's failed start
    is otherwise silent. A missing helper pack does not: it is built locally, and a router member
    without it just falls through, so it must not warn on every start retry.
  - With an automatic source, at least one installed direction must reach the target.
  - Starts the helper with the slice's threads.
  - Preloads the pair with a short warm-up line.
- **supports()** answers from the installed directions. With an unknown source (`""`) it answers
  whether an installed direction reaches the target, since `translate` guesses the source from the
  script. The directions are refreshed (debounced 500 ms) on `models:state` events for
  `mt-bergamot/*` and `runtime/bergamot`.
- **Shared helper.** The last engine instance to stop stops the helper; a router member and a second
  pipeline share one.

## Evidence

All numbers are from the dev PC (H2): Windows 11, MSVC 19.44 (VS 2022 Build Tools 17.14), CMake 4.2.3,
Rust 1.94.

- **Rust client.** `cargo test mt_local::bergamot` passes 29 tests (2 more are manual and ignored).
  They start real helper processes through `process_host`: the test binary re-invoked as a protocol-1
  fake.
  - direct and pivot translation;
  - one load per model;
  - status events;
  - `not_installed` naming both pivot legs, without starting the helper;
  - a missing runtime;
  - cancel in under 500 ms, also while a model loads and while the helper starts (the start then
    serves the next line with no second helper);
  - stop cancelling a request in flight;
  - a crash retried once for finals, and the helper coming back;
  - a load that crashes twice being refused, until the model is reinstalled or the engine stops;
  - stops during a load never counting against the model;
  - a start right after a stop keeping its helper (three rounds, with a helper that takes 1 s to exit);
  - a stop not waiting out a slow start;
  - helper errors;
  - a protocol mismatch (the helper is stopped);
  - a hung helper being restarted after the timeout;
  - interim `busy` while finals still go through;
  - LRU eviction, with the fake killing itself like a job at its memory cap if it ever holds more than
    the limit (direct lines, a pivot sharing a leg, a pivot with two new legs, an interim);
  - a load waiting for a line that still uses the only other loaded model;
  - the peak (limit × per-direction memory) under the budget and the job cap for every worker count;
  - a thread-count change restarting the helper;
  - pack manifest, CPU and sha256 checks;
  - the embedded catalog (106 usable directions, every code a canonical TS language).
- **C++ protocol shell (fake engine).** `pwsh scripts/runtime-packs/bergamot/build.ps1 -Fake` builds a
  0.27 MB exe with MSVC. The helper's own sources also compile cleanly at `/W4 /permissive-`. The build
  runs the JSON unit tests (surrogate pairs, invalid and overlong UTF-8, depth limit, id ranges,
  escaping) and passes the protocol self-test:
  - `hello`;
  - invalid JSON answered with id null;
  - `not_loaded`, unknown op and bad key;
  - a load from a folder named `…-日本`;
  - a UTF-8 and escape round trip (`héllo "quoted"\n日本 😀`);
  - cancelling a queued interim behind a running one;
  - exit code 0 on stdin EOF.

  This test caught one real bug before any user saw it: `fread` on a pipe waits for a full 64 KiB
  buffer. The reader now uses `_read`.
- **Rust engine against the C++ helper.**
  `SIGIL_BERGAMOT_EXE=…\bergamot-fake\sigil-bergamot.exe cargo test external_helper -- --ignored` drives
  that exe through the production recipe (`--jsonl --threads 1 --cache-size 1024`). Direct, pivot and
  UTF-8 lines all pass. The client, pipe and scheduler overhead is 0.3–0.8 ms per line (debug build), so
  the latency budget belongs to Bergamot itself.
- **Registry pinning.** `srcvocab.enja.spm` (796,275 B) was downloaded from the CDN. Its sha256
  `970c98d1…091889` equals the registry's `attachment.hash`. The CDN reports identity encoding and
  `immutable` caching.
- **TS.** vitest: 26 engine tests (`__tests__/bergamot.engine.test.ts`), including Bergamot selected
  directly as the pipeline backend (it connects with a configured or an automatic source and names a
  missing direction model), and 7 generator tests (`scripts/runtime-packs/bergamot/__tests__`).
- **Not measured yet:**
  - the real `sigil-bergamot.exe` build;
  - real translation latency;
  - memory per direction (the 128 MB workspace figure comes from the configuration, not a
    measurement).

## Consequences

- **Manual step: build and measure.** Fetching the upstream source in the P4-E session was refused by
  the session's permission classifier ("untrusted code integration"), so the real helper is not built
  yet. Steps:
  1. `pwsh scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives`: about 7 archives,
     sequential. This records their sha256 in `helper/UPSTREAM.json`; keep that change.
  2. `pwsh scripts/runtime-packs/bergamot/build.ps1`: builds and stages
     `%LOCALAPPDATA%\sigil-native\packs\sigil-bergamot-69455ac-s1-cpu.zip`.
  3. Import that zip for **Firefox Translations engine (Bergamot)** in Models & Runtimes, then download
     *English → Japanese* and *Japanese → English* (Firefox Translations models).
  4. `pwsh scripts/runtime-packs/bergamot/build.ps1 -SelfTest -Models "$env:APPDATA\com.sigil.app\models\mt-bergamot"`.
     The acceptance is p95 < 150 ms per line, en↔ja. Optionally also
     `SIGIL_BERGAMOT_EXE=<pack exe> SIGIL_TEST_MODELS_DIR=<app data>\models cargo test external_helper -- --ignored --nocapture`.
  5. Record the numbers here and in `docs/benchmarks/SPEECH_BENCHMARKS.md` (P4-I0 / P7).
- **What may need adapting on the first real build.**
  - `engine_bergamot.cpp` is written against the upstream API listed in its header (`AsyncService`,
    `TranslationModel(config, MemoryBundle&&, replicas)`, `Response::target.text`,
    `setThrowExceptionOnAbort`). It is the only file that depends on it.
  - The upstream CMake may also want nested submodules that are not fetched today (for example ruy's
    cpuinfo): add them to `UPSTREAM.json`.
  - CMake 4 needs `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for old third-party trees; `build.ps1` passes it.
  - ssplit-cpp (sentence splitting) needs PCRE2. Its internal-PCRE2 mode, the one used for WebAssembly
    builds, may download the PCRE2 release while configuring: one more request, pinned only by
    ssplit's own hash, if it has one. To keep every fetch pinned, add PCRE2 (BSD) to `UPSTREAM.json`
    and point ssplit at the extracted tree.
  - If upstream's WebAssembly-compatible sources do not build with MSVC, the alternative is
    `USE_WASM_COMPATIBLE_SOURCE=OFF` with a BLAS (OpenBLAS or MKL), which is heavier. Record that choice
    here if it is taken.
- **No-go path (in effect until the manual build succeeds).**
  - Bergamot stays uninstalled. Its router member fails to start with "engine not installed", and the
    chain falls through to NLLB-200 600M int8 (P4-D) and Hy-MT2 1.8B on llama-server CPU (P4-B).
  - That failure is quiet: the TS engine does not put a missing helper pack in the status (it does for
    a missing direction model), and the router tries the member again only every
    `ROUTER_START_RETRY_MS` (60 s), which costs one `bergamot_pairs` and one pack lookup. So `bergamot`
    stays in the default chain, the interim order and the local fallbacks (P4-C), and a pack imported
    later is used with no settings change. Selecting Bergamot directly still fails its start with the
    missing pack named, which the inspector offers for import.
  - If the MSVC build turns out to be impossible, not just unattempted, remove `bergamot` from the
    default chain and the interim order (P4-C) and set this ADR's status to rejected.
- **Updates.**
  - Models: re-run `gen-catalog.mjs --fetch --write` (one request). The version field changes, and the
    store treats that as an update.
  - Engine: bump `UPSTREAM.json` (commit, `shim_revision`), pin, and rebuild. Bump the catalog version
    and the zip name together.
- **Follow-ups.**
  - P4-D / INTEGRATE: add a `bergamot` engine to `--headless mt-bench`
    (`mt_local::bergamot::translate_line`), so chrF is measured on the same corpus.
  - P7: Bergamot CJK CPU speed under VRChat load.
  - Publishing the pack waits for a user decision (asset hosting is local only).
