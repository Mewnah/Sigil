# ADR 0005: Process host for helper processes, with llama-server as the local LLM runtime

- **Status:** accepted. The process host is implemented and tested. The live llama-server + Hy-MT2
  1.8B smoke run is still manual (see Evidence).
- **Date:** 2026-09-29
- **Workstream:** P1-E
- **Plan decision:** "Helper processes and llama.cpp"; global rule "Helpers run in a
  KILL_ON_JOB_CLOSE Job Object with CREATE_NO_WINDOW, bind 127.0.0.1 with a random key, and log to
  the file log" (research §3.2, and §2.2 "any sidecar Sigil starts must bind to 127.0.0.1")

## Context

LLM translation (Hy-MT2 1.8B/7B, MiLMMT, TranslateGemma) needs a GGUF runtime. In-process
`llama-cpp-2` is blocked on Windows for three reasons:

- its static ggml clashes with whisper-rs's static ggml at the MSVC link;
- GGUF files over 4 GB fail to load on MSVC;
- builds have crashed on PCs other than the build machine.

Other runtimes will be separate processes too: Bergamot (P4-E), qwentts.cpp (P6-E), sbv2, and the
research voice-changer sidecars (P5-E). Until now Sigil's only child process was `open_browser`.

A helper Sigil starts must meet five requirements:

- **No orphans.** A helper that outlives Sigil (after a crash, or a kill from Task Manager) keeps
  holding VRAM and a port.
- **No LAN exposure.** Sigil's own web server already listens on `0.0.0.0` without auth (CLAUDE.md).
  A second unauthenticated LAN listener would be worse.
- **No console windows,** and no dialogs: no Windows Error Reporting dialog when a helper crashes,
  and no loader box when a DLL it needs is missing.
- **Logs in Sigil's file log.**
- **Protection for real-time work.** A CPU-heavy helper must not starve VRChat or the voice changer.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| **A. `process_host` module: std `Command` + one Job Object per helper, llama-server from the official ggml-org release zips as a model-store runtime pack** | Crash and VRAM isolation. No link clash. Official builds for Vulkan, CPU and CUDA 13. Kill-on-close covers every exit path. | Must manage ports, keys, health and restarts itself. | Chosen. 58 tests below. |
| B. In-process `llama-cpp-2` | No IPC | ggml link clash with whisper-rs, >4 GB GGUF bug on MSVC, cross-machine crashes | Research §3.2. Rejected. |
| C. `tauri-plugin-shell` sidecar (`externalBin`) | Built in | No Job Object, so an orphan after a crash. No memory cap. The binary is bundled in the installer. | Rejected |
| D. Point the OpenAI-compatible client at a user-run Ollama / LM Studio | Nothing to manage | Not zero-setup | Kept as the plan's fallback (P4-A presets) |

How a process joins its job without a race:

| Option | Verdict |
| --- | --- |
| Spawn normally, then `AssignProcessToJobObject` | Rejected: a helper could start a grandchild, or Sigil could die, before the assignment. |
| `PROC_THREAD_ATTRIBUTE_JOB_LIST` | Rejected: std's `raw_attribute` is still unstable (Rust 1.94), and the alternative is a hand-written `CreateProcessW` with pipes and argument quoting. |
| `CREATE_SUSPENDED`, assign, then resume the main thread | Rejected: std does not expose the thread handle on stable (`main_thread_handle` is unstable). |
| **`CREATE_SUSPENDED`, assign, then `NtResumeProcess`** | **Chosen.** ntdll has exported it since XP; it is resolved with `GetProcAddress`, and the host falls back to spawn-then-assign if it is ever missing. |

## Decision

1. **`services/process_host` runs every helper.** Files:
   - `job.rs`: Job Objects;
   - `spawn.rs`: validation, placeholders, keys, launch;
   - `instance.rs`: one process;
   - `lifecycle.rs`: registry, start, supervisor, stop;
   - `ports.rs`, `health.rs`, `logs.rs`;
   - `worker.rs`: one-shot isolated workers;
   - `recipes.rs`: llama-server.

   The frozen P1-S0b API is unchanged: `spawn`, `stop`, `get`, `list`, `shutdown_all`, the IPC
   commands `list_helpers` and `stop_helper`, and the `process_host:state` event. The additions are
   additive:
   - spec fields `env_remove`, `priority`, `high_qos`, `stop_grace_ms` and `stdio`, plus
     `HealthCheck.send_api_key`;
   - `handle`, `logs`, `take_stdio` and `run_worker`;
   - IPC `get_helper_logs`.
2. **Launch.**
   - Flags are `CREATE_NO_WINDOW | CREATE_SUSPENDED`, with piped stdin, stdout and stderr.
   - Each helper gets its own anonymous, non-inheritable Job Object with these limits:
     - `KILL_ON_JOB_CLOSE`;
     - `DIE_ON_UNHANDLED_EXCEPTION`;
     - optional `JOB_MEMORY`, which caps committed memory across the whole job (memory-mapped GGUF
       weights are not commit);
     - optional `PRIORITY_CLASS` (`below_normal` or `idle`).
   - The process is assigned to the job, then resumed. Optionally it is opted out of EcoQoS.
   - Children inherit the process error mode at creation. Sigil adds `SEM_FAILCRITICALERRORS |
     SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX` to its mode only around the `CreateProcess`
     call, under a lock, and restores it right after, so Sigil's own crash reporting is unchanged.
     Without this, a helper with a missing static-import DLL (an antivirus quarantine, a GPU
     runtime DLL absent on a friend's PC) blocks on the loader's modal "The code execution cannot
     proceed" box instead of exiting with 0xC0000135.
   - When the helper's main process exits, the rest of its job is terminated.
   - The job's I/O completion port reports memory-limit hits, so the exit reason can say "it ran into
     its N MB memory limit".
   - NTSTATUS exit codes are explained. For example, 0xC0000135 is "a required DLL was not found".
     Loader failures during a start (0xC0000135, 0xC0000139, 0xC000007B) are deterministic, so they
     fail the start at once instead of going through the restart policy.
3. **Loopback only.**
   - `auto` ports are chosen by binding `127.0.0.1:0`. If the helper exits before it ever answered
     and the port is now held by another program, that is a collision. It is retried on a new port,
     up to 3 times, and not counted as a failure.
   - Before the first HTTP probe, and so before a bearer key is ever sent, the OS listener table
     must show that the socket receiving `127.0.0.1:port` connections belongs to a process in the
     helper's job. If another program took the port in the pick-to-bind window, that is also a
     collision (for a `fixed` port it is an error). The foreign program only ever sees TCP
     connects, and its answers never count as the helper's. If the table cannot be read, the check
     is skipped.
   - A `fixed` port that is already in use is refused up front.
   - Once the helper is ready, the **exposure audit** reads the OS listener table
     (`GetExtendedTcpTable` with owning pids, iphlpapi resolved at run time). It checks the pids in
     the helper's job (`JobObjectBasicProcessIdList`), on every port, not only the announced one.
   - Any listener of those processes on a non-loopback address (`0.0.0.0`, `::`, a LAN IP) gets the
     helper killed, and the start fails with "helpers must bind 127.0.0.1 only".
   - If the table cannot be read, the audit falls back to connecting to the announced port on every
     non-loopback address of the PC. The audit is instant and has no timing false negatives.
4. **API key.**
   - 256 bits from the OS CSPRNG: `ProcessPrng`, falling back to `RtlGenRandom`. It is hex-encoded.
   - It is substituted for `{api_key}` and redacted from every log line. Lines longer than 2 KB are
     cut into pieces; the key is replaced in the byte stream before a cut, and the cut keeps the
     last `len(key) - 1` bytes for the next piece, so a key is never split across two pieces.
   - It stays the same across restarts. Recipes pass it through the **environment**, never the
     command line, which any local process can read.
5. **Health.**
   - Until the port listens, each round is a bare TCP connect bounded by `interval_ms`. On Windows a
     connect to a closed loopback port is not refused at once: the stack re-sends the SYN after each
     RST for about 2 s. Waiting on HTTP probes would therefore notice the listener up to 0.5 s late.
   - Once the port listens, the host sends `GET http://127.0.0.1:{port}{path}` every `interval_ms`
     until the server answers 2xx.
   - Any other HTTP answer (llama-server returns 503 while loading) means "alive, keep waiting".
   - An exit or a timeout fails the attempt, and the process is killed.
   - The HTTP client uses no proxy (system or `HTTP_PROXY`), follows no redirects and keeps no pooled
     connections.
6. **Restart.**
   - `on_failure` waits `backoff_ms · 2^(n-1)` (capped at 30 s) and allows `max_restarts`
     consecutive failures. A helper that ran *ready* for 60 s (counted from its successful health
     check, not from its launch) resets the budget. A model that loads for longer than 60 s and
     crashes soon after becoming ready therefore still uses up the budget and ends in `failed`,
     which is what lets P4 fall back to another translator.
   - A health timeout or a crash during the first start also uses the policy.
   - Deterministic errors are never retried: the program is missing, the loader refused it (missing
     or mismatched DLL), the port is busy, or the helper broke the loopback policy.
7. **Stop.**
   - Close stdin (EOF means "quit" for stdio helpers), wait `stop_grace_ms`, then `TerminateJobObject`
     and wait for the process to be reaped. A stop marks stdin as closing: client writes are refused
     from then on, and if a client write holds the pipe, the pipe is closed right after that write.
     A client that writes in a loop therefore cannot hold off the EOF. A write stuck on a full pipe
     is ended by the terminate.
   - `spawn` and `stop` run on detached tasks. A caller that stops waiting (a timeout, a dropped
     future) never leaves a helper half-started.
   - A stop during a slow start cancels the start at once. Spawning an existing name replaces the
     helper.
   - The last request wins. Each `spawn` and `stop` takes a per-name ticket when it is called. A
     request that is no longer the latest once it holds the per-name gate does nothing. A spawn
     installs its helper only while its ticket is still the latest, which is checked under the
     registry lock. So a `stop` issued while an earlier `spawn` still waits (for example behind the
     replacement of the old helper) cancels that spawn, and nothing keeps running.
   - `shutdown_all` (`RunEvent::Exit`) kills everything, waits at most 3 s and refuses later spawns.
8. **Logs.**
   - stdout and stderr are split at `\n`, `\r` and 2 KB, cleaned of ANSI codes and redacted.
   - They go to `tracing` target `sigil::helper` at INFO, rate limited to a burst of 200 then 50
     lines/s, and to a 400-line ring buffer per helper (IPC `get_helper_logs`).
   - Lifecycle problems are logged at WARN, so the UI log shows them.
   - Error messages include the last 4 output lines.
9. **Stdio-protocol helpers** (`stdio: protocol`, for Bergamot's JSON lines):
   - stdout is kept for the client through `take_stdio`, once per process;
   - stderr still goes to the log;
   - the stdin writer is shared with the host, so a stop can still close it.
10. **Isolated workers.** `run_worker(&WorkerSpec)` covers the `--headless vc-import-worker` child
    (P3-B). It uses the same job with a memory cap and a wall-clock limit, captures bounded output,
    and reports timeouts and memory-limit hits.
11. **llama-server recipe.** `recipes::llama_server(exe, model, opts)`:
    - arguments `--host 127.0.0.1 --port {port} -m <gguf>`, plus `-c 4096 -np 1` by default and
      optional `-ngl`, `-fa`, `-t` and `--alias`;
    - the key in `LLAMA_API_KEY`;
    - `LLAMA_ARG_*` and `LLAMA_API_KEY*` removed from the inherited environment, so a user's global
      llama.cpp settings cannot move the bind or the key;
    - health `/health` with a 120 s load timeout;
    - restart on failure 3 times with a 1 s backoff;
    - EcoQoS off.

    llama-server keeps `/health` and the model list public; inference routes need the key.
12. **Runtime packs** (`src-tauri/model-catalog/runtime-llama.json`, status `published`, license
    MIT): the official ggml-org llama.cpp **b11146** Windows x64 zips. That build is the binary set
    of llama.cpp v0.5.0, released 2026-09-23, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`.

    | Id | Variant | Files (size, sha256) |
    | --- | --- | --- |
    | `runtime/llama-server-vulkan` (default GPU: NVIDIA, AMD, Intel) | `vulkan` | `llama-b11146-bin-win-vulkan-x64.zip` 32,127,004 B `55a378aa…4d7bd6` |
    | `runtime/llama-server-cpu` (fallback) | `cpu` | `llama-b11146-bin-win-cpu-x64.zip` 18,560,055 B `14cf1303…2fac1d1` |
    | `runtime/llama-server-cuda13` (opt-in) | `cuda13` | `llama-b11146-bin-win-cuda-13.4-x64.zip` 149,758,833 B `b1866c0c…41f047`, plus `cudart-llama-bin-win-cuda-13.4-x64.zip` 423,535,356 B `738f8c25…fbe668` (CUDA 13.4 runtime DLLs, same folder) |

    - Only `llama-server.exe`, `*.dll` and `LICENSE*` are extracted. `llama-server.exe` is a 9 KB
      launcher for `llama-server-impl.dll`; the CPU backends are `ggml-cpu-*.dll` variants chosen by
      CPUID at run time.
    - The CUDA pack ships its own CUDA 13.4 runtime and does not reuse Sigil's NVIDIA ORT pack
      (CUDA 13.2). A binary built against 13.4 may need newer cudart/cuBLAS exports than 13.2 has.
    - Hashes are GitHub's release-asset digests. The Vulkan zip was also downloaded and re-hashed
      here.

## Evidence

All results are from the H2 dev rig (Windows 11 Pro 26200, Rust 1.94.0, debug build).

- `cargo test --manifest-path src-tauri/Cargo.toml process_host`: **58 passed**, 4 ignored, about
  5 s, repeated with no failures (after the review fixes below). The first version (47 tests) also
  ran with `--test-threads=2` and `--test-threads=1`, left no `sigil-*` processes behind, and the
  full crate suite passed then (660 passed, 12 ignored).
- The helper in these tests is the test binary itself, re-run as a fake: an HTTP server, a crasher,
  a sleeper, a process tree, a memory hog, a JSON-lines echo, or a fake Sigil. Key cases:
  - `helpers_die_when_sigil_is_killed`: the plan's manual check, automated. A fake Sigil starts a
    helper through the real process host, and the helper starts a grandchild. The fake Sigil is then
    killed with `TerminateProcess`, which is what Task Manager does. Both the helper and the
    grandchild exit within the 10 s wait (in practice immediately).
  - `shutdown_all_kills_every_helper`: runs in a child process. Helper and grandchild die, and later
    spawns are refused.
  - `stop_kills_the_whole_process_tree`: the grandchild cannot escape the job.
  - `health_timeout_kills_the_helper_and_fails`: fails after 1.5 s, and the pid is dead.
  - `port_collision_is_retried_on_another_port`: the forced first port is held by another socket. The
    retry uses a new port, and `restarts` stays 0.
  - `fixed_port_in_use_is_rejected_without_starting`.
  - `crashed_helper_is_restarted_then_given_up`: 2 restarts, then `failed`, with "gave up after 2
    restarts" and the crash output in the error.
  - `restart_brings_a_crashed_helper_back`: new pid, same key, healthy again.
  - `never_policy_marks_a_crash_failed`.
  - `spawning_the_same_name_replaces_the_helper`.
  - `stop_during_a_slow_start_cancels_it_promptly`.
  - `abandoning_the_spawn_future_still_finishes_the_start`.
  - `stop_closes_stdin_first_for_a_graceful_exit`: exit code 0, not killed.
  - `protocol_helper_talks_over_stdio_and_stops_gracefully`.
  - `helper_memory_cap_is_reported`: a 4 GB allocation under a 1024 MB cap fails, and the error says
    "ran into its 1024 MB memory limit".
  - Review fixes. Each of these tests fails when its fix is disabled:
    - `slow_loading_helper_that_crashes_after_ready_still_exhausts_restarts`: 1.2 s load, crash
      0.3 s after ready, stable period 1 s. It ends `failed` after 2 restarts. With the old
      launch-time reset it looped (27 restarts in 40 s).
    - `a_helper_that_ran_ready_long_enough_gets_a_fresh_restart_budget`: the reset still works.
    - `helpers_start_with_error_dialogs_disabled`: a fake Sigil sets its error mode to 0. Its worker
      starts with all three `SEM_*` bits, and its own mode is still 0 afterwards. A copy of
      `hostname.exe` whose first import is renamed to a missing DLL ends with 0xC0000135 at once.
    - `a_helper_with_a_missing_dll_fails_at_once_and_is_not_restarted`: "could not start (exit
      code 0xC0000135: a required DLL was not found)", `restarts` 0, despite `max_restarts: 3`.
    - `stop_cancels_a_spawn_queued_behind_a_replace`: the queued spawn returns "start cancelled",
      and the final state is `stopped` with no process.
    - `a_foreign_server_on_the_port_is_a_collision_and_never_gets_the_key`: a local HTTP server on
      the forced port answers 200, and the helper lingers after its failed bind. The host retries
      on another port, kills the lingering instance, and the foreign server receives no request.
      `health::a_foreign_listener_is_reported_and_never_sent_a_request` covers the same at the
      probe level.
    - `stop_reaches_a_protocol_helper_whose_client_keeps_writing`: a client writes 64 KB chunks in a
      loop, and the stop still ends with exit code 0 in well under the 10 s grace. The old code
      waited the full 10 s.
    - `logs::a_key_straddling_the_long_line_cut_is_redacted`: every offset around the 2 KB cut and
      three read-chunk sizes; no 8-character run of the key is left in any piece.
  - `healthy_helper_gets_a_loopback_port_and_key_then_stops`: 503 is held until ready. `/auth` returns
    200 with the key and 401 without it or with a wrong key. The key is redacted in the logs.
  - Worker tests: output capture, a time limit of 0.8 s, a memory cap with `memory_limit_hit`, and
    bounded output.
  - `ports::listener_table_finds_our_own_loopback_listener`: the OS table sees the test's own
    `127.0.0.1` listener with the right pid and does not flag it. Row parsing and the address
    classification are unit-tested (`0.0.0.0`, `::`, LAN IPs and mapped addresses are exposed;
    `127.x` and `::1` are not).
  - `health::notices_a_late_listener_within_an_interval`: a server that binds 300 ms late is seen
    as ready less than 200 ms after the bind.
- **Latency.** Measured with `measure_start_stop_latency`, which is ignored by default (run it with
  `-- --ignored --nocapture`). It uses the fake HTTP helper, 10 starts and stops, debug build, 50 ms
  health interval.

  | Stage | p50 | max |
  | --- | --- | --- |
  | spawn → process running in its job | 3 ms | 6 ms |
  | helper's own start-up (to its first output line) | 19 ms | 23 ms |
  | host notices 2xx, and the audit is done | 54 ms | 55 ms |
  | **total** | **76 ms** | **80 ms** |
  | stop → reaped | 1.9 ms | 2.5 ms |

  The first design took 635 ms p50. HTTP-only probes waited on the SYN retries, and the connect-based
  exposure audit spent 300 ms per start timing out on LAN addresses where nothing listens. The TCP
  pre-probe and the listener-table audit removed both.
- **Pack install.** `Sigil.exe --headless fetch runtime/llama-server-vulkan`, run with
  `SIGIL_LOCAL_ASSETS=%LOCALAPPDATA%\sigil-dev-cache\llama.cpp\b11146` and a scratch
  `SIGIL_APP_DATA_DIR`, verified the sha256 and extracted 90.5 MB into
  `runtimes/llama-server-vulkan/b11146/vulkan/`: `llama-server.exe`, `llama-server-impl.dll`, the
  ggml DLLs and `libomp.dll`.
- **Not measured yet: llama-server itself and Hy-MT2 1.8B.** This session's permission policy did
  not allow running the downloaded `llama-server.exe`, so the 1.1 GB Hy-MT2 GGUF was not fetched.
  The smoke test exists as `process_host::tests::llama_server_smoke`, which is ignored and driven by
  environment variables. It checks four things:
  1. llama-server starts through the recipe and `/health` reaches 200;
  2. the exposure audit finds no LAN listener;
  3. `/v1/chat/completions` without the key returns 401;
  4. with the key, it translates a line and prints the load time, the first-translation time and
     llama-server's `timings`.

  After those checks, the test stops the helper and asserts that the process has exited.

  Manual procedure, in PowerShell from the repo root:

  ```powershell
  # 1. Runtime pack (32 MB): from the network, or offline from the dev cache
  $env:SIGIL_LOCAL_ASSETS = "$env:LOCALAPPDATA\sigil-dev-cache\llama.cpp\b11146"   # optional
  src-tauri\target\debug\sigil.exe --headless fetch runtime/llama-server-vulkan
  # 2. Hy-MT2 1.8B Q4 GGUF (~1.1 GB): the P4-B catalog entry, or any local copy
  $env:SIGIL_TEST_LLAMA_SERVER = "$env:APPDATA\com.sigil.app\runtimes\llama-server-vulkan\b11146\vulkan\llama-server.exe"
  $env:SIGIL_TEST_LLAMA_GGUF   = "<path>\Hy-MT2-1.8B-Q4_K_M.gguf"
  $env:SIGIL_TEST_LLAMA_NGL    = "999"   # all layers on the GPU; omit for CPU
  cargo test --manifest-path src-tauri/Cargo.toml llama_server_smoke -- --ignored --nocapture
  ```

  Then run the plan's Task Manager check in the real app once P4-B's UI can start the server. End
  `Sigil.exe` in Task Manager, and `llama-server.exe` disappears from the Details tab. The automated
  equivalent is `helpers_die_when_sigil_is_killed`.

## Consequences

- **P4-B (`llm_server`)** builds on `process_host::spawn(recipes::llama_server(..))`. It:
  - chooses the variant (Vulkan by default, CUDA 13 opt-in, CPU as the fallback) and the `-ngl`/`-c`
    values from `resources::recommend`;
  - pins the GGUF catalog in `llm-gguf.json` (Hy-MT2 1.8B Q4 first);
  - reads status through `get`, `handle` and `logs`;
  - uses `priority: below_normal` for the CPU pack, so CPU inference yields to VRChat and the voice
    changer.

  After a restart the port can change (only when the old one was taken), so clients re-read
  `handle(name)`.
- **The key stays out of every status surface.** `HelperHandle.api_key` is never serialized, and
  `list_helpers`, the state event and `get_helper_logs` never carry it. *Superseded in part by P4:*
  the frozen llm-server contract (P4-S0) returns the key to TypeScript through
  `llm_server_endpoint` / `llm_server_start`, so the webview can call the loopback server directly
  (LAN and CORS-blocked hosts still go through `llm_proxy_chat`, which adds headers in Rust). The key
  is random per session, the server listens on `127.0.0.1` only, and `llm_server_status` never
  contains it.
- **P3-B** runs the import worker with `run_worker` (for example `memory_limit_mb: 4096`,
  `timeout_ms: 120_000`).
- **P4-E** (Bergamot) uses `stdio: protocol` and `take_stdio`. **P5-E / P6-E** use HTTP specs like
  the llama-server recipe.
- **Upgrading llama.cpp** means new pins for all three entries, plus the smoke test above. ids never
  change. Bump `version` and the file names.
- **Firewall:** helpers that bind `127.0.0.1` never trigger the Windows Firewall prompt. The exposure
  test (`non_loopback_listener_is_rejected`) listens on `0.0.0.0` on purpose, so it is opt-in:
  `SIGIL_TEST_EXPOSURE=1`, run with `--ignored`.
- **Rollback and fallback:** if a helper misbehaves, stop it. The OpenAI-compatible client can point
  at a user-run Ollama or LM Studio (plan fallback). If `NtResumeProcess` were ever unavailable, the
  host spawns without `CREATE_SUSPENDED` and assigns immediately, which reopens a microsecond-scale
  race.
- **P4 update (P4-B, recorded by P4-I0).** `services/llm_server` is built on this ADR:
  - Flags checked against the b11146 `--help` output: every launch adds `--no-webui -fit off
    --reasoning off --log-colors off`, plus `--poll 0` on full GPU offload and `--device none` on the
    CPU pack. The recipe's own flags (`--host`, `--port`, `-m`, `-c`, `-np`, `-ngl`, `-fa`, `-t`,
    `--alias`) all exist in b11146.
  - **Multi-GPU pinning.** With more than one hardware GPU the server is pinned to the primary one:
    `--device <name>` from `llama-server --list-devices`, else `-sm none`. The dev PC lists its
    RTX 4090 and its AMD integrated GPU as Vulkan0 / Vulkan1; that output is the parser's fixture.
  - The server holds a `claim_external_vram` sized to the plan, then resized to the device buffers
    llama-server logs (plus 100 MB Vulkan / 400 MB CUDA base). If the log format differs, the claim
    stays at the plan estimate.
  - A fake llama-server built into the test binary runs through the real start path
    (`server::start_launch` → `process_host::spawn`): start, ready, reuse, replace, stop during a slow
    load, and a crash loop ending `failed` with the claim released (`cargo test llm_server`).
  - **The live llama-server + Hy-MT2 1.8B smoke is still manual.** The P4 session's permission
    classifier refused to run llama-server with a model. The verified GGUF (sha256 `dc5f44fc…c06699`)
    and the b11146 Vulkan zip are in `%LOCALAPPDATA%\sigil-dev-cache\p4b-local-assets`. Run it with
    the P4-B test (the older `llama_server_smoke` above still works for the bare recipe):

    ```powershell
    $env:SIGIL_TEST_LLM_RUNTIME = '<folder containing llama-server.exe>'
    $env:SIGIL_TEST_LLM_GGUF    = "$env:LOCALAPPDATA\sigil-dev-cache\p4b-local-assets\Hy-MT2-1.8B-Q4_K_M.gguf"
    cargo test --manifest-path src-tauri/Cargo.toml real_model_end_to_end -- --ignored --nocapture
    ```

    Then the in-app check: Translation → AI model (LLM) → Local model server, Test load, translate a
    line, end `Sigil.exe` in Task Manager and confirm `llama-server.exe` is gone.
- **Known limits:**
  - The exposure audit runs once, after the helper becomes ready. Sockets a helper opens later are
    not re-audited (a periodic audit in the supervisor is a cheap follow-up).
  - The audit sees TCP only. A helper that serves UDP would need a UDP table check.
  - The port-owner check runs when the port first accepts connections. If the listener table shows
    no listener three rounds in a row although connects succeed, the host proceeds unverified
    rather than time out a working helper.
  - A helper that relaunches itself and exits (a "launcher") would be killed with its job. llama-server
    does not do this.
  - The peak job memory figure includes a refused allocation, so it is informational only.
