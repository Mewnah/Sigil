# P7 speech benchmark campaign

Everything the P7 measurement campaign (plan section 8: STT accuracy on VRChat-like audio, CPU
speed, the GPU pick matrix under load, MT blind A/B, the TTS listening test) needs outside the Rust
runner: the frozen file formats, the corpus builder, the scorers, the human-check pages
(`tools/listening-test/`) and the Python environment. Scaffolded by P7-S0 while P3B measured the
voice engine; the runner itself, `Sigil.exe --headless speech-bench`, is P7-A's
(`src-tauri/src/services/bench/speech.rs`).

| Path | What it is |
|---|---|
| `schemas/` | The frozen JSON Schemas (draft 2020-12): spec, report, export, corpus manifest, corpus recipe, text-set lines |
| `examples/` | One spec per mode, plus a report and its export (format examples: their numbers are made up) |
| `corpus/build_corpus.py` | Builds the STT corpus from the prefetched FLEURS / FMA / ESC-50 (and real drop-in clips) |
| `corpus/build_text_sets.py` | Builds the MT sets (WMT24++, FLORES-200 and its European directions `flores200-eu`, BSD, the P4 casual lines) and the TTS prompts |
| `corpus/recipe.json` | The frozen corpus recipe (languages, selection, levels, conditions, subsets) |
| `score/score_stt.py` | WER / CER per engine, language and condition, bootstrap intervals, paired comparisons |
| `score/score_mt.py` | chrF2 (sacrebleu), BLEU, MetricX merge, paired bootstrap, A/B pair export |
| `score/metricx.py` | MetricX-24 runner that only runs when torch / transformers and the checkpoint are present |
| `score/score_tts.py` | TTS pre-screen: TTFA, RTF, intelligibility, DNSMOS, speaker cosine |
| `score/make_export.py` | Reference implementation of the friend-export rules |
| `score/merge_exports.py` | Friends' exports and reports into one speed table per task |
| `validate.py` | Validates any campaign file against its schema (plus spec cross-references and export privacy) |
| `setup-venv.ps1`, `requirements.txt` | The light Python environment (no torch) |
| `tests/` | Unit tests (`python -m unittest discover -s scripts/speech-bench -p "test_*.py"`) |
| `campaign/` | The P7-A campaign: the specs it ran, the run / lock / process-sampling drivers, the table printers and the real-app footprint harness ([`campaign/README.md`](campaign/README.md)); results in [`docs/benchmarks/SPEECH_BENCHMARKS.md`](../../docs/benchmarks/SPEECH_BENCHMARKS.md) |

Outputs never go into the repo: the venv, the corpus and every run live under
`%LOCALAPPDATA%\sigil-dev-cache\p7\` and the git-ignored `bench-results/`.

## Environment

```powershell
pwsh scripts/speech-bench/setup-venv.ps1          # creates %LOCALAPPDATA%\sigil-dev-cache\p7\venv
$py = "$env:LOCALAPPDATA\sigil-dev-cache\p7\venv\Scripts\python.exe"
& $py -m unittest discover -s scripts/speech-bench -p "test_*.py"
```

The script installs the pinned packages of `requirements.txt` one at a time without dependency
resolution (the file pins the whole transitive set), two seconds apart, and skips what is already
installed. Python 3.12 (3.11 works). Environment variables the campaign uses:

| Variable | Meaning |
|---|---|
| `SIGIL_APP_DATA_DIR` | The model store the runner reads: `%LOCALAPPDATA%\sigil-dev-cache\appdata-p7` holds every prefetched P7 model |
| `SIGIL_LOCAL_ASSETS` | Local files the store installs from without network: `%LOCALAPPDATA%\sigil-dev-cache\p7\local-assets` (Bergamot models; Sigil-built packs, see prerequisites) |
| `SIGIL_P7_ROOT` / `SIGIL_P7_CORPUS` | Expanded in spec paths; default `%LOCALAPPDATA%\sigil-dev-cache\p7` and `<root>\corpus` |
| `SIGIL_BENCH_RIG` | Optional rig label written into reports (`H1`, `H2`, `H3`, `friend-3`); never the host name |
| `SIGIL_GPU_LOCK` | The GPU timing lock file (see `scripts/bench/gpu-lock.ps1`) |

## Pipeline

1. **Corpus** (done once; already built on the dev PC, see "Corpus").

   ```powershell
   & $py scripts/speech-bench/corpus/build_corpus.py --workers 2 --priority idle --hash-sources
   & $py scripts/speech-bench/corpus/build_text_sets.py
   & $py scripts/speech-bench/corpus/build_corpus.py --verify       # re-hash every file
   ```

   Real VRChat clips join later through the drop-in folder (`corpus/real.py` has the layout):
   record, drop, then `build_corpus.py --real-only`.

2. **Runs** (P7-A's runner; timing runs under the GPU timing lock on the dev rig):

   ```powershell
   $env:SIGIL_APP_DATA_DIR = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p7"
   $env:SIGIL_LOCAL_ASSETS = "$env:LOCALAPPDATA\sigil-dev-cache\p7\local-assets"
   $exe = "src-tauri\target\release\sigil.exe"
   & $exe --headless speech-bench scripts\speech-bench\examples\spec.accuracy.json --out bench-results\speech\accuracy
   pwsh scripts/bench/gpu-lock.ps1 -Run P7-A -Wait 60 -Command "$exe --headless speech-bench scripts\speech-bench\examples\spec.gpu_under_load.json --out bench-results\speech\gpu"
   ```

   Every run writes `report.json` (the measurement), `report.md`, `hyps/<run>.<engine>.jsonl`,
   `audio/<engine>/*.wav` (TTS) and `export.json` (what a friend sends back).

3. **Scoring**

   ```powershell
   $c = "$env:LOCALAPPDATA\sigil-dev-cache\p7\corpus"
   & $py scripts/speech-bench/score/score_stt.py --manifest $c\corpus.json --report bench-results\speech\accuracy\report.json --out bench-results\speech\accuracy\scores --items
   & $py scripts/speech-bench/score/score_mt.py --report bench-results\speech\mt\report.json --set $c\mt\wmt24pp.jsonl --set $c\mt\flores200.jsonl --set $c\mt\bsd.jsonl --out bench-results\speech\mt\scores --export-pairs bench-results\speech\mt\pairs
   & $py scripts/speech-bench/score/metricx.py --check --python <a Python with torch>      # optional, see prerequisites
   & $py scripts/speech-bench/score/score_tts.py --report bench-results\speech\tts\report.json --prompts $c\tts\prompts.jsonl --out bench-results\speech\tts\scores
   ```

   mt-bench output folders (`Sigil.exe --headless mt-bench`) are scored the same way with
   `score_mt.py --mt-bench <folder>`.

4. **Human checks** (`tools/listening-test/README.md`): `make_mt_test.py` from the exported pairs,
   `make_tts_test.py` from the `tts_listening` renderings; zip `test/`, friends open `index.html`
   (offline, any browser) and send back their `answers-*.json`; `aggregate.py` scores them.

5. **Friend exports**: a friend runs the same spec (speed modes need no corpus beyond the quick
   subset) and sends `export.json`; `merge_exports.py` checks each against the privacy rules and
   tabulates every machine. `make_export.py` re-derives an export from a report.

6. **Decisions** (P7-A, P7-I0): the router table (`src-tauri/model-catalog/stt-routing.json`), the
   translation default chain and interim engine (`src/core/services/translation/router.ts`), the TTS
   presets and `resources::policy` footprints change only from these numbers. Rule for a default
   flip of the **own-microphone** STT pipeline (user decision 2026-10-09: a streamer's own mic hears
   no game audio through headphones, so the bleed conditions `music_*`, `sfx_*`, `babble_*` and
   `vrchat_mix` take no part in the choice, not even as a tie-breaker): rank the engines on `clean`
   plus the mic-side conditions `room_light` and `mic_codec` (pooled error rate, paired utterance
   bootstrap); the lowest wins, an engine within the paired interval of zero and at most 1.5 points
   of it ties it, and among tied engines the cheapest takes the language, except that Granite and Nemotron
   are never a final and a language whose files lack an own-mic condition for some engine gets no winner
   ([`campaign/own_mic.py`](campaign/own_mic.py) prints the table and both rankings, and exits 2 on such a
   partial file list). The bleed
   numbers are for the "other players' captions" (loopback) feature, whose input is VRChat's own
   audio, and are tabulated apart. A flip also has to meet the latency / VRAM budget of its tier
   under the H1 load profile (and on at least one friend export when the tier is H1 / H3). Results go to
   `docs/benchmarks/SPEECH_BENCHMARKS.md` with the corpus `content_sha256`, the normalizer stamp and
   the rig.

## Frozen formats

Every format carries its id in a `schema` field; `validate.py <file>` checks it. Change a format only
through an INTEGRATE step: fields may be added (optional, defaults keep old files meaning the same),
never renamed or removed.

| Schema id | File | Written by | Read by |
|---|---|---|---|
| `sigil-speech-bench-spec/1` | `schemas/speech-bench-spec.schema.json` | campaign authors (`examples/spec.*.json`) | the runner (P7-A) |
| `sigil-speech-bench-report/1` | `schemas/speech-bench-report.schema.json` | the runner | the scorers, docs |
| `sigil-speech-bench-export/1` | `schemas/speech-bench-export.schema.json` | the runner, `make_export.py` | `merge_exports.py` |
| `sigil-speech-corpus/1` | `schemas/speech-corpus.schema.json` | `build_corpus.py` | the runner, `score_stt.py` |
| `sigil-speech-corpus-recipe/1` | `schemas/speech-corpus-recipe.schema.json` | `corpus/recipe.json` | `build_corpus.py` |
| text-set lines | `schemas/speech-text-sets.schema.json` (`mt_line`, `tts_prompt`) | `build_text_sets.py` | the runner, `score_mt.py`, `score_tts.py` |
| `sigil-listening-{mt,tts,key,results}/1` | `tools/listening-test/schemas/` | the page builders and pages | `aggregate.py` |

### The spec (`Sigil.exe --headless speech-bench <spec.json> [--out <dir>]`)

- **Engines** name a task (`stt`, `mt`, `tts`), a backend, a model (catalog id, or an absolute path
  for bring-your-own), an `execution_provider` (`auto`, `cpu`, `cuda`, `dml`, `vulkan`), optional
  `variant` / `threads` / `languages` / `directions`, and backend `options`. Backends: stt =
  the speech plugin's `EngineKind` wire names (`whisper`, `http`, `sherpa-online`,
  `sherpa-offline`, `transcribe`) plus `pipeline` (the shipped pipeline: Silero VAD, router,
  two-pass), `crispasr` (R2T2, prerequisite) and `hypotheses` (JSON lines made elsewhere); mt =
  mt-bench's `nllb`, `openai`, `hypotheses` plus `bergamot` and `llama-server` (Sigil starts the
  bundled helper); tts = `local-tts`, `tts-helper` (`options.helper` = `qwen3`, `magpie`, `breeze`,
  `omnivoice`, `s2`), `openai-speech`, `rendered`.
- **Runs** pick a mode, engines (default: every engine whose task the mode takes) and options:

  | mode | tasks | measures |
  |---|---|---|
  | `accuracy` | stt | hypotheses of every selected corpus item (scored later) |
  | `cpu_speed` | stt, mt, tts | compute, RTF, latency, process CPU on the CPU, optionally under the load simulator |
  | `gpu_under_load` | stt, mt, tts | the same on GPU engines under the GPU / CPU load profile and the GPU lock, plus process VRAM |
  | `mt_quality` | mt | translations and latency over an MT set (`mt.set`) |
  | `tts_listening` | tts | renderings of every prompt (`tts.prompts`) with TTFA / RTF and optional pre-screens |

  `feed.kind` decides how STT audio arrives (`utterance`: whole clip; `stream`: chunks; `pipeline`:
  the real pipeline), `feed.pace` whether it is paced in real time. `load_profile` reuses the bench
  contention profiles (`h1_vrchat_obs`, `h2_load70`, `h2_load70_dml`, `h2_load70_oop`,
  `h3_cpu_game`, `custom`) and `load` is exactly the bench's `LoadSpec`; `external_load` describes a
  real load (a friend's VRChat). `gpu_lock.require` (default on for speed modes) makes the runner
  refuse a timing run unless the lock file names `gpu_lock.tag`; the campaign script takes the lock,
  the runner only checks it. `thresholds` gate aggregate keys.
- **Paths** expand `${NAME}` from the environment (`SIGIL_P7_ROOT`, `SIGIL_P7_CORPUS` have defaults)
  and are relative to the spec file. Keys never appear in a spec (`options.apiKeyEnv` names a
  variable); `validate.py` flags option names that look like secrets.

### The report

`report.json` is the measurement, not the verdict: per run and engine an `aggregate`, `groups` (by
language and condition, direction, or language) and `items`. Item text is the hypothesis,
translation or back-transcription; scores are never in a report (one normalizer implementation, in
Python). Aggregate keys are frozen (`schemas/speech-bench-report.schema.json`, `$defs/aggregate`):
`items`, `errors`, `audio_s`, `wall_ms`, `rtf_mean/_p50/_p95`, `compute_ms_p50/_p95/_p99`,
`final_latency_ms_p50/_p95/_p99`, `first_partial_ms_p50/_p95`, `latency_ms_mean/_p50/_p95/_p99`,
`chars_per_s`, `ttfa_ms_p50/_p95`, `cpu_cores_mean`, `cpu_ms_per_audio_s`, `vram_peak_mb`,
`vram_delta_mb`, `ram_peak_mb`, and `chrf` as a quick look. Percentiles are nearest-rank;
`spec_sha256` is the sha256 of the spec's canonical JSON (sorted keys, no whitespace). The
environment block is the bench's `Environment` plus physical cores, RAM, power plan / mode, runtime
packs and the optional rig label; a run's `state` is the bench's `RunState` (clocks, other GPU
processes, warnings).

### The export (what friends send back)

`export.json` = the report without audio, without `files`, without absolute paths (rewritten to
`<path>/<file name>`; folders may hold spaces, as in `C:\Users\Jane Doe\...`, and a bare profile folder leaves no name),
without host or user names, with `spec_summary` instead of the spec and a bring-your-own model as `byo`.
It also names no other program: a run state's other GPU users (the image names of whatever else used
the GPU, such as a game) become a count (`state.summary.other_top_processes_count`), the run warning
that lists them keeps only the GPU share, and a Windows power plan that is not one of the built-in
ones (Balanced, High performance, Power saver, Ultimate Performance; a localized Windows names them
differently) reads `custom`. Item text stays only when the spec's `output.export.include_text` is true,
and never for real clips (`real/...` ids). `score/make_export.py` is the reference
implementation and `validate.py` checks the rules (`privacy_errors`, which also flags a path that
was only partly replaced).

### The Rust side (P7-A)

- `pub fn headless_speech_bench(spec: &Path, out: Option<&Path>) -> Result<String, String>` in
  `bench::speech` (same contract as `bench::headless_bench` and `mt_local::bench::headless_mt_bench`:
  default output `bench-results/speech/<name>-<timestamp>/`, `Ok` text printed by the CLI).
- Reuse, do not mirror: `bench::spec::LoadSpec` / `GpuLoadKind` for `load` / `gpu_load_kind`,
  `bench::contention` profiles for `load_profile`, `bench::load_sim::LoadHandle` and its CPU tally,
  `bench::gpu_state::RunStateProbe` for `state`, `bench::report::Environment` for the environment,
  `speech::engine_api::EngineSpec` / `RouteRequest` for STT engines, the mt-bench engines for MT.
- Serde structs with `deny_unknown_fields` (except `$comment`), defaults as in the schema.
- Contract test: deserialize every `scripts/speech-bench/examples/spec.*.json` and
  `report.example.json` with the Rust types (the examples are the fixtures; both sides must keep
  accepting them).

**As built (P7-A1, `src-tauri/src/services/bench/speech.rs` and `speech/`).** Where the schema left a
choice open, the runner does this (change the runner and this list together):

- `$comment` is removed before the strict parse and kept in the spec the report echoes and hashes;
  `spec_sha256` hashes the spec as written (comments included), canonical as `score/common.py` does.
- A run's `select` replaces the corpus' default `select` (it does not merge), so `{"kinds": ["real"]}`
  does not inherit `subset: core`. `limit_per_language` takes the first k utterances of a language by
  `rank` and every condition of them. Items of paced (`realtime`) feeds are shuffled with `seed`.
- `load` replaces the profile's GPU part and / or CPU part individually; `none` plus a `load` applies the
  load; `custom` needs one. `h2_load70_dml` / `_oop` always run the GPU load on DirectML, the other
  profiles on `cuda` when an engine of the run asks for CUDA, else DirectML.
- `gpu_lock`: path from the spec, else `$env:SIGIL_GPU_LOCK`, else the `$DefaultLockPath` read from
  `scripts/bench/gpu-lock.ps1`; the owner is the lock's JSON `tag` (or the text of a hand-written lock);
  the tag matches when equal or followed by a space.
- Percentiles are nearest-rank in integer arithmetic; per-item timings of `repeat` passes are the median,
  text and audio come from the first pass. `rtf = compute_ms / 1000 / audio_s` with `audio_s` the clip
  (lead / tail silence of stream feeds is extra work for the engine, not extra audio).
- STT timings: an `utterance` feed's `final_latency_ms` is its decode time. A `stream` feed pushes lead
  silence, the clip and tail silence in `chunk_ms` blocks; `compute_ms` is the engine's busy time
  (push + end), `first_partial_ms` runs from the first fed sample (minus the lead silence at realtime
  pace, where the lead is paced with the clip), `final_latency_ms` is the time from the end of the fed audio to the final (end_stream alone at
  asap pace). At realtime pace the feed itself runs inside the item's deadline, so `timeout_s` is the
  engine's time on top of the audio's own duration (lead, clip and tail), not a limit on the whole feed. An engine that cannot stream decodes once at the end of the paced clip (no partials; a
  warning says so). A `pipeline` feed (or the `pipeline` backend) runs the speech plugin's own session
  code on a private registry, one language group at a time, feeding the next item only when the previous
  one has settled, in 20 ms blocks like the capture hub; `final_latency_ms` is the pipeline's
  VAD-close-to-final latency of the item's last final, `compute_ms` (asap only) the wall time from the
  item's first sample to its last final. `first_partial_ms` runs from the item's first fed sample; the
  lead silence is pushed at once and takes no wall time, so at realtime pace it is also the time from the
  clip's start (the lead is not subtracted, unlike the stream feed, which paces its lead). A session
  loads the models, and there is one per run of same-language items: a real-time run's shuffled order is
  regrouped by language for a pipeline engine (order inside a language kept), and the engine row's
  `extra.sessions` counts the sessions, because their model loading sits inside the engine's timed window
  (`cpu_cores_mean`, `ram_peak_mb`, `wall_ms`) and the pipeline gets no warm-up item. A route whose only engine is a streaming one (the CPU tier
  without a final engine) is always fed at real time, because the pipeline skips the live text of a
  streaming engine that falls more than 3 s behind; the run's warnings say so.
- `crispasr` starts `crispasr --server` (pack by execution provider: cuda13, vulkan, cpu) under
  `process_host` and speaks to it through the app's HTTP STT engine (`/v1/audio/transcriptions`), so it
  is an utterance engine; `options.crispasrBackend` is its `--backend`. A catalog `model` id is passed
  as the entry's file (its first one, or `options.modelFile` as the catalog lists it), a bring-your-own
  path as that file.
- `vram_peak_mb` / `vram_delta_mb`: engines in this process are read per process through DXGI (peak and
  peak minus the reading before the engine loaded); engines in a helper process (`llama-server`,
  `tts-helper`, `crispasr`, `bergamot`) are read device-wide through NVML as the growth over the baseline
  (both keys then hold that growth). `cpu_cores_mean` is process CPU over wall time of the engine's timed
  passes minus the load simulator's CPU tally; an item's `cpu_ms` the same per item. `ram_peak_mb` is the
  process working-set peak. CPU-only runs never call DXGI / NVML.
- MT: nllb in this process (`options.beam`), `bergamot` through the helper, `llama-server` started with
  `llm_server::server::start` (variant from the execution provider, over-budget allowed unless
  `options.allowOverBudget` is false) and spoken to over its OpenAI endpoint, `openai` with
  `options.baseUrl` as given (include `/v1`); `context: previous` prepends the previous source line of the
  same document to LLM prompts. `chrf` in aggregates and groups is mt-bench's sacrebleu-equal chrF2.
- TTS: `local-tts` drives the production `local_tts` worker into a recording sink (first clause alone,
  as for the first sentence of a burst); `tts-helper` starts the helper with the app's planning code and
  streams `/audio/speech` as raw PCM (`ttfa_ms` = first audio byte); a rendering is written as 16-bit PCM
  WAV. The intelligibility, DNSMOS and speaker-similarity pre-screens run after the timed render; a
  pre-screen that cannot run is a note in the item's `extra` and a run warning, not an item error.
- `hypotheses` files accept `{item, text}` (STT, MT) and mt-bench's `{id, hyp}`. A line whose text is
  `null` (what the runner's own `hyps/` files hold for an item that failed) is skipped, so that item
  reads "no hypothesis" and the other lines of a runner-written file stay usable.
- `export.json` is checked with the privacy rules before it is written; a leak writes no export, adds a
  failure to the report and fails the run.
- A failed item, engine or threshold makes the process exit 1 (the files are written either way).
- **Limits the P7-A campaign found** (details and workarounds in [`campaign/README.md`](campaign/README.md)): the figures of an
  engine that runs in a helper process (`llama-server`, `crispasr`, TTS helpers, `bergamot`) do not include the helper's CPU and
  RAM (sample the process tree); `threads` of a `llama-server` engine is not forwarded to the server; a `pipeline` row at
  `pace: asap` includes the backlog of a streaming engine in its latencies (use `realtime`); `first_partial_ms` of a `stream`
  feed includes the clip's own leading silence.

## Corpus

Built on the dev PC on 2026-10-03 into `%LOCALAPPDATA%\sigil-dev-cache\p7\corpus\` (outside the
repo): **12,600 items, 39.9 hours, 5.4 GB** (9 languages x 100 utterances x 14 conditions), plus
`mt/` (4,046 MT lines) and `tts/prompts.jsonl` (171 prompts). The manifest's `content_sha256`
(sha256 of the sorted item ids and file hashes) is the corpus identity every score cites: the
dev-PC build is `459c2ff05c9274e0667d4a0897c1a88bb5cfacef549606522197e8147bb78f0f` (recipe
`d259200a3049...`, builder 3); `--verify` matched 12,600 of 12,600 files, and a separate
five-utterance English build reproduced its 25 items byte for byte.

- **Speech**: FLEURS test (google/fleurs at 70bb2e8, CC-BY-4.0) for en_us, ja_jp, cmn_hans_cn,
  ko_kr, es_419, de_de, fr_fr, ru_ru, pt_br. 3-20 s recordings, one per sentence, sentences in
  seeded order, and per sentence the recording of the gender chosen least so far (de and pt test
  sets are male only, ja has few female speakers). The babble pool (24 per language) never shares
  a sentence with the selection. Speech is brought to an active level of -26 dBFS +- 4 dB (seeded
  per utterance, the same gain under every condition): FLEURS levels range from about -20 to
  -66 dBFS.
- **Conditions** (`corpus/recipe.json`; SNR = active-level ratio, see `corpus/dsp.py`):

  | id | what |
  |---|---|
  | `clean` | the utterance |
  | `music_snr{20,10,5}` | music bleed: 30 s segments of 96 FMA commercial-use tracks (one 16 kHz shard) |
  | `sfx_snr{20,10,5}` | effects: 320 ESC-50 events of 16 room / game categories (keyboard, clicks, footsteps, claps, laughs, knocks, glass, fireworks, ...), about 4 per 10 s |
  | `babble_snr{20,10,5}` | three other FLEURS speakers of the same language |
  | `babble_xlang_snr10` | three speakers of other corpus languages |
  | `room_light` | synthetic reverb tail, RT60 0.35 s, direct-to-reverberant 6 dB |
  | `mic_codec` | 200-3800 Hz band, 8-bit mu-law, hiss at -60 dBFS |
  | `vrchat_mix` | speech in a light room + music through the room at 15 dB + effects at 18 dB + two talkers at 20 dB (about 12 dB combined), mic 100 Hz-7 kHz |

  Subsets: `quick` (clean, vrchat_mix), `core` (clean, music_snr10, babble_snr10, vrchat_mix),
  `snr_sweep`, `mic`, `all`, `real`. Within a language, `rank` orders the utterances, so
  `limit_per_language: k` selects the same k utterances under every condition.
- **Provenance**: every item records its FLEURS file and sentence id, the seed, the speech gain, the
  exact bank files, offsets and gains of every component and the processing chain;
  `_bank/bank.json` names each FMA track (title, artist, licence code, URL) and ESC-50 clip
  (Freesound id). Builds are deterministic: the same recipe, builder version and libraries give
  the same `content_sha256` (a test checks it).
- **Limitations** (state them next to any number): FLEURS is read speech, not VRChat chatter; the
  bleed is mixed digitally, not played through speakers; a babble talker may be the target's own
  speaker reading another sentence (FLEURS has no speaker ids); digits vs number words outside
  English count as errors for every engine alike. The real drop-in clips are the authority once
  they exist.

## Scoring

`score/normalize.py` (`sigil-norm/1`, stamped into every scores file): English goes through Whisper's
`EnglishTextNormalizer` (vendored with its Apache-2.0 / MIT attribution; spelled numbers to digits,
contractions, fillers, British spelling: Whisper's full `english.json` is used when copied to
`score/data/`, else Sigil's short table), other space-delimited languages through Whisper's
`BasicTextNormalizer` (diacritics kept; Russian ё -> е) and are scored by WER; ja, zh, yue and ko by
CER on grapheme clusters after NFKC, with punctuation, symbols and spaces removed (Traditional ->
Simplified for Chinese only when `opencc` is installed: recorded). Corpus rates are
`sum(errors) / sum(reference units)`. A failed item counts as an empty hypothesis (`--failed-as
empty`, default) or is dropped (`exclude`); both are counted. Intervals are percentile bootstraps
over items (1,000 resamples, seed 20261002); comparisons between engines or against `clean` draw
the same items for both sides. chrF2 is sacrebleu's `CHRF()` (the number mt-bench reports), with the
same paired bootstrap on its sufficient statistics.

## Remaining prerequisites

From the P7 prefetch manifest; each blocks only its own rows.

| Item | State | Next step |
|---|---|---|
| The Rust runner | implemented (P7-A1): `speech.rs` + `speech/`, `pub mod speech;` in `bench/mod.rs`, the `speech-bench` arm in `services/cli/mod.rs`. The P7-A campaign ran the sherpa, whisper, transcribe (CPU and CUDA), pipeline, nllb, llama-server (CPU and Vulkan) and local-tts backends; `http`, `crispasr`, `bergamot`, `tts-helper` and `openai-speech` have still never run against a live pack or server | see "As built" under "The Rust side" for what the runner decided and `campaign/` for the campaign |
| Bergamot runtime (`runtime/bergamot`) | unbuilt (only a fake pack); the six CJK models are in `p7/local-assets/mt-bergamot/` | `pwsh scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives` (one pinned source fetch), then `pwsh scripts/runtime-packs/bergamot/build.ps1 -SelfTest -Models "$env:LOCALAPPDATA\sigil-dev-cache\p7\local-assets\mt-bergamot"`; copy `sigil-bergamot-69455ac-s1-cpu.zip` to `p7\local-assets\runtime\bergamot\` and run `Sigil.exe --headless fetch runtime/bergamot mt-bergamot/en_ja mt-bergamot/ja_en mt-bergamot/en_zh mt-bergamot/zh_en mt-bergamot/en_ko mt-bergamot/ko_en` with the two variables above (ADR 0008 has the no-go fallback) |
| qwentts packs | built locally but not installed; the CUDA zip is a one-GPU `-dev` build whose name differs from the catalog's | dev rig: copy `%LOCALAPPDATA%\sigil-native\packs\sigil-qwentts-server-6fae929-s1-cuda13-dev.zip` to `p7\local-assets\runtime\qwentts-server-cuda13\sigil-qwentts-server-6fae929-s1-cuda13.zip` and the cpu zip to `p7\local-assets\runtime\qwentts-server-cpu\`, then `Sigil.exe --headless fetch runtime/qwentts-server-cuda13 runtime/qwentts-server-cpu` (the entries are unpinned: a dev build, or Import in Models & Runtimes). For friends: the five-architecture build, `pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13` (1.5-2 h, resumes where it stopped; run it while the PC is otherwise idle) |
| Confucius4-R2T2 runtime | transcribe.cpp 0.2.4 cannot load r2t2; both GGUFs are in `p7/stt/` | Decision for the campaign: CrispASR v0.8.40 (`runtime/crispasr-cuda13` / `-vulkan`, already in `appdata-p7`, the runtime P6-F uses for OmniVoice) with the cstr GGUF (`confucius4-r2t2-q8_0-crispasr`, `--backend qwen3`). P7-A adds the `crispasr` STT backend through `process_host` (reuse `tts_helpers::omnivoice`'s pack lookup; read `crispasr.exe --help` of the pinned release for its transcription flags). A catalog entry (`stt/confucius4-r2t2-q8_0`) and a transcribe.cpp re-pin wait for the measurement |
| MetricX-24 | ready on the dev PC with the system Python (see below); the P7 venv has no torch (on purpose) | `score/metricx.py --python <system python.exe> --pairs <pairs.*.jsonl> --out metricx.jsonl`, then `score_mt.py --metricx metricx.jsonl`. On another PC, in a separate venv, one package at a time: `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu`, `pip install transformers==4.39.3`, `pip install sentencepiece==0.2.0`, then `--check` and `--self-test` (CPU, about 5 GB RAM) |
| transcribe.cpp `vulkan` pack | never built (no Vulkan SDK on the dev PC) | install the Vulkan SDK, build with `scripts/native/build-runtime-packs.ps1`; until then non-NVIDIA GPU rows are CPU rows |
| Hy-MT2 1.8B 1.25-bit | not runnable (llama.cpp b11146 lacks the STQ1_0 kernel) | re-pin llama-server once STQ1_0 has an x86 path; low-memory CPU rows use the Q4 build |
| Magpie v2607 (ko, ar, pt-BR) | not run (needs the NeMo-Speech.cpp v0.2.0 pack and NVIDIA's 569 MB GGUF, which v0.1.0 cannot read) | `real_ggml_engine_speaks` with `SIGIL_GGML_MODEL=v2607` (`scripts/runtime-packs/nemo-speech/README.md`) after the user OKs the download; v2602 (en, es, de, fr, it, vi, hi) is installed for the listening test |
| Real VRChat clips | none yet | the user and friends record the raw mic (before any voice changer) with world audio, drop them into `corpus/real/<session>/` with transcripts and consent, then `build_corpus.py --real-only` |
| CAM++ absolute thresholds | the bench asset matches upstream at 4.4 dB (P5 outcome) | speaker cosine is fine for ranking clones; re-export CAM++ before using absolute thresholds |

### MetricX on the dev PC (checked by P7-S0, 2026-10-03)

`score/metricx.py --check --python %USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe`
is ready (Python 3.12.10, torch 2.3.1+cu121, transformers 4.39.3, the prefetched checkpoint,
tokenizer and code), and `--self-test --threads 2` on the CPU took 10 s: the correct German
translation scored 0.14, a wrong one 9.06 (0-25, lower is better). The P7 venv has no torch, so
pass `--python` with that interpreter. CPU runs hide CUDA from the worker; a CUDA run
(`--device cuda`) belongs under the GPU timing lock.
