# Competitor comparison (`scripts/vc-bench`, plan T6)

Plan T6 asks whether Sigil's voice engine beats the other voice changers on the same voice, the
same inputs and the same PC. The targets are statistical (accepted target revisions):

- **DNSMOS**: a mean paired difference of at least +0.05 with the 95 % CI lower bound above 0, at
  matched latency, on loudness-normalised clips of 9.01 s or more.
- **Gross pitch error** and **MCD to a neutral reference**: non-inferiority (the CI upper bound of
  Sigil − competitor within a margin fixed in advance), with pitch scored against EGG or consensus
  truth and holding under every analyzer.
- **Latency ≤ 0.7×**, judged as Pareto: for each competitor operating point (its minimum stable
  chunk included), some Sigil configuration at 0.7× its latency that is non-inferior on every
  quality row.
- **Normalized RTF**: decided only from same-GPU runs.
- **Blind A/B ≥ 60 %** (`tools/listening-test`): it carries the quality claim when the corpus is
  smaller than the DNSMOS row needs.

Sigil's `compare` mode (`src-tauri/src/services/bench/compare.rs`) decides all of it the moment the
WAVs exist. It scores every system the same way, refuses unfair pairings (another transpose,
another voice, another input, mismatched latency kinds) and prints per-clip paired statistics with
the verdicts. Nothing here installs a competitor. Each driver runs in the environment you already
have for that app; Sigil only reads the WAVs and JSON sidecars they write.

## Files

| File | Runs in | What it does |
|---|---|---|
| `vcbench_common.py` | any Python | Shared helpers: hashes, sidecars (`sigil-vc-bench-sidecar/1`), sweeps, BS.1770 loudness, corpus reading. |
| `rvc_webui_offline.py` | your RVC WebUI env | The block loop of `realtime_gui.py` over a clip or a corpus; `--sweep 40,60,100,250`; a sidecar per WAV. |
| `applio_offline.py` | your Applio env | `rvc.realtime.core.VoiceChanger.on_request` block by block; `--sweep`; sidecars. |
| `play_record.py` | any Python with sounddevice | A live recording through two cables, for PCs without Sigil. It logs status flags and measures no latency; Sigil labels the offset it reads from the recording `uncalibrated` and never sets it against a calibrated latency. |
| `rvc_offline_reference.py` | your RVC WebUI env | Neutral references: upstream `vc_single`, fp32, RMVPE, whole utterance, 2+ noise seeds. |
| `egg_f0.py` | numpy + soundfile | F0 truth from an EGG channel (dEGG glottal closure instants), with ARCTIC concatenation. |
| `fcpe_check.py` | env with torchfcpe | Sigil's FCPE adapter against upstream torchfcpe on the same clips. Report only. |
| `vcrs_probe.py` | any Python | Same-rig compute rows: Sigil reports, vc-rs `latency_probe` CSVs, driver sidecars; RTF verdicts only on one GPU. |
| `compare_export.py` | any Python | From a compare report: `<run>.make_test.json`, `<run>.pareto.svg`, `<run>.t6.md`. |
| `tests/test_vcbench.py` | numpy + soundfile | Self-checks, no network: `python -m pytest scripts/vc-bench/tests -q`. |

## 1. Build the corpus (manual: about 650 MB of downloads, one at a time)

```
python scripts/bench/fetch-public-corpus.py --dry-run        # the plan, no network
python scripts/bench/fetch-public-corpus.py --voice <path>/default.pth
```

This writes about 92 clips (LibriSpeech test-clean: 40 speakers × 2; CMU ARCTIC bdl, jmk and slt:
4 each with EGG truth) to `%LOCALAPPDATA%\sigil-dev-cache\corpus\public-v1\` with `corpus.json`.
Every clip is 10-20 s at -26 LUFS. Never commit the audio. If you already have the archives,
point at them with `--archives <dir>`. At the measured paired SD (about 0.21), the +0.05 DNSMOS row
needs about 68 clips (50 % power) or about 139 (80 %).

For an interim local corpus from the dev cache's speech clips (no download):
`python scripts/bench/make-test-clips.py --local-corpus --out <dir>`.

## 2. Neutral references (manual, in your RVC WebUI environment)

```
python scripts/vc-bench/rvc_offline_reference.py --rvc-root C:/RVC --pth C:/RVC/assets/weights/default.pth \
    --corpus <corpus>/corpus.json --out-dir <corpus>/refs --seeds 1,2 --index-rate 0 --protect 0.33
```

This writes `<corpus>/corpus.with-refs.json`; use it as the corpus from here on. Each clip's two
seeds give the stochastic floor (`reference_floor_db`).

## 3. Each competitor's renditions

Use the same voice everywhere. The MIT default voice `default.pth` from
PhoenixStormJr/RVC-V2-default-voice is the one Sigil auto-imports; its sha256 is in Sigil's slot
`manifest.json` as `source_sha256`. Use the same transpose (0), index rate and protect.

**Offline tier** (a driver's block loop over each clip; one operating point per block size):

```
python scripts/vc-bench/rvc_webui_offline.py --rvc-root C:/RVC --pth C:/RVC/assets/weights/default.pth \
    --corpus <corpus>/corpus.with-refs.json --out-dir <corpus>/out/rvc_webui --sweep 40,60,100,250 --f0method rmvpe
python scripts/vc-bench/applio_offline.py --applio-root C:/Applio --pth C:/Applio/logs/default/default.pth \
    --corpus <corpus>/corpus.with-refs.json --out-dir <corpus>/out/applio --sweep 40,60,100,250 --f0-method rmvpe
```

Each prints the `points` entries to paste into the manifest. A point whose per-block p99 reaches its
block period is marked infeasible and does not count. The smallest feasible block is that app's
minimum stable chunk on this GPU.

Both drivers build a fresh RVC / `VoiceChanger` object for every clip (RVC WebUI reuses the loaded
models through `last_rvc`; Applio reloads them, which takes a few seconds per clip), so a rendition
never depends on the clip rendered before it, and feed silent blocks after each clip so the last
speech leaves the loop (the flush blocks are not timed). Use the same `--index-rate`, `--protect` and
rms-mix settings for every system and for Sigil's rows (the drivers default to Sigil's shipping
settings: index 0, protect 0.33, volume envelope 1 = off; RVC WebUI's GUI starts at an envelope of 0
and Applio's tab at index 0.75 and protect 0.5). Sigil flags a competitor whose index rate, protect,
loudness mix or F0 method differs from the Sigil rows' (`operating point: ...`, repeated under
`t6.caveats`). Keep one sweep's renditions in one folder: if the
sidecars of a point's clips disagree on the voice, the transpose or the block (a clip re-run with
another `--pitch`, a stale file), the whole point is refused.

**Live tier** (every system through the same devices, Sigil's live voice changer included). Route
the meter's output into the app (for example "CABLE Input" as the app's microphone source) and the
app's output into a second cable. Add a `captures` entry for the app's live point
(`corpus.example.json`), start the app, and run the compare spec once. The meter plays every clip
and records the response with glitch and dropout flags. Measure each app's end-to-end latency with
an `e2e_loopback` run through it (`detector: onset`, `calibration_ms` = a direct cable's median)
and put that in the point's `latency_ms` with `latency_kind: e2e_loopback`. Find the minimum stable
chunk by stepping the app's chunk down until the capture shows glitches or dropouts. On PCs without
Sigil, `play_record.py --corpus ... --out-dir ...` records the same way, with status flags only; its
latency stays `uncalibrated` unless you state a calibrated `e2e_loopback` one in the point.

A capture cannot know which voice the app loaded, so its sidecar never claims the corpus's voice:
put the voice you loaded in the point's `voice_sha256` (otherwise the point is flagged "voice not
verified").

## 4. Score

```json
{
  "name": "h2h",
  "engine": { "kind": "rvc", "slot_dir": "C:/Users/me/AppData/Roaming/com.sigil.app/voice-changer/models/default",
              "execution_provider": "cuda", "pitch_extractor": "rmvpe", "preset": "low_latency", "block_ms": 40 },
  "runs": [
    { "mode": "compare", "name": "h2h", "corpus": "C:/corpus/public-v1/corpus.with-refs.json",
      "sigil": { "configs": ["shipping_default", "autotuned", "low_latency", "quality"] } }
  ],
  "output": { "write_audio": true }
}
```

`Sigil.exe --headless bench h2h.json --out <dir>` scores every point and clip, and pairs every Sigil
configuration with every competitor point in its tier. `details.v2` holds the verdicts, the T6
decision (with its `caveats`), the Pareto data, the corpus sizing and the listening manifest.

A row that misses clips decides nothing: if a point has no complete score on some clip (a missing,
refused or muted rendition, an alignment failure, a conversion error, no DNSMOS on a clip of 9.01 s
or more) it is listed under `unscored_clips`, counted in `points_incomplete` / `clips_unscored`, and
its verdicts read undecided. Sigil's own rows are held to the same rule. The stand-in references of a
dry run (`stand_in_references`, Sigil's own render) are scored apart (`mcd_stand_in_db`) and never
feed the T6 MCD row, which needs the upstream references of step 2. The blind A/B manifest leaves out
refused and real-time-infeasible points (`make_test.skipped`, printed in `t6.md`). Then:

```
python scripts/vc-bench/compare_export.py <dir>/report.json      # h2h.make_test.json, h2h.pareto.svg, h2h.t6.md
python tools/listening-test/make_test.py <dir>/h2h.make_test.json --out listening-1
```

## 5. Same-rig compute (normalized RTF)

```
python scripts/vc-bench/vcrs_probe.py --sigil <sigil offline report.json> --vcrs <vc-rs latency_probe.csv> \
    --vcrs-gpu "NVIDIA GeForce RTX 4090" --sidecars <corpus>/out/rvc_webui --json rows.json
```

Verdicts are given only between rows on the same GPU, at the same block and, where both name one,
the same F0 method; several Sigil rows at that point are judged by their upper median, never by
the best of them. The published numbers are listed as references
and never judged: vc-rs 13.9 ms p95 per 100 ms chunk is trtexec model-only time (TensorRT 11, RTX
3060 Ti), and rvc-realtime is p50 15.7 / p95 21.2 ms per 60 ms block (FCPE, fp16, RTX 3060 Ti). Sigil's
component rows at vc-rs's shape (100 ms chunk + 100 ms extra) are labelled "model step incl. host
I/O".

## 6. FCPE adapter check (manual, report only)

Run a corpus with `"dump_f0": true` and `"analyzers": ["fcpe", "rmvpe", "swiftf0", "crepe", "dio"]`,
then run `python scripts/vc-bench/fcpe_check.py <dir>/report.json` in an environment with
`torchfcpe`. Any fix to `fcpe.rs` goes to the integrator.

## Status

- No competitor is installed on the dev rig. The RVC WebUI and Applio drivers were written against
  RVC WebUI `81eed5e8` (2026-08) and Applio `c7665ac9`, and they have not run here. On the first real
  run, check the output length against the input, and check the printed notes.
- The neutral references, the public corpus download, the live captures (the dev PC's virtual-cable
  endpoints are disabled) and every competitor sweep are manual steps.
- The T6 dry run (`scripts/bench/specs/p3b/c/run_dryrun.ps1`) compares SVE with its own ablation on
  local clips. Its margins are never competitor results. Results from 2026-10-03, on 10 local clips
  of 11-28 s at -26 LUFS, MIT default voice, with stand-in references:
  - SVE minus the ablation (anchored noise and phase0 off) at 40 ms blocks shows no difference
    larger than its CI on any automatic metric. DNSMOS: DirectML -0.011 [-0.120, +0.098], CPU
    -0.058 [-0.157, +0.041]. Envelope MCD to the stand-in references (reported as `mcd_stand_in_db`
    since the review fixes; it was `mcd_ref_db` then): +0.007 / +0.007 dB. GPE: 0 under
    RMVPE and SwiftF0. Artifacts and seam clicks are within their CIs.
  - At the GUI-like shape (200 ms, 2 s extra), DNSMOS reads -0.057 [-0.134, +0.020] on DirectML
    and +0.002 on CPU.
  - The shape itself separates clearly: mean DNSMOS is 3.11-3.14 at the GUI-like shape against
    2.71-2.79 at 40 ms, and MCD is 1.98-1.99 against 2.40-2.43 dB. The metrics are therefore
    sensitive, and time anchoring buys nothing they can see at n = 10.
  - Per the plan, a short blind listening check of the 40 ms pairs comes before any conclusion about
    metric sensitivity. The `make_test` manifest is written with `-WriteAudio`.
  - The paired DNSMOS SD is 0.14-0.15, so the +0.05 row needs 31-36 clips (50 % power) or 63-73
    (80 %). The public corpus (about 92 clips) can carry it.
  - The 4-analyzer consensus covers only 2-44 % of the speech frames. SwiftF0 voices 2-47 % of the
    frames RMVPE voices 61-88 % of, and CREPE-tiny voices 78-100 %. Prefer EGG truth (ARCTIC) for
    the GPE row, or set `consensus_min_agree: 3`.
  - Onsets after 300 ms of silence come out 4.8-5.0 dB under the input's own onset shape at 40 ms,
    and 7.1-7.5 dB under it with 2 s of context (the long-context onset attenuation).
