# Competitor comparison drivers (`scripts/vc-bench`)

Plan T6 compares Sigil's voice engine with the other voice changers on the same voice, the same
input and the same PC: DNSMOS P.835 at least 0.05 above each, gross pitch error no higher than any,
a blind A/B preference of at least 60 %, and end-to-end latency at most 0.7x the best competitor's
at matched quality. Sigil scores every candidate the same way in the benchmark's `compare` mode
(`src-tauri/src/services/bench/compare.rs`, `docs/benchmarks/README.md`).

Nothing here installs a competitor. The drivers run in the environment you already have for that
app; Sigil only reads the WAV files they write.

## 1. Get each candidate's output for the same input

Use one input clip (a consented recording of real speech, 10-30 s; `scripts/bench/corpus.example.json`
describes the local corpus) and the same voice in every app (the `.pth` Sigil imported, or the
ONNX export an app needs).

| App | How | Notes |
|---|---|---|
| RVC WebUI realtime | `python scripts/vc-bench/rvc_webui_offline.py --rvc-root <checkout> --pth <voice.pth> --input clip.wav --out rvc-webui.wav` | The block loop of `realtime_gui.py` (input rings, 16 kHz resampler, `rtrvc.RVC.infer`, output resampler, volume envelope, SOLA), offline. Set `--block-time`, `--crossfade-time`, `--extra-time`, `--f0method` to what you use live. |
| Applio realtime | `python scripts/vc-bench/applio_offline.py --applio-root <checkout> --pth <voice.pth> --input clip.wav --out applio.wav` | Feeds `rvc.realtime.core.VoiceChanger.on_request` block by block. |
| w-okada VCClient, the Deiteris fork, any other app | `python scripts/vc-bench/play_record.py --clip clip.wav --play "<cable into the app>" --record "<cable out of the app>" --out wokada.wav` | A live recording through virtual cables (the app's own settings apply). The file includes the app's delay; Sigil aligns it. |
| Sigil | the compare run converts the input itself (`sigil.include`, default on) | Same voice, same clip, the spec's engine settings. |

The offline drivers print the app's per-block compute p50 / p95 / p99, comparable with Sigil's
`block_ms_*` (on the same PC and GPU).

## 2. Measure each app's end-to-end latency

Sigil's e2e-loopback meter times any app through virtual cables: an `e2e_loopback` run with
`output_device` = the cable into the app and `input_device` = the cable out of it, `detector:
onset` (a voice changer changes the timbre), `calibration_ms` = the median of a direct cable run
with the same devices. Put the result into the candidate's `latency_ms`. Sigil's own latency comes
from an `e2e_loopback` run through Sigil in the same spec (`sigil.latency_from`).

## 3. Score them

```json
{
  "name": "h2h-default-voice",
  "engine": { "kind": "rvc", "slot_dir": "C:/Users/me/AppData/Roaming/com.sigil.app/voice-changer/models/default",
              "execution_provider": "cuda", "pitch_extractor": "rmvpe", "preset": "low_latency", "block_ms": 40 },
  "input": { "kind": "wav", "path": "C:/clips/speech-01.wav" },
  "runs": [
    { "mode": "e2e_loopback", "name": "sigil-e2e", "output_device": "CABLE Input", "input_device": "CABLE-B Output",
      "detector": "onset", "calibration_ms": 11.8 },
    { "mode": "compare", "name": "h2h",
      "sigil": { "latency_from": "sigil-e2e", "latency_metric": "e2e_ms_p50" },
      "candidates": [
        { "name": "rvc-webui", "wav": "C:/clips/out/rvc-webui.wav", "latency_ms": 310 },
        { "name": "applio", "wav": "C:/clips/out/applio.wav", "latency_ms": 280 },
        { "name": "wokada", "wav": "C:/clips/out/wokada.wav", "latency_ms": 190 }
      ],
      "scoring": { "speaker_reference": "C:/clips/target-voice-sample.wav" },
      "thresholds": { "dnsmos_ovrl_margin": { "min": 0.05 }, "gross_pitch_error_margin": { "min": 0 },
                      "latency_ratio": { "max": 0.7 } } }
  ]
}
```

`Sigil.exe --headless bench h2h.json` writes the per-candidate table (DNSMOS, pitch errors against
the input, speaker similarity to the target clip, artifact events per minute, latency) and the T6
margins. The blind listening test (T6 preference) uses the same WAVs: `tools/listening-test`.

## Status

The drivers were written against RVC WebUI `81eed5e8` (2026-08) and Applio `c7665ac9`; they were
not run on the dev rig (neither app is installed there). The first run in a real environment should
check the output length against the input and the printed notes.
