# Blind listening and reading tests

Offline human checks of the P7 campaign. Every test is a folder (`test/`) you zip and send: the
listener or reader opens `index.html` in any browser (no install, no network, works from file://),
answers, and clicks **Export my answers**, which downloads a small JSON file they send back.
Nothing is uploaded anywhere; answers live in the browser (localStorage) until exported. The key
that says which system is which (`key.json`) never leaves the organizer.

| Test | Build | Page | Score |
|---|---|---|---|
| Voice changer A/B (plan T6) | `make_test.py` | generated inside the folder | `score.py` |
| Translation A/B (P7 MT check) | `make_mt_test.py` | `mt-ab.html` | `aggregate.py` |
| Voice / TTS listening test (P7: Qwen3-TTS vs Kokoro vs Magpie) | `make_tts_test.py` | `tts-listen.html` | `aggregate.py` |

The builders and `aggregate.py` need the P7 Python environment (`scripts/speech-bench/setup-venv.ps1`:
numpy, soundfile, scipy). File formats: `schemas/` (`sigil-listening-mt/1`, `sigil-listening-tts/1`,
`sigil-listening-key/1`, `sigil-listening-results/1`); `scripts/speech-bench/validate.py` checks them.

## Translation A/B

For friends who read both languages. Each trial shows a source line (and, with `--context`, the
line before it in the same dialogue or document) and two translations in a random order; the
reader picks A, B, "about the same" or "both bad" (keys 1-4).

```powershell
$py = "$env:LOCALAPPDATA\sigil-dev-cache\p7\venv\Scripts\python.exe"
# pairs.<direction>.jsonl come from: score_mt.py --report <mt report> --set <MT set> --out <dir> --export-pairs <pairs dir>
& $py tools/listening-test/make_mt_test.py --pairs pairs\pairs.ja-en.jsonl --pairs pairs\pairs.en-ja.jsonl `
    --systems hy-mt2-1.8b,milmmt-4b --per-direction 30 --context "$env:LOCALAPPDATA\sigil-dev-cache\p7\corpus\mt\bsd.jsonl" --out mt-ab-1
```

- With three or more `--systems`, every pair gets `--per-direction` trials.
- Lines where a system failed or where both translations are identical carry no preference and are
  skipped (counted in `key.json` `selection`).
- `--show-reference` also shows the human reference (off by default: readers judge the source).
- The A / B sides and the trial order come from `--seed` (random when omitted, kept in `key.json`
  only); the page shuffles the order again for every reader; the test id is random.

## Voice / TTS listening test

From a speech-bench `tts_listening` run (renderings under its `audio/`):

```powershell
& $py tools/listening-test/make_tts_test.py --report bench-results\speech\tts\report.json `
    --prompts "$env:LOCALAPPDATA\sigil-dev-cache\p7\corpus\tts\prompts.jsonl" `
    --engines kokoro,qwen3-tts-0.6b,qwen3-tts-1.7b,magpie --per-language 8 --out tts-1
```

- `--mode mushra` (default, MUSHRA-lite): each trial plays one sentence by every chosen engine that
  rendered it (two or more), in a random order, and the listener rates each 0-100 (Bad to
  Excellent). A slider unlocks only after its sample was played (80 %). There is no hidden
  reference or low anchor: synthetic speech has no ground-truth recording, so scores are relative
  within the test. `--mode pairwise`: A / B / no preference for every pair of engines.
- Prompts: per language, those every chosen engine that speaks the language rendered without error,
  `--kinds` first (`casual,read,numbers,long`), `--per-language` of them.
- Audio: every rendering is converted to one format (mono, 16-bit, the highest rate among them or
  `--rate`) and to the same active level (`--loudness`, -23 dBFS by default; a louder voice
  otherwise wins), peak-limited to -1 dBFS by lowering only that rendering when needed (recorded in
  `key.json`), and written under random names.

## Scoring MT and TTS answers

```powershell
& $py tools/listening-test/aggregate.py mt-ab-1\key.json answers\*.json --json mt-ab-1\results.json --md mt-ab-1\results.md
& $py tools/listening-test/aggregate.py tts-1\key.json answers\*.json --baseline kokoro --md tts-1\results.md
```

- Answers of another test are refused; a reader who exported twice counts once (the latest);
  readers who completed fewer than `--min-complete` (0.5) of the trials are listed and left out.
- Pairwise: per system pair, overall and per direction / language, the wins of each side, "about the
  same" and "both bad", the preference share among decided answers with a Wilson 95 % interval,
  and a Bradley-Terry strength per system when three or more are compared.
- MUSHRA-lite: per system the mean rating with a 95 % bootstrap interval over trials (each trial's
  ratings by every listener move together), the mean of each listener's z-scored ratings as a check
  on listener bias, per language, and against `--baseline` the mean paired difference (same
  listener, same sentence) with its interval.

The page smoke test (`tests/pages_smoke.mjs`, run by `scripts/speech-bench/tests/test_listening.py`)
drives both pages headless with the repo's Playwright from file:// up to the exported answers.

## Voice changer A/B (plan T6)

Plan T6 asks for a blind preference of at least 60 % for Sigil's voice engine over each competitor
(5 or more listeners, 20 or more pairs each), on the same voice and input. These two scripts make
and score such a test; nothing is uploaded anywhere.

1. Render the same clips with each system: Sigil (a benchmark `quality` or `compare` run with
   `"output": {"write_audio": true}` writes `<run>.output.wav`) and the competitors
   (`scripts/vc-bench`). Level-match them (the page does not normalize).
2. Write a manifest of trials (two renditions of one clip each) and build the test:

   ```powershell
   python tools/listening-test/make_test.py manifest.json --out ab-1
   ```

   `ab-1/test/` is the page and the blinded audio: zip it for the listeners. Keep `ab-1/key.json`
   yourself; it maps every trial's A and B to the systems and keeps the shuffle seed. Every
   rendition is converted to one format (16-bit PCM WAV, mono, at the highest rate among them:
   `--rate` / `--channels` change that), so a competitor's FLAC or MP3 looks like Sigil's WAV. It
   needs `numpy`, plus `scipy` to resample (or `ffmpeg` on the PATH) and `soundfile` (or `ffmpeg`)
   for formats other than WAV.
3. Each listener opens `test/index.html` (offline, any browser), answers every trial (A, B or no
   preference) and clicks "Download my answers".
4. Score:

   ```powershell
   python tools/listening-test/score.py ab-1/key.json answers/*.json --system sigil
   ```

   It prints each system's preference share with a Wilson 95 % interval and whether T6 holds.

The page stores answers in the browser only (localStorage) until the listener downloads them. The
A/B side of each system is drawn per trial from the shuffle seed (`--seed`, else drawn at random),
which only `key.json` holds; the test id on the page and the audio file names come from the OS's
secure random source, so nothing a listener can see leads back to the seed.
