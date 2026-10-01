# Blind A/B listening test

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
