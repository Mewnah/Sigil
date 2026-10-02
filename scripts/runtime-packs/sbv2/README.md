# Style-Bert-VITS2 / AivisSpeech voices (`sbv2` engine): no runtime pack

The plan (P6-G) ran `sbv2_core`'s REST server as a Sigil-built helper pack, because `sbv2_core`
links its own ONNX Runtime (ort rc.9) and a second static ORT clashes with Sigil's in one
executable. P6-G ported the engine instead: the Japanese frontend (jpreprocess, the pure-Rust
OpenJTalk), the DeBERTa BERT and the VITS voice graph run on **Sigil's own ONNX Runtime** in
`src-tauri/src/services/tts_helpers/sbv2/` (MIT code from `sbv2_core`, checked against
Style-Bert-VITS2's Python). So there is nothing to build here and no pack to publish:

| What | Where it comes from |
|---|---|
| Engine | Sigil itself (`tts_helpers::sbv2`, worker thread `sigil-sbv2-tts`) |
| `tts/sbv2-bert-ja` | published: `tsukumijima/deberta-v2-large-japanese-char-wwm-onnx` (`model_fp16.onnx` 653 MB, `tokenizer.json`), pinned in `src-tauri/model-catalog/runtime-sbv2.json` |
| `text/jpreprocess-naist-jdic` | published: jpreprocess v0.15.0 `naist-jdic-jpreprocess.tar.gz` (29 MB) |
| Voices | the user's `.aivmx` files (AivisHub) or Style-Bert-VITS2 ONNX exports, imported in the TTS panel |

The jpreprocess dictionary must come from the same jpreprocess release as the crate in
`src-tauri/Cargo.toml`: when the crate is updated, update the catalog entry's URL, version and hash
together (`node scripts/models/pin-catalog.mjs --fragment src-tauri/model-catalog/runtime-sbv2.json --write`).

## Testing with real models

Unit tests run on stand-in graphs. The tests that need real files are ignored unless
`SIGIL_TEST_MODELS_DIR` points at a folder laid out like this:

```
<SIGIL_TEST_MODELS_DIR>/
  naist-jdic/              the extracted naist-jdic-jpreprocess.tar.gz (dict.da, matrix.mtx, ...)
  sbv2-bert-ja/            model_fp16.onnx + tokenizer.json (the tts/sbv2-bert-ja entry's files)
  sbv2-voice.aivmx         any AivisSpeech voice (for example the default voice from AivisHub)
```

Fetch the two catalog entries with the model store (one file at a time):

```
Sigil.exe --headless fetch text/jpreprocess-naist-jdic tts/sbv2-bert-ja
```

then copy (or point at) their install folders, and run:

```
$env:SIGIL_TEST_MODELS_DIR = "D:\sigil-test-models"
cargo test --manifest-path src-tauri/Cargo.toml tts_helpers::sbv2 -- --ignored
```

`real_model_sbv2_speaks` prints the load time and the real-time factor of one sentence on the CPU.
The other ignored tests need less: the dictionary alone (`real_dictionary_*`, and
`the_engine_speaks_an_imported_voice_through_the_worker`, which runs the whole engine on stand-in
graphs), plus `sbv2-bert-ja/tokenizer.json` (0.4 MB) for `real_tokenizer_knows_what_the_frontend_writes`
(every character the frontend keeps is in the BERT's vocabulary). Those run without the 650 MB BERT:

```
cargo test --manifest-path src-tauri/Cargo.toml --bin sigil -- --ignored tts_helpers::sbv2::g2p tts_helpers::sbv2::tests::the_engine tts_helpers::sbv2::tests::real_tokenizer
```

## Manual check (acceptance: "an AIVMX voice speaks")

1. `pnpm tauri dev`, Text to speech, backend **Style-Bert-VITS2 / AivisSpeech voices**.
2. Download the two files from the card at the top of the panel.
3. Drop an `.aivmx` on the panel (or **Add voice…**); it is copied into
   `%APPDATA%\com.sigil.app\tts\sbv2\<id>\`. An `.aivm` (PyTorch) file is refused with how to get
   the `.aivmx` version.
4. Pick a style, press **Preview**, then start TTS and type Japanese text: it speaks through the
   chosen output device, and through the voice changer when the route says so.
5. Stop: the engine frees its memory (about 0.7 GB for the BERT plus the voice).
