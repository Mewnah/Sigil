# Speech benchmarks

Sigil's own speech measurements. Defaults and routing come from these numbers (speech-stack plan,
Principles and §8), not from vendor figures. The first half is the **P7-A campaign** (2026-10-09/10):
light speech-to-text, translation and text-to-speech engines measured for the streamer who runs
captions into OBS while playing VRChat, plus Sigil's own footprint. The second half keeps the earlier
probe numbers (P2 speech engines, P4 translation) that the campaign did not redo. Reproduce everything
with [`scripts/speech-bench/campaign/`](../../scripts/speech-bench/campaign/README.md).

**What was and was not measured.** User scope (2026-10-09): the main audience streams STT captions
into OBS while in VR, so the question for every engine is "good captions without taking the game's CPU
or GPU". Light and mid-weight engines were screened first and only the finalists got the full test.
**Heavy GPU models are opt-in and were not benchmarked or tuned**: Cohere Transcribe, Qwen3-ASR 1.7B,
Voxtral, Confucius4-R2T2, Hy-MT2 7B, MiLMMT, TranslateGemma, Qwen3-TTS 1.7B, Magpie, Breeze, OmniVoice,
Fish S2. Nothing here ran in VR or inside VRChat; the game is stood in for by the benchmark's load
simulator. There are no real VRChat clips yet: the corpus is synthetic (FLEURS read speech with
simulated room, codec and bleed), and the real-clip drop-in is the authority once it exists.

## How it was measured

- **Rig.** Ryzen 7 7800X3D (8 cores, 16 threads), RTX 4090 24 GB, Windows 11 (build 26200), release
  build (LTO) of `Sigil.exe` with the shipped ggml flags, process at below-normal priority, the PC otherwise
  quiet (the campaign's runs took the dev rig's GPU timing lock, so no other measurement overlapped them). The dev CPU is faster than a typical 6-core
  gaming PC, so absolute times are optimistic; ratios are meaningful.
- **A mid-range CPU.** CPU rows ran on a 12-logical-processor slice of the 16 (six cores with SMT, "6C")
  and, for one question, on 8 logical processors ("4C"), selected by processor affinity. The slice does
  not change what the engines plan (thread counts, llama-server's threads come from the real 8 cores).
- **The game.** `h3_cpu_game` of the benchmark: four busy-loop threads at 60 % duty (about 2.4 cores, VRChat's
  main, render and OBS threads on a 6-core PC). "Idle" rows run without it. The GPU hammer (`h1_vrchat_obs`: a
  convolution load at 70 % duty on a 16.7 ms frame, plus 3 CPU threads at 50 %) was used for the one light
  GPU row (the Vulkan translator). For the whole app the game is four Node processes spinning 60 % of every
  100 ms beside it.
- **Corpus.** FLEURS test speech (en, de, es, fr, pt, ru, ja, ko, zh; read speech, CC-BY-4.0), 9
  languages x 14 conditions, `content_sha256` `459c2ff05c92...`, scored with `sigil-norm/1` (WER; CER for
  ja / ko / zh; Whisper's English normalizer). Digits against number words outside English count as errors
  for every engine alike. Conditions used:

  | Group | Conditions | What it stands for |
  |---|---|---|
  | **Own microphone** | `clean`; `room_light` (synthetic reverb tail, RT60 0.35 s); `mic_codec` (200-3800 Hz band, 8-bit mu-law, hiss: a bad headset or a Bluetooth link) | The streamer's own captions. Headphones keep the game out of this microphone. **These alone choose the defaults.** |
  | **Bleed** | `music_snr10` (FMA tracks 10 dB under the speech), `babble_snr10` (three other talkers 10 dB under), `vrchat_mix` (speech in a light room with music, effects and two talkers, about 12 dB combined, mic band 100 Hz-7 kHz) | Speakers instead of headphones, and the "other players' captions" feature, whose input is VRChat's own audio. Tabulated apart; they play no part in the own-microphone choice (user decision 2026-10-09). |

  **Screening** = `quick` (clean, vrchat_mix), 10 utterances per language; **finals** = 30 utterances per
  language and condition on `core` (clean, music, babble, vrchat_mix) plus the two mic-side conditions
  (Nemotron outside English: 12 utterances, on `core` only; Korean: 100 utterances on the own-mic conditions). Intervals are paired percentile bootstraps over
  utterances (1,000 resamples, seed 20261002); the own-microphone intervals resample whole utterances (all
  conditions of an utterance together).
- **Limits to state next to any number.** Read speech, not VRChat chatter; the bleed is mixed digitally, not
  played through a speaker; a babble talker may be the target's own speaker; one machine; 30 utterances per
  cell, so intervals of a few points.
- **Spec identities** (`spec_sha256`, first 12 hex): screening `d8d85d450786`, finals `82aab70317b1` /
  `486c0fe2f8d2` / `06ff7a6a5151` / `5e3bef68870c`, mic-side `37fc946c3a8c` / `f30bdd706e97` / `7d6d71607a6e`, auto mode `5b74b52815b3`, decode speed
  `855225650a1d`, streaming `077ffa8b00d8`, pipeline 6C `6fcca72e66f2`, pipeline 4C `2a3b31a36d34`, translation
  `ea478c95198d`, Kokoro `26ebc77e6b80`.

## How the defaults are chosen

A streamer's own captions come from their own microphone, and with headphones the game's audio does not
reach it. So **the own-microphone STT defaults are chosen only on the clean results plus the mic-side
conditions (`room_light`, `mic_codec`)**; the bleed conditions (`music`, `babble`, `vrchat_mix`) play no part in
that choice, not even as a tie-breaker. They are tabulated apart because the "other players' captions" (loopback)
feature hears VRChat's own audio, and a streamer on speakers hears the game in the microphone.

The rule is code ([`campaign/own_mic.py`](../../scripts/speech-bench/campaign/own_mic.py), unit-tested):

1. Per language, pool the error counts of every own-mic condition and rank the engines on that rate. The
   interval and the paired differences resample whole utterances.
2. An engine **ties** the best one when the paired interval of the difference contains zero and its point
   estimate is at most 1.5 points (WER or CER) above the best.
3. Among tied engines the **cheapest** (CPU and RAM per audio second) takes the language. An engine that is
   significantly worse, or more than the margin behind, is out whatever it costs.
4. Granite TurboCTC (it writes lower case without punctuation) and Nemotron (a streaming partials engine) are
   **never a caption final**, whatever they score or cost: `--exclude-finals` (default `granite-cpu,nemotron-560`)
   keeps them in the tables and in the comparison, but removes them from the tie set, so Granite's 6.1 in English
   can be the reference the others are measured against and still never be the default. Nemotron outside English has
   12 utterances per language and only the `core` conditions, so there it is listed but not ranked.
5. **Incomplete data gives no default.** If the files for a language lack an own-mic condition for an engine that
   could be a final (for example Qwen3-ASR 0.6B without its `mic-b` / `mic-c` runs), the language gets no winner,
   the tool says so on stderr and exits 2 (`--allow-incomplete` keeps a flagged provisional winner). With the
   `finals-all` and `mic-a` files alone an earlier version printed SenseVoice as the Korean winner, the reverse of
   the result with all four files.

## The defaults this campaign sets

CPU tier (the recommendation for streamers; `src-tauri/model-catalog/stt-routing.json`, the shared fixture
`src-tauri/test-fixtures/routing.json` and both router test suites are updated to match):

| Language | Live partials | Final (second pass) | Basis (own mic: clean + room_light + mic_codec, WER / CER %) |
|---|---|---|---|
| English | Nemotron 3.5 streaming 560 ms | Parakeet Ultra int8 | Ultra 6.2 [4.4, 8.3], within 0.1 points of the best engine (Granite 6.1, which is never a final: no casing or punctuation); Qwen3-ASR 0.6B (6.9) ties; Parakeet v3 (7.4, +1.3 [+0.2, +2.5] behind the best) and SenseVoice (8.7, +2.5 [+0.4, +4.7]) are behind; Ultra needs 2.3x less CPU and 2.6x less RAM than Qwen3-ASR |
| de, es, fr, pt, it, nl and the other 17 Parakeet languages | Nemotron | Parakeet Ultra int8 | Ultra is the best engine in every language: de 10.1, es 7.2, fr 8.5, pt 9.1 against v3 11.2 / 8.9 / 11.9 / 10.7 and Qwen3-ASR 12.4 / 10.2 / 15.8 / 14.2 |
| **Russian** (was Qwen3-ASR 0.6B) | Nemotron | **Parakeet Ultra int8** | Ultra 7.3 [4.1, 11.2] against 22.1 [17.1, 27.2] for Qwen3-ASR 0.6B (+14.8 [+10.4, +19.1] points behind) |
| **Japanese** (was Qwen3-ASR 0.6B) | Nemotron | **SenseVoice small int8** | SenseVoice 6.8 [4.7, 9.2] against 10.0 [7.9, 12.3] for Qwen3-ASR 0.6B (+3.2 [+1.5, +4.9] points behind); 5.5x less CPU and 5.6x less RAM |
| Korean (unchanged) | Nemotron | Qwen3-ASR 0.6B int8 | Qwen3-ASR 0.6B 5.1 [3.8, 6.5] against 7.9 [5.9, 9.9] for SenseVoice on 100 utterances per condition (SenseVoice is +2.8 [+1.4, +4.4] points behind); SenseVoice is 5.5x cheaper but behind on the own microphone (under babble it wins; that does not count here) |
| Chinese, Cantonese (unchanged) | none | SenseVoice small int8 | SenseVoice 4.5 [2.9, 6.3] against 3.8 for Qwen3-ASR 0.6B: a tie (SenseVoice +0.7 [-1.0, +2.3] behind), so the 5.5x cheaper engine keeps the language |
| vi, id, ms, th, fil, fa, mk | Nemotron (vi) | Qwen3-ASR 0.6B int8 | the only CPU engine that covers them; not measured (no corpus language), unchanged |
| ar, hi, tr, nb | Nemotron alone | - | unchanged (transcribe.cpp's FLEURS table: Nemotron's own finals beat Qwen3-ASR 0.6B); not re-measured |
| anything else | - | Whisper small (6+ cores) / base | unchanged; Whisper was not re-measured (see below) |
| `auto` (language not set) | Nemotron for the detected languages it streams | Qwen3-ASR 0.6B + language ID | unchanged; see "Auto mode" for what not setting a language costs |

What stays: Nemotron 560 ms is the live-partial engine everywhere (the 160 ms variant needs a full core and
falls behind next to a game); two passes stay the default from 6 physical cores (`cpuTwoPassMinCores`, see the
4-core row below); the GPU rows (Cohere, Qwen3-ASR 1.7B) keep their P2 order, opt-in and unbenchmarked.

**Other players' captions (loopback; low priority).** The feature shares this table, so its default is the
same: its audio is VRChat's own, and the bleed numbers below pick the same engine for every language but
Korean (SenseVoice beats Qwen3-ASR 0.6B under babble, 22.8 against 39.0, and ties in the VRChat mix). No separate
row was added; a loopback-only Korean row is listed under "For later".

Translation: the default chain **[bundled Hy-MT2 1.8B, Bergamot, NLLB-200 600M] is confirmed** (Hy-MT2 takes the ja, ko and zh targets by 5-16 chrF points over the better NLLB setting).
One built-in route was added after P7 (MT-Europe, 2026-10-11): while the bundled translator runs on the processor, English with French, Spanish, Portuguese and Dutch (both directions)
goes to NLLB-200 600M with beam 4 first, with the rest of the chain behind it (+2.8 to +6.0 chrF2 with the 95 % interval above zero, 1.6 s a casual caption line against 0.4 s, 1.8 cores
against 3.7-5.0); German and Italian tied and stay on Hy-MT2, and nothing changes on a graphics card; see "European pairs on a processor-only PC". Text-to-speech: Kokoro (local), Windows SAPI and Edge (cloud) are the defaults;
Kokoro and SAPI work end to end and are cheap. Footprint: see the last section. Interim translations are off by default while the bundled translator runs on the processor (P7-I0, "App footprint").

## Speech-to-text accuracy

### Screening (`quick`: clean and VRChat mix, 10 utterances per language)

WER in % (CER for ja, ko, zh); lower is better; the bold figure is the lowest clean rate of the language.
The VRChat-mix column is for reference only (bleed).

| Language | Engine | Clean | VRChat mix (bleed, for reference) | Utterances |
|---|---|---:|---:|---:|
| en (WER) | Parakeet Ultra int8 | 8.1 | 8.6 | 10 |
| en (WER) | Parakeet TDT v3 int8 | 9.1 | 9.6 | 10 |
| en (WER) | Qwen3-ASR 0.6B int8 | 10.0 | 10.0 | 10 |
| en (WER) | SenseVoice small int8 | 11.0 | 15.8 | 10 |
| en (WER) | Nemotron 3.5 streaming 560 ms | 11.0 | 18.2 | 10 |
| en (WER) | Granite TurboCTC Q8_0 (CPU, 4 threads) | **6.7** | 8.1 | 10 |
| en (WER) | Moonshine base-en int8 | 13.4 | 26.3 | 10 |
| en (WER) | Whisper base (ggml fp16) | 25.8 | 26.8 | 10 |
| de (WER) | Parakeet Ultra int8 | **6.7** | 13.9 | 10 |
| de (WER) | Parakeet TDT v3 int8 | 7.9 | 23.6 | 10 |
| de (WER) | Qwen3-ASR 0.6B int8 | 12.1 | 19.4 | 10 |
| de (WER) | Nemotron 3.5 streaming 560 ms | 13.9 | 24.8 | 10 |
| de (WER) | Whisper base (ggml fp16) | 27.3 | 52.1 | 10 |
| es (WER) | Parakeet Ultra int8 | **6.1** | 8.4 | 10 |
| es (WER) | Parakeet TDT v3 int8 | 9.3 | 9.8 | 10 |
| es (WER) | Qwen3-ASR 0.6B int8 | 7.0 | 9.8 | 10 |
| es (WER) | Nemotron 3.5 streaming 560 ms | 12.6 | 18.7 | 10 |
| es (WER) | Whisper base (ggml fp16) | 19.2 | 28.0 | 10 |
| fr (WER) | Parakeet Ultra int8 | **8.8** | 18.8 | 10 |
| fr (WER) | Parakeet TDT v3 int8 | 16.7 | 44.4 | 10 |
| fr (WER) | Qwen3-ASR 0.6B int8 | 14.6 | 28.0 | 10 |
| fr (WER) | Nemotron 3.5 streaming 560 ms | 17.6 | 31.4 | 10 |
| fr (WER) | Whisper base (ggml fp16) | 36.0 | 84.9 | 10 |
| pt (WER) | Parakeet Ultra int8 | **3.9** | 5.9 | 10 |
| pt (WER) | Parakeet TDT v3 int8 | 3.9 | 7.8 | 10 |
| pt (WER) | Qwen3-ASR 0.6B int8 | 6.9 | 11.8 | 10 |
| pt (WER) | Nemotron 3.5 streaming 560 ms | 11.8 | 16.2 | 10 |
| pt (WER) | Whisper base (ggml fp16) | 12.3 | 20.1 | 10 |
| ru (WER) | Parakeet Ultra int8 | **6.1** | 9.9 | 10 |
| ru (WER) | Parakeet TDT v3 int8 | 8.8 | 14.9 | 10 |
| ru (WER) | Qwen3-ASR 0.6B int8 | 25.4 | 30.9 | 10 |
| ru (WER) | Nemotron 3.5 streaming 560 ms | 14.4 | 26.0 | 10 |
| ru (WER) | Whisper base (ggml fp16) | 27.6 | 39.2 | 10 |
| ja (CER) | Qwen3-ASR 0.6B int8 | **10.2** | 34.5 | 10 |
| ja (CER) | SenseVoice small int8 | 10.8 | 19.4 | 10 |
| ja (CER) | Nemotron 3.5 streaming 560 ms | 14.2 | 28.4 | 10 |
| ja (CER) | Whisper base (ggml fp16) | 28.4 | 41.3 | 10 |
| ko (CER) | Qwen3-ASR 0.6B int8 | **5.8** | 18.3 | 10 |
| ko (CER) | SenseVoice small int8 | 7.6 | 9.7 | 10 |
| ko (CER) | Nemotron 3.5 streaming 560 ms | 7.2 | 11.3 | 10 |
| ko (CER) | Whisper base (ggml fp16) | 12.8 | 20.4 | 10 |
| zh (CER) | Qwen3-ASR 0.6B int8 | **1.3** | 6.1 | 10 |
| zh (CER) | SenseVoice small int8 | 2.2 | 14.0 | 10 |
| zh (CER) | Whisper base (ggml fp16) | 28.7 | 46.2 | 10 |

What screening decided: **Parakeet TDT v3** matches Ultra on clean speech but collapses with other talkers
(the finals confirm it: babble WER en 53 %, fr 39 %, de 35 %), and on the finals' clean and mic-side conditions
it is level with or behind Ultra everywhere, so it is nowhere the final (it stays the table's second
candidate). **Moonshine base-en** (the lightest English engine) is 5 points behind Ultra on clean speech (6 of its 10 utterances run over 9.27 s and were decoded in cut pieces, see [Moonshine and long utterances](#moonshine-and-long-utterances-the-9-s-cut)).
**Whisper base** (the fp16 ggml model) is 2 to 20 times worse than the best sherpa engine of every language.
**Granite TurboCTC** (English only; 4 CPU threads or 0.9 GB of VRAM) matches Ultra but writes lower case
without punctuation and is no cheaper, so it is not routed.

### Finals: the own microphone (30 utterances per language and condition)

WER in % (CER for ja, ko, zh). "Own mic, pooled" is the rate over `clean`, `room_light` and `mic_codec` with its
95 % interval; "gap" is the paired difference to the best engine of the language. Korean ran on 100 utterances per
condition (the contested pair), English Nemotron on 30; Nemotron elsewhere has 12 utterances and no mic-side run, so it is
listed, not ranked.

| Language | Engine | clean | room_light | mic_codec | Own mic, pooled (95 % interval) | Gap to the best (95 % interval) |
|---|---|---:|---:|---:|---:|---:|
| de (WER) | Parakeet Ultra int8 | 8.4 | 10.1 | 11.7 | **10.1 [6.4, 14.8]** | best |
| de (WER) | Parakeet TDT v3 int8 | 9.4 | 11.2 | 12.9 | 11.2 [7.3, 15.8] | +1.1 [-0.3, +2.6], tie |
| de (WER) | Qwen3-ASR 0.6B int8 | 11.5 | 12.2 | 13.5 | 12.4 [8.7, 16.6] | +2.3 [-0.6, +5.3], behind by more than the margin |
| de (WER) | Nemotron 3.5 streaming 560 ms | 13.1 | - | - | 13.1 [6.1, 21.8] | not ranked (no room_light, mic_codec) |
| en (WER) | SenseVoice small int8 | 8.3 | 8.9 | 8.8 | 8.7 [5.9, 11.5] | +2.5 [+0.4, +4.7], worse |
| en (WER) | Parakeet Ultra int8 | 6.1 | 6.3 | 6.3 | **6.2 [4.4, 8.3]** | +0.1 [-1.4, +1.5], tie |
| en (WER) | Parakeet TDT v3 int8 | 6.9 | 6.9 | 8.4 | 7.4 [5.2, 9.9] | +1.3 [+0.2, +2.5], worse |
| en (WER) | Granite TurboCTC Q8_0 (CPU, 4 threads) | 6.3 | 6.0 | 6.1 | 6.1 [3.9, 8.7] | best, never a final |
| en (WER) | Qwen3-ASR 0.6B int8 | 6.9 | 7.4 | 6.4 | 6.9 [4.9, 9.3] | +0.8 [-0.6, +2.5], tie |
| en (WER) | Nemotron 3.5 streaming 560 ms | 9.4 | 10.7 | 10.7 | 10.3 [7.1, 13.8] | +4.2 [+1.7, +6.6], worse |
| es (WER) | Parakeet Ultra int8 | 7.0 | 7.5 | 7.2 | **7.2 [2.5, 13.5]** | best |
| es (WER) | Parakeet TDT v3 int8 | 8.3 | 8.5 | 9.9 | 8.9 [3.9, 14.7] | +1.7 [+0.6, +3.0], worse |
| es (WER) | Qwen3-ASR 0.6B int8 | 8.8 | 10.6 | 11.1 | 10.2 [4.9, 16.8] | +2.9 [+1.3, +4.7], worse |
| es (WER) | Nemotron 3.5 streaming 560 ms | 11.8 | - | - | 11.8 [3.5, 23.0] | not ranked (no room_light, mic_codec) |
| fr (WER) | Parakeet Ultra int8 | 8.1 | 8.5 | 8.8 | **8.5 [5.6, 11.8]** | best |
| fr (WER) | Parakeet TDT v3 int8 | 12.6 | 10.6 | 12.5 | 11.9 [8.6, 15.9] | +3.4 [+1.8, +5.3], worse |
| fr (WER) | Qwen3-ASR 0.6B int8 | 13.6 | 16.9 | 17.0 | 15.8 [11.6, 20.5] | +7.4 [+3.7, +10.9], worse |
| fr (WER) | Nemotron 3.5 streaming 560 ms | 15.8 | - | - | 15.8 [8.9, 26.1] | not ranked (no room_light, mic_codec) |
| ja (CER) | SenseVoice small int8 | 6.9 | 7.0 | 6.6 | **6.8 [4.7, 9.2]** | best |
| ja (CER) | Qwen3-ASR 0.6B int8 | 8.3 | 10.8 | 11.0 | 10.0 [7.9, 12.3] | +3.2 [+1.5, +4.9], worse |
| ja (CER) | Nemotron 3.5 streaming 560 ms | 12.3 | - | - | 12.3 [8.1, 17.9] | not ranked (no room_light, mic_codec) |
| ko (CER) | SenseVoice small int8 | 7.2 | 8.4 | 7.9 | 7.9 [5.9, 9.9] | +2.8 [+1.4, +4.4], worse |
| ko (CER) | Qwen3-ASR 0.6B int8 | 4.0 | 5.2 | 6.1 | **5.1 [3.8, 6.5]** | best |
| ko (CER) | Nemotron 3.5 streaming 560 ms | 8.7 | - | - | 8.7 [2.3, 19.0] | not ranked (no room_light, mic_codec) |
| pt (WER) | Parakeet Ultra int8 | 8.8 | 9.2 | 9.4 | **9.1 [4.8, 14.1]** | best |
| pt (WER) | Parakeet TDT v3 int8 | 10.3 | 11.2 | 10.6 | 10.7 [5.8, 16.5] | +1.6 [+0.6, +2.6], worse |
| pt (WER) | Qwen3-ASR 0.6B int8 | 12.8 | 14.8 | 15.1 | 14.2 [9.6, 20.0] | +5.1 [+3.5, +6.9], worse |
| pt (WER) | Nemotron 3.5 streaming 560 ms | 15.0 | - | - | 15.0 [6.0, 24.6] | not ranked (no room_light, mic_codec) |
| ru (WER) | Parakeet Ultra int8 | 7.7 | 7.5 | 6.6 | **7.3 [4.1, 11.2]** | best |
| ru (WER) | Parakeet TDT v3 int8 | 11.4 | 11.6 | 9.6 | 10.8 [6.7, 15.7] | +3.6 [+1.7, +5.6], worse |
| ru (WER) | Qwen3-ASR 0.6B int8 | 22.8 | 19.3 | 24.3 | 22.1 [17.1, 27.2] | +14.8 [+10.4, +19.1], worse |
| ru (WER) | Nemotron 3.5 streaming 560 ms | 13.9 | - | - | 13.9 [7.4, 20.8] | not ranked (no room_light, mic_codec) |
| zh (CER) | SenseVoice small int8 | 4.1 | 4.8 | 4.5 | **4.5 [2.9, 6.3]** | +0.7 [-1.0, +2.3], tie |
| zh (CER) | Qwen3-ASR 0.6B int8 | 4.4 | 4.1 | 2.7 | 3.8 [2.1, 5.8] | best |

Reading: Parakeet Ultra is the best (or tied-best) engine in every language it covers (own-mic WER en 6.2, es 7.2, ru 7.3, fr 8.5, pt 9.1, de 10.1), and the two mic-side conditions cost it at most 3.3 points over clean speech (de); the reverb of `room_light` and the band-limited mu-law of `mic_codec` are not where engines separate. Parakeet v3 is level with Ultra in German and behind it everywhere else. Qwen3-ASR 0.6B is 2-15 points behind Ultra on the European languages (de 12.4, es 10.2, fr 15.8, pt 14.2, ru 22.1). Japanese: SenseVoice leads Qwen3-ASR 0.6B by +3.2 [+1.5, +4.9] points. Korean is the contested pair, so it got 100 utterances per condition instead of 30 (the table's Korean rows): Qwen3-ASR 0.6B leads SenseVoice by +2.8 [+1.4, +4.4]; the 30-utterance run said +2.8 [+0.1, +6.0]. Chinese is a tie (4.5 for SenseVoice against 3.8). English: Granite (6.1), Ultra (6.2) and Qwen3-ASR (6.9) are indistinguishable, and Nemotron alone, the one-pass route of a 4-core PC, is at 10.3 (9.4 clean).

### The two rankings

The first pass of the campaign ranked on every condition including the bleed ones; the user's decision
(2026-10-09) removes the bleed conditions from the own-mic choice. Both orders, best to worst:

| Language | Own mic (clean + mic-side): best to worst | All conditions incl. bleed: best to worst | Default |
|---|---|---|---|
| en | Granite TurboCTC Q8_0 (CPU, 4 threads) > Parakeet Ultra int8 > Qwen3-ASR 0.6B int8 > Parakeet TDT v3 int8 > SenseVoice small int8 > Nemotron 3.5 streaming 560 ms | Parakeet Ultra int8 > Granite TurboCTC Q8_0 (CPU, 4 threads) > Qwen3-ASR 0.6B int8 > SenseVoice small int8 > Parakeet TDT v3 int8 | Parakeet Ultra int8 |
| de | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 > Qwen3-ASR 0.6B int8 > Parakeet TDT v3 int8 | Parakeet Ultra int8 |
| es | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 > Qwen3-ASR 0.6B int8 > Parakeet TDT v3 int8 | Parakeet Ultra int8 |
| fr | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 |
| pt | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 |
| ru | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 > Parakeet TDT v3 int8 > Qwen3-ASR 0.6B int8 | Parakeet Ultra int8 |
| ja | SenseVoice small int8 > Qwen3-ASR 0.6B int8 | SenseVoice small int8 > Qwen3-ASR 0.6B int8 | SenseVoice small int8 |
| ko | Qwen3-ASR 0.6B int8 > SenseVoice small int8 | SenseVoice small int8 > Qwen3-ASR 0.6B int8 | Qwen3-ASR 0.6B int8 |
| zh | Qwen3-ASR 0.6B int8 > SenseVoice small int8 | Qwen3-ASR 0.6B int8 > SenseVoice small int8 | SenseVoice small int8 |

The orders differ in three places. **Korean flips**: Qwen3-ASR 0.6B is better on clean speech and the mic-side conditions,
SenseVoice under babble, so the first pass of the campaign had put SenseVoice there and the own-mic ruling puts Qwen3-ASR 0.6B
back. Parakeet v3 stands ahead of Qwen3-ASR 0.6B on the own microphone in German and Spanish and falls behind it once babble
counts (its babble collapse). In English Granite and Ultra swap places (a tie either way). Chinese is Qwen3-ASR 0.6B first in both
orders, by under a point; SenseVoice takes the language because the two tie and it costs 5.5 times less. The default of every language
follows the first order and the tie rule.

### Bleed: the VRChat mix, music and babble (other players' captions, speaker setups)

These rows play no part in the own-microphone choice. Music and babble at 10 dB under the speech, and the VRChat
mix (music, effects and two talkers about 12 dB under, microphone band 100 Hz-7 kHz).

| Language | Engine | music_snr10 | babble_snr10 | vrchat_mix | All bleed |
|---|---|---:|---:|---:|---:|
| de (WER) | Parakeet Ultra int8 | 12.6 | 11.4 | 16.3 | 13.4 |
| de (WER) | Parakeet TDT v3 int8 | 15.6 | 35.3 | 24.3 | 25.1 |
| de (WER) | Qwen3-ASR 0.6B int8 | 19.9 | 26.9 | 22.6 | 23.1 |
| de (WER) | Nemotron 3.5 streaming 560 ms | 23.5 | 34.7 | 24.9 | 27.7 |
| en (WER) | SenseVoice small int8 | 11.2 | 14.7 | 12.9 | 12.9 |
| en (WER) | Parakeet Ultra int8 | 8.8 | 10.6 | 9.1 | 9.5 |
| en (WER) | Parakeet TDT v3 int8 | 13.6 | 53.4 | 10.7 | 25.9 |
| en (WER) | Granite TurboCTC Q8_0 (CPU, 4 threads) | 7.3 | 10.9 | 10.7 | 9.6 |
| en (WER) | Qwen3-ASR 0.6B int8 | 9.9 | 14.7 | 7.3 | 10.6 |
| en (WER) | Nemotron 3.5 streaming 560 ms | 15.2 | 17.7 | 17.5 | 16.8 |
| es (WER) | Parakeet Ultra int8 | 8.3 | 8.6 | 8.9 | 8.6 |
| es (WER) | Parakeet TDT v3 int8 | 10.6 | 29.6 | 12.7 | 17.6 |
| es (WER) | Qwen3-ASR 0.6B int8 | 10.7 | 19.1 | 12.0 | 13.9 |
| es (WER) | Nemotron 3.5 streaming 560 ms | 12.2 | 23.6 | 17.1 | 17.6 |
| fr (WER) | Parakeet Ultra int8 | 15.2 | 17.3 | 21.7 | 18.1 |
| fr (WER) | Parakeet TDT v3 int8 | 19.5 | 38.5 | 34.7 | 30.9 |
| fr (WER) | Qwen3-ASR 0.6B int8 | 21.4 | 34.7 | 29.9 | 28.7 |
| fr (WER) | Nemotron 3.5 streaming 560 ms | 23.6 | 35.6 | 32.2 | 30.5 |
| ja (CER) | SenseVoice small int8 | 6.9 | 20.5 | 12.0 | 13.1 |
| ja (CER) | Qwen3-ASR 0.6B int8 | 10.9 | 34.0 | 18.8 | 21.2 |
| ja (CER) | Nemotron 3.5 streaming 560 ms | 15.0 | 39.1 | 24.1 | 26.1 |
| ko (CER) | SenseVoice small int8 | 8.7 | 22.8 | 11.8 | 14.4 |
| ko (CER) | Qwen3-ASR 0.6B int8 | 6.1 | 39.0 | 12.5 | 19.2 |
| ko (CER) | Nemotron 3.5 streaming 560 ms | 10.2 | 19.3 | 12.9 | 14.1 |
| pt (WER) | Parakeet Ultra int8 | 11.2 | 19.2 | 14.5 | 15.0 |
| pt (WER) | Parakeet TDT v3 int8 | 16.8 | 35.6 | 18.3 | 23.6 |
| pt (WER) | Qwen3-ASR 0.6B int8 | 17.1 | 37.0 | 23.3 | 25.8 |
| pt (WER) | Nemotron 3.5 streaming 560 ms | 15.4 | 29.5 | 18.5 | 21.1 |
| ru (WER) | Parakeet Ultra int8 | 9.6 | 12.5 | 12.1 | 11.4 |
| ru (WER) | Parakeet TDT v3 int8 | 11.6 | 16.0 | 16.0 | 14.5 |
| ru (WER) | Qwen3-ASR 0.6B int8 | 32.4 | 34.0 | 29.0 | 31.8 |
| ru (WER) | Nemotron 3.5 streaming 560 ms | 20.8 | 33.2 | 25.2 | 26.4 |
| zh (CER) | SenseVoice small int8 | 4.4 | 40.3 | 12.8 | 19.2 |
| zh (CER) | Qwen3-ASR 0.6B int8 | 2.0 | 46.3 | 7.8 | 18.7 |

Parakeet Ultra is the most robust engine for the languages it covers; Parakeet v3 and Qwen3-ASR 0.6B lose far more to
babble (WER en 53 % and 15 % against 11 %, fr 39 % and 35 % against 17 %); for Japanese and Chinese SenseVoice and Qwen3-ASR 0.6B trade places by condition, and
for Korean SenseVoice holds up where Qwen3-ASR 0.6B does not (babble 22.8 against 39.0).

### Auto mode (no language set, the default of a new install)

Engines given no language hint on `quick` (10 utterances per language, the shipped pipeline 6): the offline
engines detect the language themselves; the shipped pipeline adds Silero VAD, the language-ID engine and
Nemotron partials for the languages it streams. Clean WER / CER in %, the VRChat mix in brackets.

| Language | Qwen3-ASR 0.6B alone | Parakeet Ultra alone | SenseVoice alone | Shipped auto pipeline |
|---|---:|---:|---:|---:|
| en | 11.5 (13.9) | 8.1 (8.6) | 11.0 (15.8) | **6.2** (11.6) |
| de | 15.2 (19.4) | **6.7** (13.9) | - | 11.5 (20.8) |
| es | 7.5 (11.2) | **6.1** (8.4) | - | 13.1 (16.9) |
| fr | 15.1 (24.3) | **8.8** (18.8) | - | 17.3 (35.3) |
| pt | 12.3 (10.3) | **3.9** (5.9) | - | 5.5 (17.4) |
| ru | 23.8 (32.6) | **6.1** (9.9) | - | 22.0 (25.4) |
| ja | 12.2 (26.0) | - | 10.8 (19.4) | **8.9** (20.6) |
| ko | 5.6 (16.3) | - | 7.6 (9.7) | **1.8** (9.7) |
| zh | **1.3** (5.4) | - | 2.2 (14.0) | 1.4 (5.3) |

The language-ID engine named the right language for 105 of 108 utterances (the three misses were de heard as ko,
fr as ja and pt as en). Parakeet Ultra transcribes its 25 languages without a hint just as well as with one, but
the auto route has one final engine, Qwen3-ASR 0.6B, which alone is 1.2 to 4 times worse than Ultra on the
European languages (ru 23.8 against 6.1). An auto user therefore gets noticeably worse captions in those
languages than a user who sets the language; recommendation under "For later".

### Moonshine and long utterances (the 9 s cut)

The lightest English engine has one defect, which the P7 smoke run of the speech benchmark found and commit `f8eedb8` fixed. sherpa-onnx 1.13.8 returns an empty result, with no error, when the Moonshine v2 model decodes 148,351 samples (9.272 s at 16 kHz) or more in one call; 148,350 samples decode, and every longer length fails (checked up to 52 s). Before the fix every caption longer than that came back empty. Sigil now cuts an utterance over 9.0 s at quiet points (a 140 ms span around the quietest frame, 0.27 s of margin below the limit), decodes the pieces and joins the texts; a piece that still comes back empty is decoded again as two halves, and the engine option `moonshineMaxSegmentMs` can only lower the limit (`0` never cuts). The engine's behaviour is in [Moonshine inside Sigil](../OPERATOR_REFERENCE.md#moonshine-inside-sigil-sherpa-onnx); the runner that found it is in the [developer guide](../DEVELOPER_GUIDE.md#speech-benchmark---headless-speech-bench-p7).

| English utterances of the benchmark corpus (`p7-synthetic-v1`, `content_sha256` `459c2ff05c92...`) | Word error rate before the fix | After the fix |
|---|---:|---:|
| Over 9.3 s: 56 of the corpus's 100 English utterances (up to 19.7 s) | 100 % (empty result) | about 19-20 % (19.9 % in the fix's own check; 19.2 % with the shipped 140 ms span, 19.1 to 19.9 % for the other spans tried) |
| 8 to 9 s | 11 % (decoded whole) | 11 % (decoded whole, as before); cut at 8 s instead they scored 19 %, which is why the limit sits at 9 s |

How to read it: the first row is the cut-span study on the 7 recording conditions it used (432 cuts; 9.6 % of the cuts fell inside a word with the quietest single frame, 5.1 % with the 140 ms span), and the second row is the same corpus's 8 to 9 s utterances. These figures come from the development runs of the fix (the commit message and the doc comments of `long_utterance.rs` and `sherpa_offline.rs`), not from a campaign spec. The ignored real-model tests reproduce them (`SIGIL_APP_DATA_DIR` with `runtime/sherpa-onnx` and `stt/moonshine-base-en-int8`, `SIGIL_TEST_MODELS_DIR` with the corpus, then `cargo test real_moonshine -- --ignored`), and a sherpa-onnx upgrade has to re-measure the limit (`real_moonshine_sherpa_length_limit_is_still_where_the_split_assumes`). The screening row for Moonshine (13.4 %) includes the cut: 6 of its 10 English utterances run from 10.0 to 14.9 s, and their hypotheses in the screening run are not empty, so it is not a whole-utterance figure.

### Browser speech recognition (Chrome and Edge Web Speech, the old default), English

Before the built-in engine, the default speech-to-text was the browser's own Web Speech recognition: Chrome (Google's servers) or Edge (Microsoft's), run from Sigil's microphone page. It was measured once, in English, on the corpus and with the scorer of the engines above, so that the one-time prompt which offers the built-in engine to a Chrome or Edge profile can quote real numbers (the prompt's figures are the `Finals, clean` column, rounded: 6, 21 and 10 errors per 100 words; `localOffer.ts` holds them and its test reads this table). WER in %, 95 % interval in brackets, lower is better.

| Engine | Screening, clean (10 utt) | Finals, clean (30 utt) | Finals, room_light (30 utt) | Finals, mic_codec (30 utt) | Own mic, pooled (90 utt) |
|---|---:|---:|---:|---:|---:|
| Chrome Web Speech (Google), Chrome 154 | 27.3 [14.1, 43.9] | 20.8 [13.6, 29.4] | 22.0 [14.2, 31.1] | 23.8 [16.2, 33.7] | 22.2 [15.1, 30.4] |
| Edge Web Speech (Microsoft), Edge 155 | 6.7 [3.9, 9.4] | 9.8 [6.3, 14.1] | 11.1 [5.8, 17.4] | 7.1 [5.0, 9.4] | 9.3 [6.4, 12.1] |
| Parakeet Ultra int8 (the built-in engine's English final) | 8.1 | 6.1 | 6.3 | 6.3 | 6.2 [4.4, 8.3] |

Reading: on the own microphone Chrome makes about 3.6 times the errors of the built-in engine (22.2 against 6.2; the intervals do not overlap) and Edge about 1.5 times (9.3 against 6.2; the intervals overlap, so at 30 utterances per cell Edge is behind by a margin the sample cannot pin down). On the 10-utterance screening Edge is ahead of Parakeet Ultra (6.7 against 8.1), a tie at that size. Timing: a browser final lands at the end of speech (median 0.16 s before the end of the clip in Chrome's clean and room_light runs; between 0.17 s before and 0.16 s after across the six runs). Each 6-minute run saw 1 (Chrome) to 4 (Edge) `network` errors, each ended the browser's session and was followed by an immediate restart of the microphone page (0.07 to 0.17 s of gap per run in total).

Method: Chrome 154 and Edge 155, installed and run headless, were fed the P7 clips (concatenated, with 2 s gaps) by a Chromium fake microphone through `getUserMedia`, a resampler and `recognition.start(track)`, with the settings of Sigil's `mic.html`. The hypotheses were attributed to clips by arrival time and scored with the P7 scorer.

Caveats: one run per browser and condition (30 clips each); the results were not independently re-scored (the verifier was stopped); the other eight languages were measured too but are not published, by the user's choice; the rig (a Python harness driving the browsers) and its raw runs are not part of the repository, so this table is the record.

## Speech-to-text cost and caption latency

CPU rows on the 6-core slice; "next to a game" adds `h3_cpu_game`. RTF = decode time / audio time.

### Decoding a whole utterance (the second pass)

| Engine | Threads | RTF idle | RTF next to a game | decode p50 / p95 ms (idle) | CPU ms per audio second | RAM peak MB |
|---|---:|---:|---:|---:|---:|---:|
| Parakeet Ultra int8 | 2 | 0.085 | 0.084 | 774 / 1326 | 182 | 898 |
| Parakeet TDT v3 int8 | 2 | 0.086 | 0.089 | 776 / 1334 | 183 | 938 |
| Qwen3-ASR 0.6B int8 | 2 | 0.219 | 0.225 | 2095 / 3645 | 415 | 2368 |
| Qwen3-ASR 0.6B int8 | 4 | 0.166 | 0.166 | 1610 / 2989 | 616 | 2374 |
| SenseVoice small int8 | 2 | 0.038 | 0.038 | 390 / 764 | 76 | 421 |
| Moonshine base-en int8 (en only) | 2 | 0.020 | 0.021 | 175 / 285 | 51 | 286 |
| Granite TurboCTC Q8_0 (en only) | 4 | 0.059 | 0.077 | 472 / 698 | 227 | 621 |
| Whisper base, ggml fp16 | 4 | 1.087 | 1.184 | 10261 / 11027 | 3795 | 422 |

Next to the game the offline engines barely change (the burners take cores that the 6C slice has to spare).
Per audio second the lightest is Moonshine (51 ms CPU), then SenseVoice (76), Parakeet (182), Granite (227),
Qwen3-ASR (415 on 2 threads) and Whisper base (3.8 s: that row is the fp16 base model; the quantized models of
the catalog use integer kernels and were not available to measure under the 50 MB download limit, so the
Whisper figure is an upper bound, not a verdict). Qwen3-ASR 0.6B is the only engine that holds 2.4 GB of RAM, and
on the 14 s Korean clips it needs 2.9 s on 2 threads (3.0 s next to a game), against 0.56 s for SenseVoice: the
Korean default trades final-caption speed (the live partials come from Nemotron meanwhile) for 2.8 CER points.

### Live partials (Nemotron, real-time stream feed)

| Engine | RTF idle | RTF game | final latency after the end of speech p50 / p95 ms (idle) | (game) | CPU cores while speaking (idle / game) |
|---|---:|---:|---:|---:|---:|
| Nemotron 3.5 streaming 560 ms | 0.37 | 0.52 | 204 / 330 | 372 / 504 | 0.31 / 0.44 |
| Nemotron 3.5 streaming 160 ms | 0.98 | 1.39 | 380 / 401 | 1695 / 3187 | 0.80 / 0.98 |

First partial after the first spoken word (clip onset at -40 dBFS): 1.34 s idle / 1.42 s next to a game (p50; p90
3.3 s) for the 560 ms model. The 160 ms variant is faster to a first word (0.93 / 1.23 s) but needs a whole core
(RTF 0.98), falls behind real time next to a game (RTF 1.39, final after 1.7 s) and should not be offered on 6-core
PCs.

### The whole caption pipeline (Silero VAD, router, two passes) at real time

| Route (6 cores; en, de, ru, ja, zh, 3 clips each) | Load | Final after speech ends p50 / p95 ms | CPU cores while speaking |
|---|---|---:|---:|
| Partials + final (two pass) | idle | 677 / 1119 | 0.39 |
| Partials + final (two pass) | game | 903 / 1383 | 0.48 |
| One pass (Nemotron's own finals in English, the final engine alone elsewhere) | idle | 319 / 691 | 0.18 |
| One pass (Nemotron's own finals in English, the final engine alone elsewhere) | game | 404 / 831 | 0.22 |
| Final only (Interim results off; Parakeet Ultra) | idle | 587 / 965 | 0.24 |
| Final only (Interim results off; Parakeet Ultra) | game | 800 / 1244 | 0.28 |

| Route (4 cores; en, ja, 3 clips each) | Load | Final after speech ends p50 / p95 ms | CPU cores while speaking |
|---|---|---:|---:|
| Partials + final (two pass) | game | 815 / 1410 | 0.50 |
| One pass (Nemotron's own finals in English, the final engine alone elsewhere) | game | 307 / 439 | 0.25 |
| Final only (Interim results off; Parakeet Ultra) | game | 845 / 1299 | 0.21 |

Per language, idle / next to a game (p50 ms from the end of speech to the final):

| Language | Partials + final | One pass | Final only (Interim results off) |
|---|---:|---:|---:|
| en | 873 / 1140 | 161 / 256 | 685 / 863 |
| de | 679 / 957 | 375 / 497 | 380 / 494 |
| ru | 764 / 1068 | 691 / 831 | 609 / 835 |
| ja | 421 / 670 | 254 / 323 | - |
| zh | 238 / 308 | 277 / 367 | - |

Reading: a streamer who talks continuously costs the CPU 0.4-0.5 cores for the two-pass English route, 0.2 for one
pass, 0.2-0.3 for finals only; with Silero VAD gating the engines, the average over real speech (a third of the time)
is a few hundredths of a core and the models' RAM (Nemotron 0.8 GB + Parakeet Ultra 0.9 GB for English, 1.2 GB for ja
with SenseVoice, 3.2 GB for ko with Qwen3-ASR) is the larger cost. Final captions arrive 0.2-1.1 s after the speech
ends in the languages of the table above; the game adds 0.1-0.3 s. **Korean is the exception**: its Qwen3-ASR 0.6B final
comes about one decode after the end of speech, 2.2 s and 3.7 s on the two 10-18 s clips of the route check below (RTF 0.2),
so a Korean streamer should expect final lines roughly 2-4 s after a long sentence, with Nemotron's partials on screen meanwhile.
The 4-core slice (8 logical processors) next to the same game keeps the two-pass
English final at 1.2 s (6C: 1.1 s) and 0.50 cores: two passes work on 4 cores with SMT too, so the
`cpuTwoPassMinCores` threshold of 6 is conservative; it was left at 6 because 4 cores without SMT could not be
emulated.

**Route check of the final tree (P7-I0, 2026-10-10).** The release `Sigil.exe` built from the final tree (it embeds `stt-routing.json` as committed) ran the shipped pipeline (Silero VAD, router, two passes, tier `cpu`, the 6-core slice, idle) on two clips each of ko, ja and ru (`quick` subset, clean): the planned engines were Nemotron 3.5 streaming 560 ms partials with a **Qwen3-ASR 0.6B final for Korean** (`stt/qwen3-asr-0.6b-int8`), a **SenseVoice final for Japanese** (`stt/sense-voice-small-int8`) and a **Parakeet Ultra final for Russian** (`stt/parakeet-ultra-int8`), all on the CPU with nothing missing; no error and no empty hypothesis. Korean's finals arrived 3.7 s after the end of a 17.6 s utterance (two segments, RTF 0.21) and 2.2 s after a 10.3 s one (RTF 0.22); Japanese 1.0 s (10.4 s and 15.6 s clips), Russian 1.4 s and 1.7 s (8.6 s and 12.8 s clips). Six clips, one run: it confirms the routes, not a latency table. Report `p7i0-route-check` (spec `a74f74efa78c`, corpus `459c2ff05c92`).

Korean: the pipeline table above measured the Korean route when it still had a SenseVoice final (0.4 s idle / 0.6 s next to a game for a two-pass final). The own-microphone ruling puts the Qwen3-ASR 0.6B final back, and the route check just above is the pipeline evidence for it: finals 2.2 s and 3.7 s after the end of speech on a 10.3 s and a 17.6 s clip (idle, six clips in all, so a confirmation and not a latency table). That matches one decode of the last segment, 2.9 s for a 14 s utterance on 2 threads idle (3.0 s next to a game) in the decode-speed table, roughly 1.2 s for a 5 s sentence (RTF 0.2), with Nemotron's partials on screen meanwhile. The latency columns of the pipeline tables above leave the three Korean clips out for that reason; the CPU column is the run's mean, which includes them.

## Translation

`Sigil.exe --headless speech-bench` `mt_quality` on three sets: 240 casual caption lines (10 directions, 24 each),
WMT24++ and FLORES-200 (24 lines per direction), chrF2 (sacrebleu-equal). The light translators: NLLB-200
distilled 600M int8 (in process, 2 threads; greedy and beam 4) and Hy-MT2 1.8B Q4_K_M in the bundled llama-server on
the CPU and on Vulkan. Bergamot is **not built** (its sources are about ten archives of unknown size and the build
was not attempted under the 50 MB download limit; ADR 0008 and `scripts/runtime-packs/bergamot/README.md` keep the
steps), so NLLB is the zero-VRAM fallback.

chrF2, lines pooled over the three sets (higher is better; bold = best of the direction):

| Direction | Hy-MT2 1.8B (CPU) | Hy-MT2 1.8B (Vulkan) | NLLB-600M beam 4 | NLLB-600M greedy |
|---|---:|---:|---:|---:|
| de-en | 65.9 | 64.3 | **69.5** | 69.4 |
| en-de | 57.9 | 57.8 | **59.1** | 56.0 |
| en-es | 64.0 | 64.0 | **66.8** | 62.4 |
| en-fr | 49.6 | 49.5 | **57.4** | 55.1 |
| en-ja | 36.6 | **36.8** | 26.4 | 20.5 |
| en-ko | **35.3** | 33.9 | 30.4 | 24.2 |
| en-pt | 59.3 | 59.2 | 62.7 | **63.2** |
| en-ru | **40.4** | 40.1 | 40.3 | 35.7 |
| en-zh | **41.3** | 41.0 | 25.1 | 22.4 |
| en-zh-Hant | **37.0** | 36.7 | 28.4 | 10.9 |
| es-en | 62.8 | 62.8 | **67.2** | 66.8 |
| fr-en | 61.9 | 61.6 | **68.1** | 67.1 |
| ja-en | 54.5 | 54.6 | **56.9** | 55.8 |
| ko-en | 54.1 | 54.0 | 54.8 | **55.0** |
| pt-en | 68.4 | 68.0 | **75.5** | 73.7 |
| ru-en | 59.2 | 58.9 | **62.8** | 60.6 |
| zh-en | **61.5** | 61.1 | 57.0 | 56.4 |
| **all lines** | 53.5 | 53.1 | 52.7 | 49.8 |

Hy-MT2 1.8B wins by 5-16 points into Japanese, Korean and Chinese over the better NLLB setting (en to ja 36.6
against 26.4 for beam 4 and 20.5 for greedy; en to zh 41.3 against 25.1 / 22.4) and by 4.5 points from Chinese.
NLLB with beam 4 is 1-8 points better into French, Spanish, Portuguese and German and from German, Spanish, French,
Portuguese, Russian and Japanese (pt to en 75.5 against 68.4; en to fr 57.4 against 49.6), and the two tie into
Russian and from Korean; beam 4 costs 4x the latency of Hy-MT2 on the CPU. The differences that decide are large
enough that MetricX-24 or a blind A/B was not needed: the Hy-MT2-first chain keeps the audience's languages (ja,
ko, zh) on the better engine. Whether the European pairs should go to NLLB first on a processor-only PC was measured
afterwards, with beam 4's own latency and cost (1.6 s a casual line, 4.2 times Hy-MT2's 0.39 s; the NLLB rows of
the table below are greedy): see "European pairs on a processor-only PC" below.

Latency and cost per caption line (casual set; the engine rows run line after line, so the CPU figures are the burst
a single caption causes, not an average):

| Engine | Load | Latency per line p50 / p95 ms | CPU while translating (cores, minus the load stand-in) | RAM, process tree (MB) | VRAM (MB) |
|---|---|---:|---:|---:|---:|
| NLLB-200 600M greedy (in process) | idle | 947 / 1344 | 1.9 | 2609 | - |
| NLLB-200 600M greedy (in process) | next to a game | 957 / 1337 | 1.6 | 2624 | - |
| Hy-MT2 1.8B, llama-server on the CPU | idle | 379 / 633 | 5.4 | 2418 | - |
| Hy-MT2 1.8B, llama-server on the CPU | next to a game | 422 / 641 | 4.7 | 2410 | - |
| Hy-MT2 1.8B, llama-server on Vulkan | idle | 51 / 83 | 0.5 | 1597 | 1507 |
| Hy-MT2 1.8B, llama-server on Vulkan | GPU load (VRChat + OBS stand-in) | 116 / 195 | 0.3 | 1759 | 1568 |

The translator that runs on the CPU is not light: llama-server takes every core it plans (physical cores minus 2,
clamped to 2-8: four on a 6-core PC, six on the dev rig) for 0.4 s per caption line and holds 2.4 GB of RAM. NLLB
in process takes about 2 cores for 0.9 s and 2.6 GB. The Vulkan variant costs about half a core and 1.5 GB of
VRAM and answers in 51 ms idle, 116 ms under the VRChat + OBS GPU stand-in. Which of them a streamer should run is
decided by where the game has headroom: the app footprint section shows what the whole pipeline costs with captions
arriving at speaking pace.

### European pairs on a processor-only PC (MT-Europe, 2026-10-11)

The question: on a PC where the bundled translator runs on the processor, should English with French, Spanish, Portuguese,
German, Italian and Dutch go to NLLB-200 600M with **beam 4** first, instead of Hy-MT2 1.8B? The P7 tables above rank the two on
chrF2 (NLLB 1 to 8 points ahead on some European pairs) but carry NLLB's latency and cost only for greedy decoding. This run
measures beam 4 itself, next to Hy-MT2 on the processor, with the same release `Sigil.exe`, the same model store and the same
below-normal 12-logical-processor slice, under the timing lock, one campaign in sequence (specs `mt_europe.json`,
`mtspeed_nllb-beam4_none.json`, `mtspeed_nllb-beam4_h3_cpu_game.json` and the existing Hy-MT2 speed specs; the tables come from
`campaign/mt_europe_table.py`).

**Text.** The European directions of the P7 sets, 24 lines each: the casual caption lines (en to de / es, de / es to en), WMT24++
(en to de / es / fr / pt; its social and speech domains first) and FLORES-200 (de / es / fr / pt to en). WMT24++ has no Italian or
Dutch, so the campaign adds `flores200-eu`, built offline from the FLORES-200 archive the P7 corpus already holds: the same
devtest sentences in the directions the first set lacks (en to de / es / fr / pt / it / nl and it / nl to en;
`corpus/build_text_sets.py`, which reproduces the three existing sets byte for byte). 480 lines per engine. There are no casual lines
for French, Portuguese, Italian or Dutch (nobody wrote references for them), so those pairs rest on WMT24++ and FLORES-200 text, which is
longer and more formal than a caption. **Scoring** is the P7 rule: chrF2 per set (sacrebleu defaults, one reference per line), a
direction is the line-weighted mean of its sets, and the interval is a paired bootstrap that resamples the lines inside every set
(1000 resamples, the same draw for both engines). A language pair pools both of its directions with English.

chrF2 per language pair (higher is better):

| Pair | Lines | NLLB-600M beam 4 | Hy-MT2 1.8B (CPU) | Difference [95 % interval] | NLLB ahead in |
|---|---:|---:|---:|---:|---:|
| en and fr | 72 | **65.3** | 59.3 | +6.0 [+3.8, +8.5] | 100 % of resamples |
| en and pt | 72 | **69.8** | 65.0 | +4.8 [+2.3, +7.3] | 100 % |
| en and nl | 48 | **62.6** | 59.0 | +3.6 [+1.2, +5.7] | 100 % |
| en and es | 120 | **64.6** | 61.7 | +2.8 [+1.1, +4.6] | 100 % |
| en and de | 120 | 63.9 | 62.2 | +1.7 [-0.6, +3.9] | 91 % |
| en and it | 48 | 60.8 | 60.0 | +0.8 [-2.6, +3.8] | 69 % |

By direction, with the chrF2 of each set (NLLB beam 4 / Hy-MT2; a dash = the set has no such direction). The casual lines are the
captions of the audience; the other three sets are text a streamer reads out less often:

| Direction | Casual lines | WMT24++ | FLORES-200 | FLORES-200, new directions | Pooled difference [95 % interval] |
|---|---:|---:|---:|---:|---:|
| en to fr | - | 57.4 / 50.5 | - | 70.4 / 65.3 | +6.0 [+3.1, +9.1] |
| fr to en | - | - | 68.1 / 62.0 | - | +6.1 [+2.8, +9.0] |
| en to pt | - | 62.7 / 58.4 | - | 71.2 / 69.0 | +3.3 [-0.4, +6.6] |
| pt to en | - | - | 75.5 / 67.7 | - | +7.8 [+4.6, +10.7] |
| en to nl | - | - | - | 59.8 / 57.3 | +2.6 [-1.5, +6.5] |
| nl to en | - | - | - | 65.4 / 60.8 | +4.6 [+2.2, +7.1] |
| en to es | 73.5 / 68.1 | 60.1 / 59.9 | - | 54.8 / 55.1 | +1.8 [-0.3, +3.9] |
| es to en | 73.7 / 68.1 | - | 60.6 / 57.4 | - | +4.4 [+1.4, +7.7] |
| en to de | 67.6 / 65.9 | 50.5 / 51.7 | - | 62.5 / 60.8 | +0.8 [-2.0, +3.7] |
| de to en | 71.5 / 70.5 | - | 67.5 / 62.1 | - | +3.2 [-0.6, +7.1] |
| en to it | - | - | - | 57.8 / 58.7 | -0.9 [-7.2, +4.2] |
| it to en | - | - | - | 63.8 / 61.3 | +2.4 [-0.9, +5.7] |

NLLB beam 4 is never significantly behind: the worst direction is Italian from English (-0.9, interval -7.2 to +4.2, 24 lines). Its
beam-4 output is deterministic and reproduces the P7 run to the digit (for example en to es casual 73.5, WMT24++ 60.1); Hy-MT2 samples at
temperature 0.2, so its sets moved by up to 1.6 points between the two runs, which is the size of difference these 24-line sets
cannot resolve (en to de and en to es on WMT24++ are ties).

Speed and cost per caption line (the P7 speed spec: 10 casual lines per direction, 100 lines in all, 10 directions including the
Asian ones; cores are the process tree's mean while a line translates, minus the game stand-in's own CPU where it ran, 2.3-2.4
cores):

| Engine | Load | Latency per line p50 / p95 ms | Cores while translating | RAM, process tree (MB) |
|---|---|---:|---:|---:|
| NLLB-600M beam 4, 2 threads (in process) | idle | 1641 / 2432 | 1.8 | 2649 |
| NLLB-600M beam 4 | next to a game | 1881 / 2759 | 1.7 | 2636 |
| NLLB-600M greedy (rerun; the P7 row above is 947 / 1344) | idle | 862 / 1231 | 1.8 | 2608 |
| Hy-MT2 1.8B, `llama-server` on the processor with 6 threads (what the plan picks on this 8-core rig) | idle | 390 / 624 | 5.0 | 2406 |
| Hy-MT2, 6 threads | next to a game | 414 / 680 | 4.8 | 2403 |
| Hy-MT2, 4 threads (what the plan picks on a 6-core PC: physical cores minus 2) | idle | 373 / 613 | 3.7 | 2364 |
| Hy-MT2, 4 threads | next to a game | 410 / 657 | 3.5 | 2352 |

Beam 4 takes 1.9 times greedy's time (1.64 s against 0.86 s) for the same 1.8 cores, and 4.2 times Hy-MT2's: NLLB spends about 3.0
core-seconds on a casual line (1.8 cores for 1.6 s), Hy-MT2 1.4 on four threads (3.7 cores for 0.37 s) or 2.0 on six, so the processor
time of a line doubles while the burst the game sees is half as tall. The four-thread rows come from a `llama-server` started by hand
with the flags the app uses on the processor (context 4096, no GPU layers, `-fa auto`, one slot, `--device none`, `-t 4`) on the
same slice and driven through the `openai` backend, because the bench does not forward `threads` to the bundled server; the thread
count changes the cores and not the latency here (373 ms against 390 ms). The quality run timed every line too: on its casual
lines (4 directions, 96 lines) NLLB beam 4 took 1.4 s at the median (1.2 to 1.5 s per direction; p95 2.0 s) and Hy-MT2 0.37 s (p95 0.49 s);
on the formal WMT24++ and FLORES-200 sentences (384 lines, median 140 characters against 46 for the 96 casual lines, longest 401) NLLB beam 4 took 4.4 s (4.0 to 4.8 s per
direction; p95 8.2 s, slowest 14.5 s) and Hy-MT2 1.0 s (p95 2.1 s). The router gives a final 6 s per 120 characters; NLLB beam 4 used
52 % of that at the median, 18 of 480 lines used more than 80 % and one overran it (8.2 s for 157 characters), so the built-in route gives
NLLB's attempt at a final 1.5 times the budget; a line that still overruns falls through to the AI model.

**Decision, per tier.**

| Tier | Decision | Why |
|---|---|---|
| CPU only (the bundled translator runs on the processor: variant `cpu`, no GPU layers, or `auto` resolved to the processor) | **NLLB beam 4 first** for English with French, Portuguese, Dutch and Spanish, both directions; German and Italian stay on Hy-MT2 first | The 95 % interval of the pair difference is above zero for those four (+6.0, +4.8, +3.6, +2.8) and straddles zero for German (+1.7) and Italian (+0.8). NLLB spends 1.7-1.8 cores instead of 3.5-5.0 for 1.6-1.9 s instead of 0.4 s, and holds the same 2.4-2.6 GB; a caption line arrives about 1.2 s later, and the AI model still answers when the NLLB model is not installed or a line overruns |
| Mid-range GPU (8-12 GB) | Unchanged: Hy-MT2 first | On Vulkan Hy-MT2 answers in 51 / 83 ms (116 / 195 ms under the VRChat + OBS GPU stand-in) for 0.3-0.5 cores and scores like the processor build (53.1 against 53.5 on the P7 sets), so NLLB would add 1.7-1.8 cores for 1.6-1.9 s per line for the same gain |
| High-end GPU (16 GB or more) | Unchanged: Hy-MT2 first | Same numbers; the larger card does not change the cost side |
| Japanese, Korean, Chinese (any tier) | Unchanged: Hy-MT2 first | Hy-MT2 wins by 5-16 chrF2 points (en to ja 36.6 against 26.4) |

**How it is applied** (`translation/routingTable.ts`, read by the router): a built-in route, not a saved setting. While the host reports that
the bundled translator is on the processor, the eight directions above (the source language the pipeline resolved for the line: your Source language setting, else the language the speech engine reported; regional spellings accepted) use the
chain NLLB, Hy-MT2, Firefox Translations and ask NLLB for beam 4 on finals (`TranslationRequest.beam`; the slice's own **Beam** is a
floor) with 1.5 times the router's attempt budget; a line with no known source, a line with a glossary term or any line while the context background text is filled in (NLLB uses neither), interims and every other pair are unchanged. A rule of your own for the pair under **Language pair rules**,
or an edited chain, always wins, so the route can be turned off without a new switch. The chain members are the default chain's, so
nothing new starts or downloads: a PC that never installed NLLB (0.9 GB, offered in Models & Runtimes) behaves exactly as before.
The tier is read per line, so the route starts once the model server's variant is known.

**Limits.** Neither engine was given the rolling context. The bench's `context: previous` takes the previous line of the same document, and the WMT24++ and FLORES-200 lines were drawn at random from their documents, so the sets have no consecutive lines to give; the app gives the AI model up to three earlier lines and NLLB cannot read them. The gain above is for a line translated alone; what the context adds to the AI model on these pairs is not measured (it would need a European set of consecutive caption lines). One reference per line and chrF2 only (no MetricX-24, no blind A/B; the P7 caveat about close engines applies to
German and Italian, which may be better or worse than a tie). French, Portuguese, Italian and Dutch have no casual caption lines, and
Italian and Dutch rest on one 24-line FLORES-200 set each. The campaign ran on the dev rig (a faster processor than a typical 6-core
PC); both engines slow down together, the ratio is what carries over. The opt-in larger models (Hy-MT2 7B, NLLB 1.3B) were not tried.
A CPU-only friend's `export.json` of `mt_europe.json` would add a second processor.

## Text-to-speech

`tts_listening` / `cpu_speed` on the casual prompts (10 per language: en, ja, zh), no listening study.

| Engine | Time to first audio p50 / p95 ms | Real-time factor | CPU while speaking | RAM |
|---|---:|---:|---:|---:|
| Kokoro v1.0, 4 threads, idle | 737 / 891 | 0.36 | 3.0 cores | 682 MB |
| Kokoro v1.0, 4 threads, next to a game | 792 / 954 | 0.39 | 2.8 cores | 687 MB |
| Kokoro v1.0, 2 threads, idle | 1118 / 1345 | 0.55 | 1.8 cores | 683 MB |
| Kokoro v1.0, 2 threads, next to a game | 1205 / 1466 | 0.59 | 1.7 cores | 691 MB |
| Windows SAPI, Zira / David Desktop (en, whole sentence rendered) | 15 / 37 and 10 / 28 | 0.004 / 0.003 | 3.3 / 2.1 ms per audio second | 134 / 128 MB |

All 30 prompts of every Kokoro run rendered without an error (English, Japanese and Mandarin voices). Kokoro is the local default and costs a burst of 2-3 cores for the length of a sentence (first sentence audible
after 0.7-0.8 s on 4 threads, 1.1-1.2 s on 2); the plan's "400 ms" target holds only when the opening clause is
short (P2), so the first-clause split was left as it is (shorter splits trade prosody and no listening study
was run). SAPI is essentially free but the voices are the two English ones Windows ships. Edge TTS is a cloud
service reached through an unofficial protocol; the runner has no Edge backend and no request was sent to the
service from the campaign, so it was not timed (its client is covered by the mock-server tests of P6-G; manual check:
Settings, TTS, Edge, speak a line).

## App footprint

What the whole app costs, measured on the real release `Sigil.exe` (frontend embedded) in a scratch Windows profile: the process tree (WebView2 and the translation server included) sampled every 2 s for two minutes after a one-minute warm-up, pinned to the 6-core slice at below-normal priority, with the game stand-in ('next to a game': four Node processes spinning 60 % of every 100 ms) beside it. The captions are the real engines listening to the dev PC's headset microphone in an ordinary room, so with nobody speaking they idle between the voice detector's hits; the cost of speech itself is the pipeline table above. The OBS and translation rows add a driver that opens `obs-captions.html` in headless Chromium (the stand-in for OBS's browser source) and feeds the host about six interim captions a second for 40 % of the time, a final after each sentence, over the host's own pubsub socket (in the OBS row the page changed its caption 208 times in two minutes and ended on the last sentence; the translation rows delivered 242 `text.translation` events for 314 caption events). RAM is the private memory of the whole tree (working sets add up shared WebView2 pages and read about 20 % higher). The release exe behind these rows was built before the `SIGIL_START_MINIMIZED` switch existed, so the harness minimized its window itself a few seconds after it appeared (the window was minimized for the whole measured part, as it is now); the scripts start the app minimized from the first frame and refuse an exe that predates the switch (campaign README, "Footprint of the real app").

| Scenario | Load | Sigil CPU (cores, mean / p95 / max) | Sigil RAM, private (MB) | Sigil GPU memory (MB) | OBS page (Chromium) CPU / RAM |
|---|---|---:|---:|---:|---:|
| App open, speech-to-text off | quiet PC | 0.00 / 0.01 / 0.02 | 336 | 64 | - |
| Captions on (English, Nemotron + Parakeet Ultra), nobody speaking | quiet PC | 0.02 / 0.03 / 0.07 | 1750 | 48 | - |
| Captions on (English, Nemotron + Parakeet Ultra), nobody speaking | next to a game | 0.03 / 0.04 / 0.31 | 1765 | 49 | - |
| Captions on, language auto (the new-install default: Nemotron + Qwen3-ASR + language ID) | next to a game | 0.09 / 0.36 / 0.74 | 2756 | 48 | - |
| Captions on, interim results off | next to a game | 0.05 / 0.19 / 0.23 | 1003 | 38 | - |
| Captions + OBS caption page open (headless Chromium) + caption traffic | next to a game | 0.14 / 0.41 / 0.51 | 1823 | 46 | 0.02 / 356 MB |
| Captions + OBS page + Hy-MT2 translation en to ja, CPU llama-server, interim translations forced on (the policy before P7-I0) | next to a game | 4.00 / 6.29 / 6.72 | 4553 | 52 | 0.03 / 361 MB |
| The same, final lines only (interim translations off) | next to a game | 1.53 / 3.22 / 3.78 | 4487 | 51 | 0.02 / 356 MB |
| The same, interim translations throttled to one per 1.5 s | next to a game | 2.70 / 4.27 / 5.30 | 4500 | 51 | 0.03 / 359 MB |
| Captions + OBS page + Hy-MT2 translation, Vulkan llama-server, interim translations forced on | next to a game | 0.43 / 1.16 / 1.20 | 5223 | 1532 | 0.02 / 358 MB |
| Captions + OBS page + Hy-MT2 translation, server variant `auto` (the default) | next to a game | 0.24 / 0.42 / 0.74 | 5762 | 1832 | 0.02 / 356 MB |

- **The app by itself is light.** Open with speech-to-text off it uses 0.00 cores and 336 MB (WebView2's seven processes included); with English captions running it adds 1414 MB (Nemotron + Parakeet Ultra) and 0.02 cores while nobody speaks (0.03 next to the game), and it touches no GPU memory beyond the WebView's own (49 MB).
- **The new-install default (language auto) costs a gigabyte more** than a set language: 2756 MB against 1765 MB, because it also holds Qwen3-ASR 0.6B and the language-ID engine, and 0.09 cores (p95 0.36). Turning interim results off drops to 1003 MB (one engine less).
- **The OBS caption feed is cheap:** with caption traffic at speaking pace Sigil used 0.14 cores in total and the page's Chromium 0.02 cores and 356 MB (OBS's own browser source costs about the same).
- **Flagged heavy: translation on the CPU.** With the bundled Hy-MT2 on the CPU and interim translations forced on (the default before P7-I0; a request every 300 ms while the speaker talks) the process tree used 4.0 cores on average (llama-server 3.7; p95 6.3, peak 6.7) and 4553 MB next to the game: that competes with the game's own threads on a 6-core PC. Final lines only: 1.53 cores (llama-server 1.43); interims throttled to one per 1.5 s: 2.70 cores. On Vulkan the same translator takes 0.43 cores and 1532 MB of GPU memory (5 % GPU utilization on average); the tree's private memory is not lower than on the CPU (5223 MB). With the server variant left on `auto` (the default) the governor picked a GPU variant here and the tree used 0.24 cores and 1832 MB of GPU memory.
- **What follows.** A PC with a GPU should leave the translator on `auto` (it takes a GPU variant: 0.2-0.4 cores and 1.5-1.8 GB of video memory, which the game has to be able to spare). On a CPU-only PC, or when the card is full, interim translations should be off: that is the finals-only row (1.5 cores on average, bursts of 3-4), and the lightest CPU setting in this table. **Applied by P7-I0:** the `interim` switch of the AI model (`llm`) and Automatic (`auto`) translation slices is now `true`, `false` or `"auto"` (the default); `"auto"` is on except while the bundled server runs on the processor (variant `cpu`, or `auto` resolved to the processor, read from the running server's status), where interim translations are off and the finals are translated as before. A boolean stored in a profile always wins, the checkbox stays in both panels (with a note while the automatic rule holds it off), and nothing else changes: the other engines keep their own switches (Bergamot on, NLLB and DeepL off). Not re-measured: the footprint rows above are the evidence (finals only 1.5 cores, interims 4.0).

## Opt-in and not benchmarked

The heavy GPU models are installable but unmeasured here: Cohere Transcribe (1.9 GB VRAM),
Qwen3-ASR 1.7B (3.5 GB Q8_0, 5.9 GB BF16), Voxtral Mini 4B Realtime, Confucius4-R2T2, Hy-MT2 7B, MiLMMT-4B,
TranslateGemma, Qwen3-TTS 1.7B / 0.6B, Magpie, Breeze, OmniVoice and Fish S2. Their P2 and P6 probe figures (VRAM,
single-clip latency) are in the earlier tables below; none was run under the game-load simulator in P7. With the
`tier` setting on "Best for my PC" (the default) a PC with a GPU is offered the GPU final models once they are
installed (the governor keeps 30 % of the card, at least 2 GB, for Windows, the game and OBS); the CPU tier keeps
the GPU entirely to the game and costs the CPU figures above. Which of the two is better next to VRChat on a
mid-range card was not measured. Two of the models listed are not routable: Voxtral Mini 4B Realtime is installable, but no row of `stt-routing.json` lists it, so no route plans it (only the speech benchmark and the developer probes run it), and Confucius4-R2T2 has no catalog entry (the pinned transcribe.cpp has no architecture for it). The GPU final models a route can offer are Cohere Transcribe and Qwen3-ASR 1.7B.

## For later (findings the campaign could not apply from its own files)

- **Auto mode.** A new install has the language on `auto`, so its final engine is Qwen3-ASR 0.6B for every language, which alone is 1.2 to 4 times
  worse than Parakeet Ultra on the European languages (ru 23.8 against 6.1, pt 12.3 against 3.9). The language-ID engine already names the language
  (105 of 108 right); choosing the final engine from the detected language's own row (Ultra for the 25 Parakeet languages, SenseVoice for ja / zh,
  Qwen3-ASR for ko and the rest) needs a per-utterance final in the router (Rust `speech::router`), not a table change. Until then, setting the language
  is the single biggest accuracy step for a European-language streamer; it also saves 1 GB of RAM (footprint table). Sigil asks for the language rather than waiting for that router change: the setup wizard's STT step (first run only), a hint in the STT panel, and a suggestion after three consistent detections in a session (all explicit, never a silent lock; see [Locking the STT language](../DEVELOPER_GUIDE.md#locking-the-stt-language)).
- **Interim translations on the CPU: applied by P7-I0** (see the footprint section). The bundled translator on the CPU with the old default took 4.0 cores on average (6.7 at the peak) at speaking pace
  next to the game; final lines only take 1.5. The default is now automatic (off on the CPU variant, on elsewhere; a stored choice wins). A throttle of 1.5 s,
  `translation.partials.throttleMs`, only got to 2.7 cores, so it is not a substitute.
- **Loopback-only Korean row.** Other players' captions share the CPU table. On the bleed conditions SenseVoice is the better Korean final (14.4 against 19.2
  over music, babble and the VRChat mix; babble 22.8 against 39.0). A separate `loopback` row set would take it; nothing measured yet says anyone needs it.
- **European translation pairs: applied on the processor tier (MT-Europe, 2026-10-11).** English with French, Portuguese, Dutch and Spanish goes to NLLB beam 4 first while
  the bundled translator runs on the processor (+2.8 to +6.0 chrF2, interval above zero); German and Italian tied and stay on Hy-MT2; the graphics-card tiers are unchanged
  (see "European pairs on a processor-only PC"). Still open: casual caption lines for French, Portuguese, Italian and Dutch (only German and Spanish have them),
  MetricX-24 or a blind A/B on German and Italian, a second processor (a friend's `export.json` of `mt_europe.json`), and the app offering the 0.9 GB NLLB download when a
  processor-only PC translates one of the four pairs (the route does nothing without the model; the offer needs a new UI string).
- **Real VRChat clips and friend exports.** The corpus is synthetic. Drop real clips into the `real` folder of the corpus and rerun the screening spec, and send
  `export.json` of the speed specs from a CPU-only PC and from an RTX 3060 / 4060 (`scripts/speech-bench/score/merge_exports.py` tabulates them).
- **The GPU tier.** Whether the opt-in GPU finals (Cohere, Qwen3-ASR 1.7B) cost the game more than they give, on a mid-range card under VRChat and OBS, and whether
  `tier: auto` should keep proposing them to a streamer, is the open question of the heavy-model benchmarks.
- **Whisper.** The long-tail languages stay on Whisper small / base (quantized q5_1). The campaign measured the fp16 base model only (its quantized siblings are over the
  50 MB download limit): Whisper's rows are an upper bound on cost and a lower bound on quality.
- **Bergamot.** Build the pack (`scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives`, then `-SelfTest`) and add the `bergamot` rows to `mt_screen.json`; it
  would also be the first choice for interim translations (the router's cheapest-first order puts it ahead of the LLM).
- **Four physical cores without SMT.** `cpuTwoPassMinCores` stays at 6; the 4C slice (4 cores with SMT) held two passes at 0.5 cores, a 4-core PC without SMT was not
  emulated.
- **Edge TTS** was not timed (no backend in the runner, no cloud request from the campaign): speak a line with Edge in Settings, Text-to-speech.
- **In VR.** Everything here ran outside VR with the load simulator; the footprint rows are what the app costs next to a CPU-bound stand-in for the game, not a
  measurement inside SteamVR.

---

# Earlier probe numbers (P2 speech engines, P4 translation)

Dev rig: Ryzen 7 7800X3D (8 cores), RTX 4090 24 GB (driver 616.56), Windows 11. Unless noted, timings come from the debug `sigil.exe` with the release runtime DLLs (sherpa-onnx 1.13.8, transcribe.cpp 0.2.4), median of 3 runs, 2026-09-29. RTF = processing time / audio time (lower is faster; below 1 is faster than real time). Raw reports live in the git-ignored `bench-results/`. The CPU STT table below was measured before the campaign (single clips, the debug build for the Rust side); where it disagrees with the campaign above, the campaign's numbers are the current ones.

## Speech-to-text on the CPU (sherpa-onnx)

`Sigil.exe --headless engine-probe sherpa <catalog id> <clip> -- repeat=3 lang=<code> threads=<n>`
(P2-B; 55 runs, all deterministic; `bench-results/sherpa/p2b-20260929-161048/summary.md`).

| Model | Threads | RTF | Notes |
|---|---|---|---|
| Nemotron 3.5 streaming 560 ms (default live model) | 1 | 0.30–0.35 (en, ja, zh, ru, de) | first partial 1.1–1.7 s into speech; 2 threads: 0.22 at 2.5 cores |
| Nemotron 3.5 streaming 160 ms | 1 | 0.80–0.87 | first partial 0.6–1.4 s; too heavy for 6-core PCs |
| Parakeet Ultra int8 / Parakeet TDT v3 int8 | 2 | 0.063–0.075 | en / de / es / fr / ru; identical transcripts except one Spanish word each way |
| Qwen3-ASR 0.6B int8 | 2 | 0.16–0.27 | en / ja / ko / ru / zh |
| SenseVoice small int8 | 2 | 0.029–0.041 | en / ja / ko / yue / zh, with lang / emotion / event tags |
| Moonshine base-en int8 | 2 / 1 | 0.017 / 0.026 | English |
| Whisper-base language ID | 2 | 0.026–0.043 | named en / ja / zh / ru / de / es correctly |

Finals through the speech pipeline (real Silero VAD), time from the end of speech to the final:
Moonshine 132 ms, Nemotron 161 ms, SenseVoice 190 ms, Parakeet 295 ms, Qwen3-ASR 1.2 s.

Utterance engines on 2 threads spin 20–75 ms of CPU per second for about 1 s after each decode
(ONNX Runtime's spin-wait cannot be turned off through sherpa's C API); streaming stays on 1 thread
and does not spin.

## Speech-to-text on the GPU (transcribe.cpp, CUDA 13)

`cargo test speech::engines::transcribe -- --ignored --test-threads=1` with the P2-D model store
(P2-D). VRAM is the process VRAM including the ~0.38 GB CUDA context.

| Model | VRAM | Result |
|---|---|---|
| Qwen3-ASR 1.7B Q8_0 | 3470–3494 MiB | ja-short in 245 ms; a 28 s clip split and decoded in 600 ms; RTF 0.023 |
| Qwen3-ASR 1.7B BF16 | 5914 MiB | jfk in 226 ms |
| Cohere Transcribe Q4_K_M | 1950 MiB | jfk in 293–306 ms; refuses ja; needs a language |
| Granite Speech 5.0 TurboCTC Q8_0 | 890 MiB | jfk in 56–68 ms; lower-case, no punctuation (never a caption final) |
| Voxtral Mini 4B Realtime Q4_K_M | ~4100 MiB | 9–10 streaming partials per clip; append-only partials never retract |

**Two-pass (P2 exit):** Japanese, Qwen3-ASR 1.7B Q8_0 final through the real two-pass queue: 198
and 204 ms after the utterance closed on an 8 s clip (target ≤ 800 ms). 380 MiB of CUDA context stay
after the last model is dropped. ORT's CUDA execution provider and ggml-cuda share one CUDA context
(canary before 509 ms, after 4 ms).

## Smart Turn v3.2

About 31 ms per prediction on 2 threads (model), window plus log-mel features 1.6 ms in a release
build; probabilities match onnxruntime + transformers within ±0.02.

## Text-to-speech on the CPU

`cargo test local_tts -- --include-ignored --test-threads=1` (P2-H; re-run at P2 integration on an
otherwise idle PC). Kokoro v1.0 fp32, 4 threads, sherpa-onnx and Sigil's ORT interleaved sentence by
sentence (3.9–5.2 s sentences):

| Kokoro runtime | Median RTF | CPU s per audio s | Notes |
|---|---|---|---|
| Sigil's ORT with the CPU memory arena (default since P2-I0) | 0.294 | 0.90–1.00 | new and repeated sentence lengths alike |
| inside sherpa-onnx | 0.315–0.319 | 1.17–1.26 | |
| Sigil's ORT without the arena (P2-H) | 0.29–0.40 on new lengths, 0.23–0.24 repeated | 0.93–1.09 | ort rc.13 default |

- First audio of a long first sentence with the first-clause split: 656–679 ms (its opening clause
  is about 2 s of audio); load plus warm-up 1.4 s (sherpa) / 2.8 s (Sigil's ORT, once per model).
- Kokoro int8 through sherpa: RTF about 1.0–1.3 (not catalogued).
- Piper ru (irina medium): RTF 0.04 idle, 0.15–0.16 under load. Cancel takes effect in about 0.6 s.
- Intelligibility (SenseVoice transcripts): English and Mandarin exact; Japanese kana correct,
  kanji read with Mandarin readings (sherpa-onnx 1.13.8 frontend).

## OpenAI-compatible TTS streaming

Against `scripts/neural-sidecar/tts_sidecar.py --chunk-delay 0.2` (P2-I): first audio after about
205 ms, full response after about 2.84 s, for both PCM and WAV (`openai_speech_live_server_streams`).

## Machine translation

`Sigil.exe --headless mt-bench <spec.json> --out <dir>` ([`scripts/mt-bench/README.md`](../../scripts/mt-bench/README.md)),
corpus chrF2 (sacrebleu 2.6.0 defaults: character order 6, β 2, whitespace removed; the Rust score
equals sacrebleu's to 1e-14) and wall time per line. Corpus: `scripts/mt-bench/corpus.sample.jsonl`,
240 casual lines (24 per direction) with a single reference each, written for the smoke test; P7's
campaign replaces it with consented real caption logs. chrF is character-based with one reference,
so Chinese and Japanese targets score low for every engine. Dev rig, debug build (dependencies at
opt-level 3), measured on 2026-09-30 while other builds were running, so latencies are pessimistic.

NLLB-200 distilled 600M, int8 (`mt/nllb-200-distilled-600m-int8`, CPU, 2 threads):

| Direction | chrF greedy | chrF beam 4 | p50 / p95 ms greedy | p50 / p95 ms beam 4 |
|---|---:|---:|---:|---:|
| en → ja | 25.4 | 27.4 | 1006 / 1337 | 2912 / 4040 |
| ja → en | 58.3 | 59.2 | 1102 / 1637 | 2685 / 5181 |
| en → de | 66.4 | 67.6 | 1141 / 1768 | 2840 / 4016 |
| de → en | 72.4 | 71.5 | 1109 / 1483 | 2739 / 3993 |
| en → es | 69.2 | 73.5 | 1147 / 1608 | 2868 / 3960 |
| es → en | 72.8 | 73.7 | 1043 / 1493 | 2400 / 3586 |
| en → zh | 20.8 | 23.0 | 1173 / 1707 | 2442 / 3429 |
| zh → en | 55.5 | 55.5 | 1190 / 1749 | 2517 / 4157 |
| en → ko | 27.5 | 29.3 | 970 / 1297 | 2232 / 2931 |
| ko → en | 55.5 | 54.8 | 1001 / 1580 | 2371 / 3301 |
| **All (corpus)** | **58.6** | **59.6** | | |

Most of NLLB's per-line cost is the 256k-token output layer at every decoding step, which is why
the `nllb` slice has partials off by default and the router uses NLLB for the long tail.

Hy-MT2 1.8B Q4_K_M (`mt/hy-mt2-1.8b-q4_k_m`) on the pinned llama-server b11146 Vulkan build, RTX 4090
(`--device Vulkan0 -ngl 999 -c 4096 -np 1 -fa auto --poll 0`, the app's flags), mt-bench `openai` engine
with the `hy-mt2` prompt and temperature 0.2, same corpus, P4 final gate (2026-09-30):

| Direction | chrF | p50 / p95 ms |
|---|---:|---:|
| en → ja | 44.0 | 70 / 107 |
| ja → en | 54.1 | 57 / 78 |
| en → de | 64.9 | 70 / 106 |
| de → en | 68.8 | 53 / 71 |
| en → es | 66.9 | 67 / 88 |
| es → en | 68.5 | 50 / 68 |
| en → zh | 44.0 | 42 / 54 |
| zh → en | 63.9 | 50 / 70 |
| en → ko | 35.7 | 74 / 109 |
| ko → en | 54.6 | 58 / 88 |
| **All (corpus)** | **59.9** | |

Against NLLB-600M greedy, Hy-MT2 scores much higher on the Asian targets (en → ja 44.0 vs 25.4, en → zh
44.0 vs 20.8, en → ko 35.7 vs 27.5). It scores a little lower from ja/de/es into English and on en → de,
and it answers in about 50–75 ms instead of about 1 s. Three lines out of 240 took 1.1–1.7 s (en → ja,
en → de, zh → en maxima; not investigated, possibly a one-time GPU warm-up); p95 stays under 110 ms. One smoke
corpus with one reference per line cannot rank close engines, so these numbers are not a default flip
(P7 decides with real caption logs).

Still to record (manual, see [RELEASE_AND_ENGINEERING.md → P4 exit matrix](../RELEASE_AND_ENGINEERING.md#p4-exit-matrix-translation-stack)):

- **Firefox Translations (Bergamot)**: needs the locally built `sigil-bergamot.exe` pack; target
  en ↔ ja p95 under 150 ms per line (`build.ps1 -SelfTest -Models <dir>`). Until mt-bench has a
  `bergamot` engine, score its output with the `hypotheses` engine.

## Reproducing

- The P7-A campaign: [`scripts/speech-bench/campaign/README.md`](../../scripts/speech-bench/campaign/README.md) (specs, drivers, the own-mic ranking `own_mic.py`, the footprint harness). One run: `pwsh scripts/speech-bench/campaign/lock-run.ps1 -Name <tag> -Spec scripts/speech-bench/campaign/specs/<spec>.json -Out bench-results/speech/<tag> -Cpu -Affinity FFF`; score with `score/score_stt.py --items`, then `campaign/own_mic.py --items .../items.jsonl ...`. A friend runs the same spec and sends `export.json` (`score/merge_exports.py` tabulates the machines).
- MT-Europe (European pairs, NLLB beam 4 against Hy-MT2 on the processor): `python scripts/speech-bench/corpus/build_text_sets.py` once (it adds `flores200-eu.jsonl` and leaves the other sets byte for byte as they were), then `pwsh scripts/speech-bench/campaign/lock-run.ps1 -Name mt-europe -Spec scripts/speech-bench/campaign/specs/mt_europe.json -Out bench-results/speech/mt-europe -Cpu -Affinity FFF` and the two `mtspeed_nllb-beam4_*` specs the same way; `python scripts/speech-bench/campaign/mt_europe_table.py --report .../report.json --set <each set>` prints the pair, direction and per-set tables with their paired intervals. A friend with a processor-only PC sends the `export.json` of `mt_europe.json`.
- sherpa engines: `SIGIL_APP_DATA_DIR=<store with the models> SIGIL_TEST_MODELS_DIR=<dev cache>
  cargo test --manifest-path src-tauri/Cargo.toml sherpa -- --include-ignored --test-threads=1`;
  the probe matrix script lives in `scripts/runtime-packs/sherpa/`.
- GPU tier: `SIGIL_APP_DATA_DIR=<store> SIGIL_TEST_MODELS_DIR=<dev cache> SIGIL_STT_WORKER_EXE=<debug
  sigil.exe copy with the ORT provider DLLs> cargo test speech::engines::transcribe -- --ignored
  --test-threads=1`.
- Whisper through the pipeline: `SIGIL_TEST_MODELS_DIR=<folder with ggml-tiny.en.bin> cargo test
  real_model -- --ignored`.
- Local TTS: `SIGIL_APP_DATA_DIR=<store> SIGIL_TEST_MODELS_DIR=<dev cache> cargo test local_tts --
  --include-ignored --test-threads=1 --nocapture`.
- Machine translation: `SIGIL_APP_DATA_DIR=<store with mt/nllb-200-distilled-600m-int8>
  sigil.exe --headless mt-bench scripts/mt-bench/spec.example.json --out <dir>` (engines `nllb`,
  `openai`, `hypotheses`); NLLB parity with the Python reference: `cargo test mt_local --
  --ignored --test-threads=1` with the same `SIGIL_APP_DATA_DIR`.
