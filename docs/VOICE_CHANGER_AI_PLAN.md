# Sigil Local AI Voice-Changer — Plan of Record

> **SUPERSEDED (2026-09-28).** This plan is replaced by the voice-changer parts of
> [SPEECH_STACK_IMPLEMENTATION_PLAN.md](SPEECH_STACK_IMPLEMENTATION_PLAN.md) (the Sigil Voice
> Engine: "VC best engine", decisions, phases P0/P3/P5) and the research in
> [SPEECH_STACK_RESEARCH_2026-09.md](SPEECH_STACK_RESEARCH_2026-09.md) §4. It is kept for its
> history and rationale only; where the two disagree, the implementation plan wins. The P0
> real-time fixes (latency budget, park/unpark wake, MMCSS on callbacks, aux back-pressure,
> Rust-side persistence, EP reporting) are shipped; see
> [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md#voice-changer).

> **Status (original):** PLAN ONLY — not yet implemented. Decisions locked 2026-07-08.
> This is the agreed design + phased roadmap for completing the local AI voice
> changer. Nothing here has been coded yet. Untracked working doc — move,
> commit, or delete as you like.
>
> **REVISION 2026-07-09 (CPU research):** The CPU-default engine changed from
> **LLVC → the RVC "CPU profile."** LLVC is **any-to-one** (each voice = a ~3-day
> GPU training run, no premade voice library, 16 kHz ceiling, no production ONNX
> streaming export), so it cannot be the default in a BYO-ONNX product. The
> best quality-per-compute CPU answer is **one engine (RVC), two profiles**:
> CPU = `fcpe` F0 + int8-quantized content encoder + fp32 vocoder; GPU = fp16 +
> `rmvpe`. This also *simplifies* the architecture (CPU↔GPU is an
> execution-provider + precision swap on the same engine, not a second backend).
> LLVC is demoted to an optional future niche; **QuickVC** (MIT) is the optional
> "lightweight mode." DDSP-SVC is **MIT** (earlier "AGPL" note was wrong) but
> still excluded (F0-dependent/singing, partial ONNX).
>
> **REVISION 2026-07-13 (VRChat coexistence):** Primary audience is **VRChat
> streamers** running Sigil + OBS + game on one machine. **Default design goal =
> compute-preserving realtime** via the **Streamer** preset (RVC CPU profile).
> **Max quality is first-class** via a **Quality** preset — same RVC engine,
> fidelity knobs turned up, with honest headroom warnings. Ship three named
> presets (Streamer / Low latency / Quality). Phase order rephased: **Phase 1 =
> RVC CPU MVP** (proves the product for streamers); **Phase 2 = GPU + Low
> latency + Quality** presets. Always-on coexistence wins: ORT spinning off,
> lock-free output ring, silence-front VAD, worker BelowNormal priority.

## Decisions locked (2026-07-08, revised 2026-07-13)

| Decision | Choice | Consequence |
|---|---|---|
| **Formats, first-party** | **RVC (v1/v2/Nono) + plain ONNX** | No Beatrice, no copyleft families in the binary; LLVC deferred (not a shipped mode) |
| **Beatrice v2** | **Dropped** — per-edition voice EULAs + native (non-ONNX) lib | CPU-default role goes to the **RVC Streamer profile**, not LLVC |
| **Model input** | **BYO ONNX** + auto-downloaded base models (HuBERT/RMVPE/FCPE) | No bundled Python/PyTorch converter; users export `.pth→.onnx` via documented RVC-WebUI step |
| **Default engine** | **CPU-first = RVC "Streamer" preset** | One engine (RVC). Streamer = `fcpe` F0 + int8 content encoder (VNNI-gated) + fp32 vocoder; Low latency / Quality = GPU opt-in on same `RvcModel` |
| **Quality path** | **First-class Quality preset** | Same RVC engine; raises Extra, index, F0 (rmvpe), VRAM — not a second backend; must stay realtime or show RTF failure |
| **Copyleft (DDSP-SVC / so-vits-svc / Seed-VC)** | **Out of scope for now** (AGPL/GPL) | Not in first-party binary; could return later only as optional user-installed sidecars |

**Audience constraint that drives every decision:** VRChat streamers whose GPU
is already rendering a game. Low VRAM, low GPU%, low CPU, and low latency are
hard requirements, not tuning afterthoughts. This biases us toward a **CPU
default (RVC Streamer profile)**, **GPU RVC as the opt-in Low latency / Quality
path** on the same engine, and an aggressive `ort` optimization layer
(IO-binding + CUDA graph + capped arena + no thread spinning + silence-skip VAD).
**Max quality is offered** via the Quality preset for users with headroom — not
the default, not buried in Advanced-only sliders.

**Hard budget (all presets):** stay realtime (`process_ms < chunk_ms`) or show a
clear RTF failure. **Streamer** acceptance bench is the three-app scene (Sigil +
OBS + busy VRChat world), not VC alone. On **4-core + software x264**, treat as
marginal; UI/docs push OBS **NVENC/QSV/AMF** hardware encoding.

---

## 1. Recommended architecture

### 1.1 Why today's code can't just be extended in place

`OnnxVoiceBridge` (`src-tauri/src/services/neural_audio/onnx_voice.rs:298`) is a
**single-tensor black box**: it binds only `inputs[0]`/`outputs[0]` and matches
*element counts*, not sample rates or feature semantics (verified). Real neural
VC (RVC) is a **multi-graph, multi-input pipeline** with cross-chunk state
(rolling context + SOLA crossfade). So `OnnxVoiceBridge` is **kept as-is** for
the "plain ONNX" mode, and a **new engine layer is built alongside it** — not a
rewrite.

Today "pitch" means the DSP phase-vocoder in `ClassicPhaseVocoderProcessor`
applied *after* conversion. Real VC conditions an **F0 contour into the model**.
So we add a genuine F0 stage that feeds the model and demote the phase-vocoder
to an optional post-effect.

### 1.2 Two new traits (`src-tauri/src/services/neural_audio/vc/traits.rs`)

```rust
/// F0 per 10 ms frame (16 kHz / hop-160 grid), Hz, 0.0 = unvoiced.
pub trait PitchExtractor: Send {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String>;
    fn hop_samples(&self) -> usize { 160 }
    fn sample_rate(&self) -> u32 { 16_000 }
}

/// A full voice-conversion backend. Owns its own sessions/state.
/// Operates on a rolling context window; returns model-SR audio for the NEW region.
pub trait VoiceConversionModel: Send {
    fn convert(&mut self, ctx: &ConversionCtx) -> Result<Vec<f32>, String>;
    fn input_sample_rate(&self) -> u32;   // 16000 for RVC content stage
    fn output_sample_rate(&self) -> u32;  // 32k/40k/48k baked into model
    fn wants_f0(&self) -> bool;           // false => f0-less models
    fn reset(&mut self);                  // clear rolling state on (re)start
}

pub struct ConversionCtx<'a> {
    pub mono_16k: &'a [f32],   // rolling buffer: extra-context + new block
    pub new_frames: usize,     // trailing samples that are "new" this tick
    pub pitch_shift_semitones: f32,
    pub formant: f32,
    pub index_rate: f32,       // RVC retrieval blend, 0 = off
    pub protect: f32,          // RVC unvoiced protect
    pub speaker_id: i64,
}
```

The concrete `PitchExtractor` is constructed in the loop and injected so the
UI's F0 dropdown works across models; RVC calls it internally to align F0 to its
own content frames.

### 1.3 The missing piece: a realtime stream harness (`vc/stream.rs`)

Neural VC cannot be "one chunk in → one chunk out" like the classic path. It
needs a **rolling buffer + SOLA stitcher**:

```rust
pub struct VcStreamHarness {
    ring_16k: Vec<f32>,     // rolling mono @16k: extra_context + block
    extra_samples: usize,   // "Extra" context (w-okada), user slider
    sola: SolaStitcher,     // crossfade join, autocorr search
    in_resampler: FftFixed, // device_in_SR -> 16k (rubato)
    out_resampler: FftFixed,// model_SR -> device_out_SR (rubato)
}
```

This is the **biggest single engineering item** and where most bugs will live
(frame-grid misalignment *sounds* broken before it sounds right). It wraps any
`VoiceConversionModel` and emits device-rate interleaved audio into the existing
output playback path (lock-free ring after Phase 0).

### 1.4 How it slots into the existing hot path

`run_voice_changer_loop` (`voice_changer.rs:930`) has a 3-arm mode branch at
`:1223` (onnx / sidecar / classic — verified). Add a fourth arm and route
neural models through the harness instead of `VoiceChunkPipeline`:

```
work_chunk (interleaved, device rate, ch_in)
  → [existing] input_gain + mix_aux_into_chunk        (voice_changer.rs:1212-1220)
  → NEW: VcStreamHarness.push_and_convert(work_chunk)  // downmix→16k, roll, F0, model, SOLA, →device rate
  → processed (interleaved, device rate, ch_out)
  → [existing] monitor / output_gain / EMA metric / lock-free output ring push
```

For neural modes the classic post-chain becomes **optional** (pitch/formant are
conditioned into the model). `VoiceChangerMode` (`voice_changer.rs:129`) gains
an `Rvc` variant (keep `Onnx`, `Sidecar`, `Classic`). Wire the new arm through
the existing registration checklist: enum → params struct (`:265`) →
`set_voice_changer_params` (`:681`) → `VoiceChangerState` arcs (`:483`) →
`start_voice_changer` clones (`:868`) + validation guard (`:838`) → loop
instantiation (`:1149`).

### 1.5 Shared ONNX/EP infra to factor out first (cheap, unblocks everything)

`session_builder_with_ep` + `apply_vc_ep_hint` + `format_onnx_ep_error` live in
`onnx_voice.rs:15-43` (a near-duplicate exists in `neural_onnx_tts.rs`). Extract
to **`neural_audio/ort_session.rs`** and extend the builder with the §4
optimization levers (memory limit, arena strategy, spinning off, intra-threads,
CUDA graph, fp16 flags, shared environment). Both TTS and every VC session build
through it. **Small refactor, high leverage.**

> **Status (implemented):** the extraction landed. `apply_vc_ep_hint` and the
> env-var EP hints (`ORT_CUDA_DEVICE_ID`) no longer exist. VC sessions build
> through `ort_session::session_builder_for_vc(hint, &VcSessionOptions)`, which
> passes the GPU device index and CUDA VRAM cap as EP options (plus Level3,
> spinning off, and single-thread intra/inter for the Streamer profile);
> `session_builder_with_ep(hint)` is the default-options wrapper and
> `format_onnx_ep_error` stays in `ort_session.rs`.

### 1.6 Model-on-disk layout + download/caching (BYO ONNX)

Reuse the `config.json` manifest pattern from neural TTS
(`neural_onnx_tts.rs:load_from_local_dir`) + a w-okada-style slot concept. Under
`app_data_dir/voice-changer/models/<slot>/`:

```
manifest.json   { format:"rvc", version:"v2", sample_rate:40000, if_f0:true,
                  emb_channels:768, emb_layer:12, use_final_proj:false,
                  generator:"gen.onnx", index:"added.index", speaker_id:0 }
gen.onnx        (RVC generator, user-exported fp16 or fp32)
added.index     (optional FAISS-style retrieval; may be absent)
```

Shared base models (HuBERT/ContentVec, RMVPE, FCPE) live once under
`app_data_dir/voice-changer/base/`, downloaded on demand by **generalizing the
existing `whisper.rs:256` `ensure_dependencies` streaming downloader** (`.partial`
rename, cancel/epoch, size + HTML-sniff validation) — no new hf-hub dep.
**No bundled weights** (consent + licensing). `.pth→.onnx` conversion is
out-of-process: document the RVC-WebUI export step; Sigil loads only `.onnx`.

### 1.7 Frontend / schema / inspector changes

Non-persisted service stays non-persisted (no `patchService`).

- **`schema.ts`** `VoiceChangerStateSchema`: add `modelFormat:
  z.enum(['classic','onnx','rvc','sidecar'])`, `preset:
  z.enum(['streamer','low_latency','quality']).default('streamer')`,
  `pitchExtractor: z.enum(['fcpe','rmvpe','crepe','dio']).default('fcpe')`,
  `indexPath`, `indexRate` (0–1), `protect` (0–0.5), `extraContextMs`,
  `crossfadeMs`, `speakerId`, `fp16`, plus GPU-headroom controls `vramCapMb`,
  `cpuOffload`, `yieldCpu`, `gpuDeviceIndex`.
- **`index.ts`** `VoiceChangerParamsRust` + `syncParams`: add snake_case fields;
  extend the hydration clamp block.
- **`inspector_voice_changer.tsx`**: model-format `InputSelect`
  (structural → restart), **preset** `InputSelect` (Streamer / Low latency /
  Quality — top-level, not Advanced-only), F0-method `InputSelect`, RVC `.index`
  file picker (reuse the existing `open()` dialog), index-rate / protect / extra
  / crossfade sliders, VRChat headroom controls, and a `chunk_ms` readout next
  to the existing `process_ms` metric poll so users self-tune to stay realtime
  (`process_ms < chunk_ms`; red when RTF≥1). Quality preset shows a headroom
  banner. Gate visibility on `modelFormat` like the existing `isOnnx`/`isSidecar`
  conditionals. Reuse unused `quick_presets` i18n namespace.

### 1.8 Named presets (Streamer default; Quality first-class)

Three presets, **one `RvcModel`**. Presets change EP / F0 / precision / Extra /
index / VRAM — not the engine.

| Preset | EP | F0 | Precision | Extra / index | ORT | Intent |
|---|---|---|---|---|---|---|
| **Streamer** (default) | CPU | fcpe | int8 enc if VNNI else fp32; fp32 vocoder | Minimal Extra; `index_rate=0` | `allow_spinning=0`, intra/inter=1, Level3 | OBS+VRChat headroom |
| **Low latency** | CUDA/DML | rmvpe | fp16 HuBERT+gen | Small Extra; index off | Same spin/thread caps + VRAM 512–768 MB | Latency when GPU free |
| **Quality** | CUDA/DML preferred (CPU + warning OK) | rmvpe (crepe under Advanced) | fp16 default; optional fp32 gen | **Higher Extra**; **index on** when `.index` present | Spinning still off; higher VRAM cap (1–1.5 GB) or warn-to-uncap; intra≤2 on GPU only | Max RVC fidelity while staying realtime |

**Invariant:** Quality raises fidelity knobs; it never re-enables ORT busy-spin
or int8-on-vocoder.

**Always-on free wins** (all engines / all presets, including today's plain ONNX
after Phase 0):

1. ORT **`allow_spinning=0`** + Streamer **intra=1 / inter=1** (fixes idle 100% CPU)
2. Shared capped arena / single `ort::init`
3. Lock-free SPSC output (replace `Mutex<OutputPlayback>` in cpal callback — `voice_changer.rs` ~1070)
4. Silence-front VAD: skip HuBERT+F0+gen on silent frames (today's buffer truncate is **not** this)
5. Worker thread **BelowNormal** priority so game/OBS win scheduling
6. `process_ms` + **`chunk_ms`** side-by-side; red when RTF≥1

**GPU / Quality honesty in UX:** Low latency = latency + F0 freedom + free CPU
for OBS. Quality = best timbre/pitch/retrieval, may contend with the game for
GPU/VRAM — banner + live RTF. Optional second GPU index (today CUDA is
hard-coded `device=0`).

---

## 2. Format support matrix

| Format | Approach | ONNX path | Effort | Priority | Licensing |
|---|---|---|---|---|---|
| **RVC v2** (768-dim, HuBERT L12, no final_proj) | New `RvcModel` = HuBERT ONNX → (opt FAISS) → F0 → generator ONNX (6-input) → SOLA | User exports `.pth→.onnx` (RVC-WebUI); Sigil loads generator + hubert + rmvpe/fcpe onnx | **XL** | **P0** | Runtime MIT; user voice models are BYO, never redistributed |
| **RVC v1** (256-dim, HuBERT L9 + final_proj) | Same engine; differs only in emb dim / layer / final_proj (from manifest) | Same | **S** on v2 | P1 | Same |
| **RVC Nono** (f0-less, 3-input) | Same engine; skip F0, 3-input tensor layout | Same | **S** | P1 | Same |
| **RVC "Streamer" (CPU profile)** | **The default** — same `RvcModel`, CPU EP: `fcpe` F0 + int8 content encoder + fp32 vocoder | Same as RVC | **incremental on RVC** | **P0 (default)** | Same as RVC |
| **Plain ONNX** (single-tensor identity/effect) | **Already works** — `OnnxVoiceBridge` unchanged | Existing | **0 (done)** | User file |
| **QuickVC** (VITS + HuBERT-Soft + iSTFT) | Optional "lightweight mode" secondary; sub-100 ms on CPU | Self-authored 2-graph export | **M** | P2 (deferrable) | **MIT** — any-to-many (must train per target set; no BYO voice library) |
| **LLVC** (single-target, causal) | **Deferred** — not a shipped mode | **Blocked** — no production ONNX streaming export; any-to-one; per-voice GPU training; 16 kHz | **L (blocked)** | Deferred | MIT (not viable as BYO default — see revision notes) |

**Explicitly excluded (this round):** Beatrice v2 (per-edition voice EULAs +
native lib), so-vits-svc (AGPL), Seed-VC (GPL); DDSP-SVC (**MIT**, but
F0-dependent/singing-oriented + partial ONNX — reference bar only), FreeVC/kNN-VC
(MIT but WavLM-Large makes them CPU-marginal-to-no). Copyleft families, if ever
wanted, return only as optional **user-installed sidecars**, never in the
first-party binary.

**Shipping order:** RVC v2 harness on **CPU Streamer profile** (Phase 1) → GPU
**Low latency + Quality** presets (Phase 2) → RVC v1/Nono → F0 menu + weak-CPU
escape → optional QuickVC only if 4-core Streamer still fails after NVENC guidance.

---

## 3. Pitch-extractor matrix

All must emit F0 on the **16 kHz / hop-160 (10 ms) grid** to align with content
features — a mismatch here is the #1 cause of garbled RVC output.

| Method | Approach | GPU/CPU cost | ONNX I/O | Priority |
|---|---|---|---|---|
| **FCPE** | LynxNet depth-sep CNN; log-mel→F0 | Cheapest neural; trivial on CPU | mel/waveform → F0 `[1,T]` + periodicity | **P0 — CPU DEFAULT** (Streamer preset) |
| **RMVPE** | 128-mel → salience `[1,T,360]`, argmax+local decode | Cheap on GPU; heavy on CPU | mel/waveform → `[1,T,360]` | **P0 — GPU DEFAULT** (Low latency / Quality) |
| **CREPE-tiny** | 1024-sample frames → `[T,360]` | Moderate | waveform → `[T,360]` | P2 (Quality / Advanced only) |
| **dio+StoneMask** (WORLD) | CPU DSP (BSD) | Realtime, jittery, zero GPU | none (Rust WORLD binding) | **P1 — weak-CPU escape hatch** |
| CREPE-full / harvest | 22M params / WORLD C++ | Heavy / not realtime | — | Offline "render" mode only |
| pm (Praat) / pyin | autocorr / pYIN | cheap / moderate | — | **Skip** (GPL) |

**Defaults by preset:** **Streamer (CPU):** `fcpe`; gray out `rmvpe`/`crepe` on
CPU EP. **Low latency / Quality (GPU):** `rmvpe`. **Weak CPU escape:** WORLD
**dio+StoneMask**. Post-process every extractor with a **confidence gate +
3–5-frame median smoother** (kills octave jumps at ~zero cost). One shared
`PitchExtractor` trait; ONNX impls share a persistent session via `ort_session.rs`.

---

## 4. Optimization plan ("incredibly optimized")

The heart of the "save resources for VRChat" ask. All levers are exposed by
`ort` v2's SessionBuilder/EP builders. **Exact `ort` v2.0.0-rc.11 method names
must be confirmed at implementation time** (rc API churn) — the underlying
onnxruntime capabilities are stable; the Rust binding surface is the risk.

### 4.1 Session/EP defaults (`ort_session.rs`)
- **One shared `ort::init()` env**; all sessions (HuBERT / F0 / generator) draw
  from **one capped arena**, not three.
- **VRAM cap:** Streamer/Low latency: 512–768 MB; Quality: 1–1.5 GB (or
  warn-to-uncap). Arena-extend strategy `SameAsRequested` (stops power-of-two
  doubling).
- **Conv workspace:** disable max workspace + heuristic algo search (sheds
  multi-GB peak and the slow exhaustive first run).
- **Graph opt:** Level3; cache the optimized model to skip re-opt on launch.

### 4.2 Per-chunk hot path
- **IO binding + CUDA graph:** pre-allocate device I/O tensors once, reuse the
  same addresses every chunk, capture on first `Run()`, replay after → zero
  host↔device copies, zero per-chunk alloc. **Requires fixed shapes** ⇒ fixed
  chunk + fixed Extra while streaming (re-capture on change).
- Kills the current per-chunk allocation storm (`adapt_*` Vecs,
  `Array::from_shape_vec`, `work_chunk.clone()` on error — `onnx_voice.rs:404-457`).

### 4.3 Precision
- **fp16** for HuBERT + generator on CUDA/DirectML (~2× on tensor cores, ~½
  VRAM, negligible quality loss). Keep RMVPE fp32 if it NaNs. TF32 for any fp32
  residual on Ampere+. Quality preset may offer optional fp32 generator for ceiling.
- **No int8 on GPU** (CUDA/DML can't run QDQ).
- **int8 dynamic quantization: content encoder ONLY** on CPU, gated on
  **AVX512-VNNI (Intel) / Zen4+** — near-lossless + ~2–3× there; ship fp32
  encoder on non-VNNI silicon. **Never quantize the HiFi-GAN generator** —
  dynamic int8 on conv/vocoder ops runs 2×–10× *slower* and adds metallic/buzz
  artifacts. **No fp16/bf16 on CPU** (Cast overhead makes it same-or-slower).

### 4.4 Threading (co-existence, not throughput)
- Sequential execution; **default intra-threads=1, inter-threads=1** (intra=2
  only on GPU Quality if RTF>0.7 and user opts in).
- **Disable spinning** (`allow_spinning=0` on intra + inter) so idle ORT workers
  **sleep** instead of burning cores VRChat's render thread needs — a top
  w-okada pain point. **Invariant across all presets including Quality.**

### 4.5 Algorithmic levers
- **Minimize Extra/context** on Streamer — attention ~O(T²); smallest stable value
  (2048–4096 samp) is the biggest latency lever. Quality raises Extra; slider
  with a quadratic-cost warning.
- **Silence-front VAD:** skip HuBERT+F0+gen entirely on silent frames (reuse the
  RMS trimming concept in `OutputPlayback::push_interleaved`). Biggest average-load win.
- **FAISS:** `index_rate=0` on Streamer and Low latency; Quality turns index on
  when `.index` present (k=8 default, k=1 when latency-bound).
- Prefer **40 kHz models over 48 kHz**; fixed-ratio rubato FFT resamplers with
  precomputed kernels on the neural path.
- **Warm up** each session with a dummy chunk at start so CUDA-graph capture /
  cuDNN algo pick happens before going live, not mid-stream.

### 4.6 "Leave headroom for VRChat" (first-class UX)
1. **Preset selector** — Streamer (default) / Low latency / Quality (top-level).
2. **VRAM cap** slider → memory limit per preset defaults above.
3. **"Yield CPU"** toggle → reinforces spinning off + intra-threads 1.
4. **"CPU offload when GPU busy"** → forces CPU EP with bounded threads (RVC
   Streamer profile).

Plus: keep the existing EP selector (CUDA/DirectML/CPU); add **GPU device index**
so a second GPU can take VC; worker thread **BelowNormal** priority; keep the
`process_ms < chunk_ms` readout visible as the self-tuning guide. Quality preset
shows a headroom banner when GPU contention is likely.

### 4.7 Fix the pre-existing RT-safety wart
The cpal output callback locks a `std::sync::Mutex<OutputPlayback>`
(`voice_changer.rs:1074`, verified) **inside a realtime callback** — contention
risk with the worker's `push_interleaved`. Replace with a lock-free SPSC ring
(mirror the rtrb input side). Not neural-specific but must land before adding
backends that lengthen the worker's critical section. **Medium, do it in Phase 0.**

### 4.8 CPU-profile specifics (Streamer preset — from CPU research 2026-07-09)
The knee of quality-vs-compute on CPU is the **F0 method** and **selective
quantization**:
- **F0 is the #1 CPU knob.** Default **`fcpe`** (near-rmvpe quality at a fraction
  of the cost). `dio+stonemask` = cheapest weak-CPU fallback (occasional octave
  wobble). **Label `rmvpe`/`crepe` as GPU-only** on CPU EP — they eat the cores
  OBS/VRChat need.
- **int8 dynamic quantization: content encoder ONLY**, and **gate it on
  AVX512-VNNI (Intel) / Zen4+** — near-lossless + ~2–3× there, negligible/worse
  elsewhere so ship fp32 encoder on non-VNNI silicon. **Never quantize the
  HiFi-GAN generator** — dynamic int8 on conv/vocoder ops is documented to run
  2×–10× *slower* and adds metallic/buzz artifacts.
- **Content encoder:** ContentVec (256-dim v1 is a legit CPU saver over 768-dim
  v2 when the user has such models). No standardized distilled sub-encoder
  exists — don't design around a magic small one.
- **ORT CPU runtime profile (free coexistence win, all engines):**
  `intra_threads=1` (2 only if RTF>0.7 on GPU Quality), `inter_threads=1`,
  sequential exec, Level3 opt, memory-pattern on, and **critically
  `allow_spinning=0`** on both op pools — otherwise ORT busy-spins and pegs a
  full core at idle (the "VC idles at 100% CPU" complaint). Fixed chunk shape.
  **No fp16/bf16 on CPU** (Cast overhead makes it same-or-slower).
- **Realistic core budget:** on a modern **6–8-core** CPU, fcpe + int8 encoder
  reserves ~2 cores and coexists with OBS x264 + a populated VRChat world. On a
  **4-core** machine already running x264 + VRChat it is **not** reliably
  realtime (dropouts / 400–700 ms). Recommend OBS **NVENC/QSV/AMF** hardware
  encoding to free CPU cores. Benchmark on the real three-app scene, not VC alone.

---

## 5. Phased roadmap

### Phase 0 — Coexistence foundations (S/M) · *no user-visible feature*
Extract `ort_session.rs` (dedupe with `neural_onnx_tts.rs`) + add §4.1/§4.4
Streamer defaults (spinning off, intra/inter=1, capped arena). Add `vc/traits.rs`
stub. Replace the RT output-callback mutex with a lock-free ring (§4.7). Worker
**BelowNormal** priority. Add `chunk_ms` metric plumbing. Generalize the
`whisper.rs` downloader into a reusable base-model fetcher (optional — can slip to
Phase 1 if scope risk).

**Deliverable:** existing Classic/Onnx/Sidecar modes still work, now on the
shared session builder with spinning-off + capped arena (measurably lower idle
CPU/VRAM). No regression.

### Phase 1 — RVC MVP on CPU Streamer profile (XL) · *proves the product for streamers*
`FcpePitchExtractor` (ONNX). `RvcModel` (HuBERT→F0→generator, 6-input contract:
`phone[1,T,C] f32`, `phone_lengths[1] i64`, `pitch[1,T] i64`, `pitchf[1,T] f32`,
`ds[1] i64`, `rnd[1,192,T] f32` → `audio[1,N] f32`). `VcStreamHarness` (rolling
buffer + SOLA + two rubato resamplers). New `Rvc` mode wired through all
registration points. Manifest loader. BYO generator + base HuBERT/FCPE fetch.
Defaults = **Streamer** preset; `dio` as emergency fallback if needed for bring-up.
Frontend: format=rvc, preset selector, F0 select (fcpe/dio; gray out
rmvpe/crepe on CPU), Extra/protect, `chunk_ms` metric.

**Deliverable:** BYO RVC v2 voice on CPU, speech in → converted out,
`process_ms < chunk_ms` on a 6–8-core with OBS hardware encode preferred;
document 4-core limits.

### Phase 2 — GPU + Low latency + Quality presets (L)
CUDA/DML fp16, **rmvpe**, IO-binding + CUDA-graph with fixed shapes (§4.2/4.3),
warmup, VRAM caps per preset, GPU device index. Silence-front VAD. FAISS for
**Quality** (`index_rate` default 0 on Streamer/Low latency). RVC v1 + Nono
variants (from manifest). All three presets wired with Quality banner + live RTF.

**Deliverable:** Low latency <100–150 ms with GPU headroom; Quality audibly
better than Streamer on same voice while still realtime; Streamer remains
default; VRAM within preset caps.

### Phase 3 — F0 menu + weak-CPU escape (M)
Live F0 swap; WORLD `dio+StoneMask` CPU binding (BSD); confidence gate + median
smoother. int8 encoder path with CPUID gate; ContentVec/v1 dim as BYO saver.

**Deliverable:** user picks FCPE/RMVPE (GPU)/CPU-dio; weak-CPU users have an
escape hatch without leaving RVC.

### Phase 4 — Slots + optional QuickVC (M, deferrable)
Slot-based model manager (BYO import/replace UI), in-app consent/licensing
disclosure, 3-point audio debug capture (mic / post-inference / playback)
reusing the metrics IPC. Optional **QuickVC** (MIT) as a "lightweight mode"
secondary **only if** 4-core Streamer RTF still fails after NVENC guidance — not
required to ship Phases 0–3.

**Deliverable:** multiple named model slots, BYO-import flow, optional QuickVC.

> **Deferred / out of scope this round:** Beatrice v2 (EULA), so-vits-svc /
> Seed-VC (copyleft), DDSP-SVC (MIT but singing/F0-oriented, partial ONNX), LLVC
> (blocked on a production ONNX streaming export; not a shipped mode), CREPE-full
> + harvest offline "render" mode.

**Reuse vs rewrite:** *Extend* `ort_session.rs` (new), the `VoiceChunkPipeline`
seam, the `whisper.rs` downloader, `sidecar_bridge.rs` framing, output playback
plumbing, and all registration checklists. *Keep unchanged* `OnnxVoiceBridge`
(becomes "plain ONNX" mode). *New* `vc/` module (traits, rvc, f0 impls,
stream/SOLA harness, manifest, faiss) — genuinely new, not a refactor of the
black-box bridge.

---

## 6. Remaining risks & open items

1. **Phase 1 is genuinely XL.** The SOLA/rolling-context harness and the
   multi-input tensor contract are where most bugs live; frame-grid misalignment
   *sounds* broken before it sounds right. Budget accordingly.
2. **`ort` v2.0.0-rc.11 API surface** for IO-binding / CUDA-graph / spinning /
   memory-limit must be verified against the pinned crate at implementation time
   (rc churn). Fallback: some levers may require session-option strings rather
   than typed builders.
3. **`.pth→.onnx` export is on the user.** Needs clear in-app docs + link; a bad
   export (wrong opset, constant-folding on) is the likely #1 support issue.
4. **Latency targets** (tune from three-app benches): **Streamer** ~200–450 ms
   CPU; **Low latency** <100–150 ms GPU; **Quality** = higher Extra/index/VRAM,
   still realtime with RTF warning when headroom is insufficient.
5. **Consent/ethics:** voice cloning of real people risks platform-ToS/legal
   issues. Proposed posture: one-time in-app acknowledgment + BYO-model stance
   (Phase 4 UI; one sentence in operator docs earlier).

## 7. Key files

Existing: `src-tauri/src/services/voice_changer.rs`,
`neural_audio/{onnx_voice,chunk_processor,sidecar_bridge}.rs`,
`services/{whisper,neural_onnx_tts}.rs`; frontend
`src/core/services/voice_changer/{index,schema,chunkFrames}.ts`,
`src/core/ui/inspector/inspector_voice_changer.tsx`.

New tree to create: `src-tauri/src/services/neural_audio/ort_session.rs` and
`neural_audio/vc/{traits,stream,rvc,rmvpe,fcpe,world,manifest,faiss,f0_post}.rs`.
