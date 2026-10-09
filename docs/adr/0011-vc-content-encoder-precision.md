# ADR 0011: Content-encoder precision defaults and the deferred plan's cross-check (P3B-D)

- **Status:** accepted (P3B-D review fix, 2026-10-09). The waivers below are recorded decisions with
  numbers; P7-B measures the one open cost (the fp32 encoder on the CPU Streamer).
- **Date:** 2026-10-09
- **Workstream:** P3B-D (the ADR belongs to its workstream; INTEGRATE links it from `docs/adr/README.md`)
- **Plan decision:** P3B-D precision rules (workstream item 4: "int8 content encoder" and "fp16 ContentVec
  parity at stream lengths"); related [ADR 0010](0010-sve-optimization-p3b.md) (the P3B defaults),
  [ADR 0006](0006-vc-zero-shot-singing.md) finding 5 (DirectML pinned-dimension behaviour),
  [ADR 0009](0009-vc-scorer-v2.md) (the scorer that judged fidelity)

## Context

Two things in the content encoder (`vc/engine/content.rs`) did not match their written rules after P3B-D:

1. **Precision.** The plan's rules were: if fp16 ContentVec fails its parity gate at stream window lengths,
   the GPU runs fp32; if the int8 encoder loses more than 0.002 speaker cosine or 0.03 dB MCD against
   fp32, the CPU runs fp32 first and int8 stays an option. Both fired on the encoder's own metrics: the
   export's `embedder.json` records `parity.gpu: fail` for fp16 and `recommended: fp32` for both device
   classes, the catalog comment says DirectML "runs fp32", fp16 misses the export gate (rel L2 <= 1e-2,
   cosine >= 0.9995) at several stream lengths on both EPs (DirectML at the live 0.44 s window by rel L2
   1.2e-2, cosine 0.99993), and CPU int8 lost 0.0026 speaker cosine. Yet `variant_order` still tries fp16
   first on GPUs and int8 first on VNNI CPUs.
2. **Plan check.** The module docs said every fixed-shape plan is checked against the dynamic session. That
   holds only for eager encoders. The production GPU path pools Sigil store variants as *deferred*
   encoders (no dynamic session, so the voice's cold load builds one content session instead of two), and
   a deferred plan was only tested for finite, not-silent features.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Apply the rules: fp32 first on GPUs and on the CPU (`recommended` from `embedder.json`) | matches the export metadata and the written rule | +~170 MB VRAM per plan (T3 already misses), +0.5-1.0 ms per content call, an unmeasured CPU cost on a Streamer that already misses its RTF target; buys no measurable fidelity on the GPU | below |
| **B. Waive both rules with the downstream numbers on record; the rules' outcome is one setting away** | keeps the measured-neutral cheaper path; nothing hidden | the export metadata and the catalog comment describe the export gate, not the engine, until the catalog text is fixed | below |
| C. Check the deferred plan against a short-lived same-EP dynamic session | exact kernels, tight limit | builds the second session the deferred path exists to avoid (+217-274 MB VRAM transient, +230-380 ms) | `content_cold_load_dev_rig` |
| **D. Check the deferred plan against a CPU session of the same file on a helper thread, verdict cached per process** | no VRAM, hidden behind the plan's build, independent of the GPU driver | looser limit (the CPU is a different numeric path) | below |

## Decision

**B and D.**

- `content::variant_order` keeps fp16 first on GPUs and int8 first on VNNI CPUs, with the waiver and its
  per-EP numbers in the function's doc comment. `precision = fp32` puts fp32 first on both classes;
  `SIGIL_VC_INT8=0` turns the CPU's int8 off alone. A test pins the order and the fp32 outcome.
- A deferred plan (and any plan built while the encoder has no dynamic session) is compared with a CPU-EP
  session of the same file on the window that built it: relative L2 at most `PLAN_CPU_CHECK_REL_L2` (0.1).
  A plan over the limit is refused with a load note, the dynamic session serves the voice, and the
  refusal is kept for the process; a passing verdict is kept per (file, rung, device, window length,
  levers). A reference that cannot run leaves the plan on its finite test with a load warning.
  Kill switch `SIGIL_VC_CONTENT_CHECK=0`.

## Evidence

All on the RTX 4090 dev rig, ORT 1.28, scorer v2 where fidelity is named; commands and tables in
`scripts/bench/specs/p3b/d/README.md`.

Fidelity of the precision variants (the ceiling's content encoder pinned to fp32; 5 clips per EP, both
sides on the fixed-shape plan; median per-clip difference against the fp32 stream):

| metric | GPU fp16 - fp32, CUDA | GPU fp16 - fp32, DirectML | CPU int8 - fp32 (Streamer) |
|---|---|---|---|
| speaker cosine | +0.0008 | +0.0001 | **-0.0026** (rule limit 0.002) |
| MCD dB | +0.0045 | -0.0062 | -0.0123 (better; limit 0.03) |
| DNSMOS OVRL | +0.017 | -0.037 | -0.041 |
| output F0 RMSE (cents) | +0.46 | +0.72 | +10.5 (2 of 5 clips better) |
| V/UV error | -0.0014 | -0.0014 | -0.0058 (better) |

GPU fp16 moves the converted voice by less than 0.001 cosine and 0.007 dB in either direction, the
size of the plan-versus-plain difference on CUDA, so the export gate's failure at some window lengths
does not reach the output. fp32 would cost +170 MB per plan (CUDA 328 -> 498 MB, DirectML 208 -> 381 MB)
and +1.0 ms (CUDA) / +0.5 ms (DirectML) per content call. The CPU rule fired by 0.0006 speaker cosine
on one rig with MCD and V/UV better; the fp32 encoder's CPU time on the Streamer was not measured.

Deferred-plan check (`content_plan_cpu_reference_dev_rig`, CUDA and DirectML, fp16 and fp32 store
variants, 8 speech windows per length): worst relative L2 of the plan against the CPU session fp16 4.9e-3
(2 s) to 1.2e-2 (0.44 s), fp32 6e-6 to 2.5e-5, identical on both EPs; a 30 % gain error reads 0.3.
Added cold-load time with the verdict not on record: median +20 ms, mean +43 ms, range -79 to +180 ms
over 20 first planned calls (the CPU session builds in 230-450 ms and runs a window in 40-75 ms, behind
the plan's 330-890 ms build), inside T9's +0.3 s. DirectML static-versus-dynamic parity (same-EP):
fp16 bit-identical, fp32 <= 2.3e-6, asserted at <= 1e-3 for both precisions by `content_plan_dev_rig`.

## Consequences

- `variant_order` and the module docs state what is checked and why; the "every plan is checked
  against the dynamic session" wording is corrected to the two checks above.
- An install whose CPU cannot run the encoder, or a machine that is out of memory at load, loses only the
  extra check (a warning says so); a driver that compiles the plan wrongly falls back to the dynamic
  session instead of playing garbled features, with no environment variable needed.
- Follow-up: P7-B measures the fp32 encoder's CPU cost on the Streamer and decides int8 versus fp32 for
  the CPU default; `embedder.json`'s `recommended` and the `vc-embedders.json` comment should say that
  they describe the export gate (hotspot request to P3B-I0 / the orchestrator).
- Rollback: `SIGIL_VC_CONTENT_CHECK=0` removes the check; `precision = fp32` and `SIGIL_VC_INT8=0` give
  the precision rules' outcome.
