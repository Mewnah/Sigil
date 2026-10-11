# Documentation gaps and recommendations

This file tracks **remaining documentation work** after the consolidation (spring 2026). Use it as a backlog; check items off in pull requests when addressed.

## Consolidation summary (resolved)


| Before                                 | After                                                                                      |
| -------------------------------------- | ------------------------------------------------------------------------------------------ |
| ~20+ separate `docs/manual/*.md` pages | **6** manual files: `README.md` + `00-overview` … `04-system`                              |
| Many standalone `docs/*.md` references | **3** reference guides: `OPERATOR_REFERENCE`, `DEVELOPER_GUIDE`, `RELEASE_AND_ENGINEERING` |
| Long `ARCHITECTURE.md`                 | Short [ARCHITECTURE.md](../ARCHITECTURE.md) pointer + full content in `DEVELOPER_GUIDE`    |


**Resolved:** Single navigation surface in-app (`[HelpManualModal](../src/core/ui/HelpManualModal.tsx)`), `[sync-manual.mjs](../scripts/sync-manual.mjs)` publishes `public/manual/` and `public/docs/`, anchor links and `rehype-slug` for in-modal scrolling.

## Resolved (follow-up)

### Sidecar READMEs vs operator reference

- **Done:** [OPERATOR_REFERENCE.md](OPERATOR_REFERENCE.md) includes a [Sidecar scripts (repo)](OPERATOR_REFERENCE.md#sidecar-scripts-repo) section in the table of contents and a dedicated section linking to [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md) and [`scripts/translation-sidecar/README.md`](../scripts/translation-sidecar/README.md).

### Historical CHANGELOG links

- **Done:** [CHANGELOG.md](../CHANGELOG.md) includes an intro sentence pointing readers to [`docs/`](../docs/) for current filenames; older release bullets are left as historical record.

### Packaged-app smoke (Phase A)

- **Done:** [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#local-packaged-smoke-manual) documents a **local packaged smoke** pass after `pnpm tauri build`. CI Playwright remains preview-only; optional Windows `workflow_dispatch` / scripts stay backlog-only (see same doc).

### i18n strings referencing repo paths

- **Done:** [CONTRIBUTING.md](../CONTRIBUTING.md) includes guidance not to translate `docs/...` path segments; `pnpm i18n:check` guards key parity. English strings use `docs/OPERATOR_REFERENCE.md#…` where applicable.

### Dependency audit cadence

- **Done:** [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md) adds **Before tagging a release (dependency hygiene)** steps, a packaging checklist item to refresh [Dependency audit notes](RELEASE_AND_ENGINEERING.md#dependency-audit-notes), and maintainer pointers in [CONTRIBUTING.md](../CONTRIBUTING.md).

### Screenshots and shot list

- **Done (structure):** [SHOTLIST.md](manual/assets/SHOTLIST.md) maps each PNG filename to a manual chapter; chapters reference `assets/<name>.png`. Minimal valid placeholder PNGs are produced by [`scripts/write-manual-placeholder-pngs.mjs`](../scripts/write-manual-placeholder-pngs.mjs) so the in-app Help loads real image URLs. **Follow-up:** swap in real screenshots (same filenames) when convenient.

## Open gaps and recommendations

- **Real VRChat clips, friends' exports and the unbenchmarked tiers.** The P7 numbers are filled ([SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md), [VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md)), but they come from one dev rig, a synthetic corpus and a game stand-in. Still open: real VRChat clips in the corpus' `real` folder, `export.json` files from a CPU-only PC and from an RTX 3060 / 4060 (`scripts/speech-bench/score/merge_exports.py`), the heavy GPU models (opt-in, not benchmarked), Bergamot (pack unbuilt), Edge TTS timing, an H3 / H1 voice-changer record, and the competitor head-to-head (T6). Screenshots in the manual are still placeholders.

## Changelog (this document)

- **2026-04-17:** Initial version after doc consolidation; gaps list above.
- **2026-04-18:** Sidecar TOC in operator reference; CHANGELOG intro for pre-2026 doc paths; renumbered open gaps.
- **2026-04-18:** Resolved packaged smoke Phase A, i18n policy, dependency cadence, screenshot structure/placeholders.
- **2026-10-09:** P7 documentation pass: engine choice by hardware tier, the speech benchmark, the headless command list, optional VR / two-PC checks, the Moonshine long-utterance behaviour; open gap for the P7 numbers.
- **2026-10-10:** P7 integrate: every `TODO(P7-numbers)` placeholder filled from the benchmark reports, the changed defaults (Japanese and Russian CPU finals, interim translations on the processor, the CPU Streamer thread policy), the release tooling and the exit guard documented; the P7 numbers gap replaced by the open real-data items.

