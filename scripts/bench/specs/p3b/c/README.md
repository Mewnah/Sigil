# P3B-C specs

Head-to-head v2 (T6).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-C live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

| File | What |
|---|---|
| `compare-ablation.template.json` | The T6 dry run: a compare v2 run over a corpus (`${CORPUS}`) with the spec's SVE row (`${EP}`, `${PITCH}`, 40 ms engine blocks, the shipping auto levers) as `sigil`. |
| `vcrs-shape.template.json` | Same-rig compute rows at vc-rs's shape (100 ms chunk + 100 ms extra), per component ("model step incl. host I/O"); a timing run (GPU lock). Read it with `scripts/vc-bench/vcrs_probe.py --sigil <report.json>`. |
| `run_dryrun.ps1` | Materializes it. It adds the dry run's rows to a copy of the corpus: `abl-40` (anchored noise and phase0 off at the spec's shape), `sve-gui` / `abl-gui` (SVE and its ablation at the nearest SVE shape to RVC WebUI's GUI defaults: 200 ms block, 50 ms crossfade, 2.0 s extra), and stand-in references (the voice's whole-utterance offline render, seeds 7001 and 7002). It runs the bench under the GPU lock, records the background load, and writes `dryrun-<ep>.t6.md` / `.pareto.svg` with `scripts/vc-bench/compare_export.py`. |

```
python scripts/bench/make-test-clips.py --local-corpus --out <scratch>/corpus-local        # 10 clips of 10 s or more
pwsh scripts/bench/specs/p3b/c/run_dryrun.ps1 -Exe <exe>/sigil.exe -AppData <appdata> `
     -Corpus <scratch>/corpus-local/corpus.json -Ep cpu -Pitch fcpe -Out <scratch>/dry-cpu
pwsh scripts/bench/specs/p3b/c/run_dryrun.ps1 ... -Ep dml -Pitch rmvpe -Out <scratch>/dry-dml
```

The ablations are labelled ablations and never competitors. Their paired differences (versus −
ablation, per clip) show what SVE's time anchoring buys at matched shapes, and their paired DNSMOS
SD sizes the T6 corpus (`details.v2.sizing`, `clips_needed_dnsmos_50` / `_80`). If the margins are
not positive, run a short blind listening check (the run's `make_test` manifest, with `-WriteAudio`)
before concluding anything about metric sensitivity.
