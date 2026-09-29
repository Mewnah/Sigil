# ADR NNNN: <short decision title>

- **Status:** proposed | accepted | superseded by ADR NNNN | rejected
- **Date:** YYYY-MM-DD
- **Workstream:** <e.g. P1-G1> (the ADR belongs to its workstream; INTEGRATE links it from `docs/adr/README.md`)
- **Plan decision:** <name of the decision in docs/SPEECH_STACK_IMPLEMENTATION_PLAN.md, if any>

## Context

What problem forces a decision now, and what constraints apply (quality > local > free, Windows first,
native isolation, network gentleness, no license warnings in UI or docs). Link the research section.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A | | | measured numbers / links |
| B | | | |

## Decision

The option chosen, stated so an implementer can act on it without reading the rest.

## Evidence

Measurements that decided it (latency p50/p99, RTF, VRAM, CPU, WER/CER, parity error, build/link
results), with the exact command, hardware (rig H1/H2/H3), model and commit. Mark numbers from vendors
as such; defaults ship only from Sigil's own measurements.

## Consequences

- What changes in code, packaging and docs.
- What becomes harder, and the rollback or fallback path (from the plan's `fallback:` line).
- Follow-up work and the workstream that owns it.
