# Architecture decision records

Decisions that are expensive to reverse (runtimes, native loading, installers, security model) are
recorded here. Each ADR belongs to the workstream that made the decision; the phase integrator links
it from this index. Start a new one from [0000-template.md](0000-template.md) and number it after the
highest existing record (numbers reserved by the [implementation plan](../SPEECH_STACK_IMPLEMENTATION_PLAN.md)
keep their slot even when written out of order).

| ADR | Title | Status | Workstream |
|-----|-------|--------|------------|
| [0000](0000-template.md) | Template | — | — |
| [0003](0003-gpu-ep-delivery.md) | Delivering GPU execution providers and reporting the one in use | Accepted (P0 interim; P1 revisits CUDA provider placement after the ORT rc.13 gate) | P0-H |

Reserved by the plan, not written yet: 0001 (ORT version gate), 0002 (sherpa-onnx), 0003b (transcribe.cpp
DLL pack), 0004 (RVC templates), 0005 (process host / llama-server), 0006 (zero-shot and singing export
feasibility), 0008 (Bergamot helper).
