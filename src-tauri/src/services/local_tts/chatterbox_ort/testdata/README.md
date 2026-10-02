# Chatterbox test data

Small files of the Chatterbox Multilingual V3 ONNX export, used by `chatterbox_ort/tests.rs` so the
tests exercise the real tokenizer, the real prompt-embedding graph and the real flow-step
declarations without the 1.8 GB model:

| File | From | sha256 |
|---|---|---|
| `tokenizer.json` | `tokenizer.json` | `c8889d91e083486df2454fcda24ef11a1586247ac3ed2b81cf6b6b91070053ee` |
| `token_embedding.onnx` | `onnx/token_embedding.onnx` (graph only; the tests write a stand-in for its 69 MB `token_embedding.onnx.data` with the same layout) | `7e70760aa14023c9dbc3cb2a7bd98214f765ebf77b0dc0851a00c81a6eefa543` |
| `flow_step_slim.io.pb` | `onnx/flow_step_slim.onnx` (the CPU flow step, 286 MB with its weights inline): the graph's `input` and `output` declarations (`GraphProto` fields 11 and 12, 315 bytes) cut from a ranged read of the file's last 2 MB, where they follow the weights | `2ec4b5e123ff2501c8384690e9b8a80ffd83905beba03ba0c4cda42e4a53711f` |
| `flow_step_fp16.io.pb` | `onnx/flow_step_fp16.onnx` (the optional graphics-card step, sha256 `38d69d93…6149`): the same declarations, cut from the graph file | `2ec4b5e123ff2501c8384690e9b8a80ffd83905beba03ba0c4cda42e4a53711f` |

Both flow steps declare the same eight inputs (`x`, `mask` bool, `mu`, `speakers`, `cond`, `time`,
`delta`, `guidance`) and one output (`next_x`), byte for byte; `both_flow_step_exports_take_what_
the_driver_binds` checks them against what `model::Flow::render` binds.

Source: [KitsuMate/chatterbox-multilingual-v3-onnx](https://huggingface.co/KitsuMate/chatterbox-multilingual-v3-onnx)
at commit `3f44461db006cc1ffb553cda02d6f6841125288a`, an ONNX export of Resemble AI's
[Chatterbox](https://huggingface.co/ResembleAI/chatterbox) (MIT License, Copyright (c) 2025
Resemble AI). The same files are installed with the `tts/chatterbox-mtl-v3` catalog entry (and
`tts/chatterbox-mtl-v3-gpu-flow` for the fp16 step).
