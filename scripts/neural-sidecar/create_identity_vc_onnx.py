#!/usr/bin/env python3
"""
Build a float32 Identity ONNX graph with fixed shape [1, N] -> [1, N] for testing
Sigil voice changer **ONNX model file** mode.

By default N = CHUNK_FRAMES * channels (512 * 2 = 1024 for stereo). Sigil can adapt
other sizes at runtime (mono/stereo and length), but this script is still the
simplest way to produce a known-good test model.

Requires: pip install onnx

Example:
  python create_identity_vc_onnx.py -o identity_vc_stereo.onnx
  python create_identity_vc_onnx.py --channels 1 -o identity_vc_mono.onnx
"""

from __future__ import annotations

import argparse

import onnx
from onnx import TensorProto, helper


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channels", type=int, default=2, help="Input device channels (1 or 2 typical)")
    p.add_argument(
        "--chunk-frames",
        type=int,
        default=512,
        help="Frames per chunk (must match Sigil CHUNK_FRAMES)",
    )
    p.add_argument("-o", "--output", default="identity_vc.onnx", help="Output .onnx path")
    args = p.parse_args()
    n = args.chunk_frames * args.channels
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity_vc", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, args.output)
    print(f"Wrote {args.output} with shape [1, {n}] (float32 Identity)")


if __name__ == "__main__":
    main()
