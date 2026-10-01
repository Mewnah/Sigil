"""Cut the community FCPE export (gzivdo/fcpe-onnx fcpe.onnx: waveform -> torchfcpe mel -> model ->
local-argmax F0) at its log-mel tensor, giving two real mel-input FCPE graphs for Sigil's dev-rig test
`vc::fcpe::tests::real_mel_fcpe_matches_the_waveform_export`:
  fcpe-mel-latent.onnx  mel [1, T, 128] -> latent [1, T, 360]
  fcpe-mel-f0.onnx      mel [1, T, 128] -> f0 [1, T, 1] (Hz, the graph's own decode)
Usage: python fcpe_mel_extract.py <dev-cache>/upstream-assets/fcpe/fcpe.onnx
"""
import os, sys
import onnx
from onnx.utils import Extractor

src = sys.argv[1]
out_dir = os.path.dirname(src)
model = onnx.load(src)
mel = "/model/Concat_1_output_0"
for name, out, new_out in [("fcpe-mel-latent.onnx", "/model/Sigmoid_output_0", "latent"), ("fcpe-mel-f0.onnx", "f0_hz", "f0")]:
    sub = Extractor(model).extract_model([mel], [out])
    g = sub.graph
    for n in g.node:
        n.input[:] = ["mel" if x == mel else (new_out if x == out else x) for x in n.input]
        n.output[:] = [new_out if x == out else x for x in n.output]
    g.input[0].name = "mel"
    g.output[0].name = new_out
    # Declare the shapes the export would have: [1, T, 128] in.
    dims = g.input[0].type.tensor_type.shape.dim
    del dims[:]
    for d in (1, "n_frames", 128):
        dd = dims.add()
        if isinstance(d, int):
            dd.dim_value = d
        else:
            dd.dim_param = d
    # (ONNX Runtime validates the graph when it loads it.)
    onnx.save(sub, os.path.join(out_dir, name))
    print("wrote", os.path.join(out_dir, name), os.path.getsize(os.path.join(out_dir, name)))
