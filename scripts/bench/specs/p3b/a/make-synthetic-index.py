"""Write a synthetic retrieval index for the P3B-A kNN-30k benchmark row (knn30k.template.json).

The GPU kNN graph is exact: it reads every row of the index on every tick, so its VRAM and compute
depend on the row count and width only, not on the values. The rows here are clustered Gaussian
features at about ContentVec's scale (the default voice ships no index, so retrieval was never
benchmarked). The file uses the reader's legacy raw layout (`faiss.rs`: little-endian u32 dim,
u32 rows, rows x dim f32; read as an L2 flat index), so no FAISS install is needed.

    python scripts/bench/specs/p3b/a/make-synthetic-index.py --out <app data>/bench/synthetic-30k.index

It is written locally (about 92 MB at 30000 x 768) and never committed.
"""
import argparse
import os
import random
import struct
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="index file to write")
    ap.add_argument("--rows", type=int, default=30_000)
    ap.add_argument("--dim", type=int, default=768, help="768 (ContentVec v2 voices) or 256 (v1)")
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    if args.rows < 64 or args.dim not in (256, 768):
        print("rows must be >= 64 and dim 256 or 768", file=sys.stderr)
        return 2
    try:
        import numpy as np  # fast path
    except ImportError:
        np = None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<II", args.dim, args.rows))
        if np is not None:
            rng = np.random.default_rng(args.seed)
            centres = rng.normal(0.0, 0.6, size=(args.clusters, args.dim)).astype(np.float32)
            done = 0
            while done < args.rows:
                n = min(4096, args.rows - done)
                pick = rng.integers(0, args.clusters, size=n)
                rows = centres[pick] + rng.normal(0.0, 0.25, size=(n, args.dim)).astype(np.float32)
                f.write(rows.astype("<f4").tobytes())
                done += n
        else:
            rng = random.Random(args.seed)
            centres = [[rng.gauss(0.0, 0.6) for _ in range(args.dim)] for _ in range(args.clusters)]
            for _ in range(args.rows):
                c = centres[rng.randrange(args.clusters)]
                f.write(struct.pack("<%df" % args.dim, *[x + rng.gauss(0.0, 0.25) for x in c]))
    os.replace(tmp, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {args.rows} x {args.dim} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
