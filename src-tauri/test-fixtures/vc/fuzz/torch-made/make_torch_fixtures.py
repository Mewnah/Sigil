#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real ``torch.save`` output for the importer's checkpoint reader (``vc::import::pth`` tests).

    python src-tauri/test-fixtures/vc/fuzz/torch-made/make_torch_fixtures.py

needs PyTorch (any version >= 1.6 writes the zip format; these were made with 2.3.1). The fuzz
corpus generator next door assembles pickles opcode by opcode without torch; these files are the
check that what it assembles matches what torch writes. A state dict saved by ``torch.save`` is an
``OrderedDict`` followed by a ``BUILD`` of its ``_metadata`` version table, which is what
training checkpoints (``G_*.pth``) and bare state dicts carry.

Writes, from one small module whose parameter names follow RVC's generator (``enc_p.emb_phone``,
``emb_g``, ``dec.ups.0`` with the legacy ``weight_norm``, ``dec.conv_post`` with the
parametrization API):

- ``state-dict.pth``: ``torch.save(net.state_dict())``;
- ``data-parallel.pth``: ``torch.save(nn.DataParallel(net).state_dict())`` (``module.`` keys);
- ``G_7.pth``: an RVC training checkpoint (``model``, ``iteration``, an AdamW ``optimizer`` state
  after one step, ``learning_rate``);
- ``expected.json``: every tensor of the state dict (flattened, float32) under its normalised
  name (``*.parametrizations.weight.original0/1`` -> ``*.weight_g/v``).
"""

import json
import os

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))


class EncP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_phone = nn.Linear(4, 2)


class Dec(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.ModuleList([nn.utils.weight_norm(nn.ConvTranspose1d(2, 2, 4, 2))])
        self.conv_post = nn.utils.parametrizations.weight_norm(nn.Conv1d(2, 1, 3))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_p = EncP()
        self.emb_g = nn.Embedding(1, 2)
        self.dec = Dec()


def normalize(key):
    for i, suffix in ((0, "g"), (1, "v")):
        tail = f".parametrizations.weight.original{i}"
        if key.endswith(tail):
            return key[: -len(tail)] + f".weight_{suffix}"
    return key


def main():
    torch.manual_seed(20260930)
    net = Generator()
    state = net.state_dict()
    torch.save(state, os.path.join(HERE, "state-dict.pth"))
    torch.save(nn.DataParallel(net).state_dict(), os.path.join(HERE, "data-parallel.pth"))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    loss = sum(p.square().sum() for p in net.parameters())
    loss.backward()
    opt.step()
    torch.save(
        {"model": net.state_dict(), "iteration": 7, "optimizer": opt.state_dict(), "learning_rate": 1e-4},
        os.path.join(HERE, "G_7.pth"),
    )
    expected = {normalize(k): v.detach().float().flatten().tolist() for k, v in state.items()}
    after = {normalize(k): v.detach().float().flatten().tolist() for k, v in net.state_dict().items()}
    with open(os.path.join(HERE, "expected.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"torch": torch.__version__, "state_dict": expected, "G_7": after}, f, indent=1)
        f.write("\n")
    print(f"torch {torch.__version__}: wrote {len(expected)} tensors")


if __name__ == "__main__":
    main()
