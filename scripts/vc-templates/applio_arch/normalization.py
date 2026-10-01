# Vendored from IAHispano/Applio (MIT, see LICENSE in this folder)
# Source: rvc/lib/algorithm/normalization.py at commit c7665ac9a305b3683570ed914f1577d53d4b75c4
# Upstream sha256: a341c27623a2f42c43547a3ed96789aecea20b7f4e6b5420cfe14cc84cb57172
# Sigil modification: package-relative imports only (`from rvc.lib.algorithm ...` -> `from . ...`).
# Do not edit the model code below; Sigil's export wrappers live in rvc_arch/split.py.
import torch


class LayerNorm(torch.nn.Module):
    """
    Layer normalization module.

    Args:
        channels (int): Number of channels.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Transpose to (batch_size, time_steps, channels) for layer_norm
        x = x.transpose(1, -1)
        x = torch.nn.functional.layer_norm(
            x, (x.size(-1),), self.gamma, self.beta, self.eps
        )
        # Transpose back to (batch_size, channels, time_steps)
        return x.transpose(1, -1)
