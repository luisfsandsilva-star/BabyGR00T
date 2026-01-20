import torch
from torch import nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization over time dimension for per-sample features.
    Normalizes inputs x: [B, L, F] to zero-mean unit-variance per (B,F) over L.
    Provides inverse using the same per-sample statistics captured in forward.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def _stats(self, x: torch.Tensor):
        # x: [B, L, F]
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        return mu, std

    def normalize(self, x: torch.Tensor):
        # returns normalized x and a context dict for inversion
        mu, std = self._stats(x)
        xhat = (x - mu) / std
        if self.affine:
            xhat = xhat * self.gamma + self.beta
        ctx = {"mu": mu, "std": std}
        return xhat, ctx

    def denormalize_out(self, y: torch.Tensor, ctx):
        # invert affine, then re-scale to original stats
        if self.affine:
            y = (y - self.beta) / (self.gamma + self.eps)
        mu, std = ctx["mu"], ctx["std"]
        return y * std + mu

    def normalize_in(self, y: torch.Tensor, ctx):
        # Apply same normalization as input (for targets)
        # (y - mu) / std
        # Then affine
        mu, std = ctx["mu"], ctx["std"]
        y_norm = (y - mu) / std
        if self.affine:
            y_norm = y_norm * self.gamma + self.beta
        return y_norm
