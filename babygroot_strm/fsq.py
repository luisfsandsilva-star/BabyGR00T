"""Finite Scalar Quantization (Mentzer et al. 2024).

Drop-in replacement for VQ1d_EMA — same forward signature, same `.K`,
`.D`, `.emb(idx)` API used by ActionVQVAE1d. Differences:

  * No learned codebook → no dead codes possible by construction.
  * Quantization is just `tanh → scale → round` (with straight-through grad).
  * Codes are integers obtained by mixed-radix encoding of the d_fsq
    quantized dims.  K = ∏ levels.

Implementation pattern from the FSQ paper, simplified for our 1d use:
  z (B, D, T)
    → linear D → d_fsq
    → tanh ⋅ (L-1)/2
    → round (straight-through)
    → mixed-radix integer (per token)
    → linear d_fsq → D (so the rest of the VAE sees the same shape)
"""
import torch
import torch.nn as nn


def _pick_levels_for_K(K: int):
    """Heuristic factorization of K into 3-5 ODD integers (FSQ levels).

    Why odd: with odd L the grid {-(L-1)/2, ..., -1, 0, 1, ..., (L-1)/2} consists of
    exact integers, so `round()` lands on a grid point cleanly. Even L would need
    a half-shift before rounding (FSQ paper detail) — we sidestep that here.
    """
    table = {
         27:  [3, 3, 3],
         125: [5, 5, 5],     # closest to K=128 with odd levels
         343: [7, 7, 7],
         729: [9, 9, 9],
         875: [7, 5, 5, 5],  # the FSQ-paper recommendation
         1125:[5, 5, 5, 9],
         1715:[7, 7, 7, 5],
    }
    if K in table: return table[K]
    raise ValueError(f"No default FSQ levels for K={K}; pass `levels=` explicitly. "
                     f"Available: {sorted(table.keys())}")


class FSQ1d(nn.Module):
    """1D FSQ — drop-in for VQ1d_EMA.

    Args:
        K:      number of unique codes (must equal product of `levels`).
        D:      channel dim of the input (B, D, T).
        beta:   ignored (kept for API parity).
        levels: list of per-dim quantization levels. Defaults: a small factor
                table picks 3-5 dims that multiply to K.
    """
    def __init__(self, K: int, D: int, beta: float = 0.0, levels=None):
        super().__init__()
        levels = levels or _pick_levels_for_K(K)
        assert int(torch.tensor(levels).prod().item()) == K, \
            f"levels {levels} must multiply to K={K}"
        self.K = K
        self.D = D
        self._levels = nn.Parameter(torch.tensor(levels, dtype=torch.long), requires_grad=False)
        # mixed-radix multipliers for converting per-dim codes ↔ flat integer
        cm = torch.ones(len(levels), dtype=torch.long)
        for i in range(1, len(levels)):
            cm[i] = cm[i - 1] * int(levels[i - 1])
        self._radix = nn.Parameter(cm, requires_grad=False)
        # projection D ↔ d_fsq
        self.d_fsq = len(levels)
        self.proj_down = nn.Linear(D, self.d_fsq)
        self.proj_up   = nn.Linear(self.d_fsq, D)
        # match VQ1d_EMA's `.emb` API — _Emb is NOT an nn.Module to avoid the
        # FSQ1d ↔ _Emb reference cycle through .children() / state_dict tracking.
        object.__setattr__(self, 'emb', _Emb(self))

    # ── core quantization ────────────────────────────────────────
    def _quantize(self, z_low: torch.Tensor) -> torch.Tensor:
        """z_low: (..., d_fsq) any range → quantized to integer grid in [-(L-1)/2, (L-1)/2]."""
        L = self._levels.to(z_low.dtype)
        # tanh keeps continuous bounded; multiply gives grid; round + STE quantizes
        z_bounded = torch.tanh(z_low) * (L - 1) / 2
        z_q = z_bounded + (z_bounded.round() - z_bounded).detach()
        return z_q

    def _idx_of(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (..., d_fsq) on integer grid → (...,) flat integer in [0, K)."""
        L = self._levels                                         # (d_fsq,)
        per_dim = (z_q + (L.to(z_q.dtype) - 1) / 2).round().long()
        per_dim = torch.minimum(per_dim, L - 1).clamp(min=0)     # PER-DIM clamp
        return (per_dim * self._radix.to(per_dim.dtype)).sum(-1)

    def _z_of_idx(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (...) flat int → (..., d_fsq) on the integer grid."""
        L = self._levels
        idx = idx.long().unsqueeze(-1)
        per_dim = (idx // self._radix) % L.to(idx.device)
        return per_dim.to(torch.float32) - (L.to(torch.float32) - 1) / 2

    def forward(self, z: torch.Tensor):
        """z: (B, D, T) → (z_q (B,D,T), vq_loss (scalar, 0), idx (B,T))."""
        B, D, T = z.shape
        zt = z.permute(0, 2, 1)                              # (B,T,D)
        z_low = self.proj_down(zt)                           # (B,T,d_fsq)
        z_q_low = self._quantize(z_low)                      # (B,T,d_fsq)
        z_q = self.proj_up(z_q_low).permute(0, 2, 1)         # (B,D,T)
        idx = self._idx_of(z_q_low)                          # (B,T)
        return z_q, torch.zeros((), device=z.device, dtype=z.dtype), idx


class _Emb:
    """Callable that mimics nn.Embedding(K, D) — needed because the VAE
    code uses `self.vq.emb(idx)` to decode.

    PLAIN object (not nn.Module) to avoid a parameter/child cycle with the
    parent FSQ1d, which uses _Emb itself but should not own it twice.
    """
    def __init__(self, parent):
        # store via object.__setattr__ on a weak-ref-free direct slot;
        # plain object is fine here, GC handles it.
        self._parent = parent

    @property
    def weight(self):
        """Materialize the (K, D) codebook on demand. Used by _vq_distances
        in the SNCE / encode_with_soft path. Cheap for K≤4096."""
        idx = torch.arange(self._parent.K, device=self._parent._levels.device)
        z_low = self._parent._z_of_idx(idx)              # (K, d_fsq)
        return self._parent.proj_up(z_low)               # (K, D)

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (...,) → (..., D). Matches nn.Embedding(K, D)(idx) signature."""
        z_low = self._parent._z_of_idx(idx)
        return self._parent.proj_up(z_low)
