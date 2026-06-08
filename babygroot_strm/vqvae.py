"""Single-bottleneck convolutional VQ-VAE — direct comparison baseline for
the 3-level CQ-VAE (`ActionRQUNet1d`).

The architecture is **identical to the CQ-VAE's encoder** up to the
deepest bottleneck (stem → f1 → proj12 → f2 → proj23 → f3) and the
**single VQ has the same shape as the CQ-VAE's coarsest level vq3**
(4 tokens × D·4 channels × K codes). The only differences:

  1. **One codebook** instead of three (no residual quantization).
  2. **Decoder has no skip connections** — only the quantized bottleneck
     propagates forward, mirrored by the same two upsample steps the
     CQ-VAE uses but without concatenating encoder features. This is the
     direct test of whether the CQ-VAE's hierarchical residual structure
     and U-Net skips earn their parameter cost on a given dataset.

Information capacity: 4 tokens × log2(K=128) = 28 bits/chunk
                       vs CQ-VAE's (4 + 8 + 16) × 7 = 196 bits/chunk.

The reduced capacity is the cost of the simpler design — the experiment
asks whether that cost shows up in reconstruction MSE and/or in the
downstream policy probe accuracy, given otherwise-identical training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cqvae import (
    GLUAct, ResBlock1d, conv_same_1d, conv_down_1d, VQ1d_EMA, _vq_distances,
    ACTION_DIM, D, K, BETA, CHUNK_LEN,
)


# ════════════════════════════════════════════════════════════
#  Single-level VQ-VAE
# ════════════════════════════════════════════════════════════

class _UpBlockNoSkip1d(nn.Module):
    """Pure-upsample block — same internals as CQ-VAE's UpBlock1d minus the
    encoder-skip concatenation. Doubles the temporal resolution.
    """
    def __init__(self, ic, oc):
        super().__init__()
        self.up = nn.ConvTranspose1d(ic, ic, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(ic, oc, 3, padding=1, bias=False),
            nn.GroupNorm(8, oc), GLUAct(oc), ResBlock1d(oc),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class ActionVQVAE1d(nn.Module):
    """Convolutional VQ-VAE with a single codebook at the bottleneck.

    Encoder: identical to ActionRQUNet1d up to vq3 (stem → f1 → proj12 →
    f2 → proj23 → f3). Output shape (B, D·4, T/4).

    Bottleneck VQ: K codes × (D·4) channels. Default K=128, T=16 → 4 tokens.

    Decoder: two pure-upsample blocks (no skip from encoder), then a 1×1
    conv to action_dim. Mirrors the CQ-VAE decoder's depth without its
    cross-level information shortcuts.
    """
    def __init__(self, action_dim=ACTION_DIM, d=D, k=K, beta=BETA,
                 vq_cls=VQ1d_EMA, chunk_len=CHUNK_LEN, dropout=0.0,
                 binary_last=False, dead_threshold=5):
        super().__init__()
        self.action_dim = action_dim
        self.d = d
        self.chunk_len = chunk_len
        self.bottleneck_T = chunk_len // 4    # 4 tokens for default chunk_len=16
        # binary_last: treat the last action dim (gripper) as a Bernoulli variable —
        # the decoder's last output channel is a LOGIT, trained with BCE and read out
        # via sigmoid (see recon_to_actions). Encoder input + codes are unchanged
        # (gripper stays precision-normalized on the way in), so the policy's
        # encode/target path is identical; only the decode→action readout differs.
        self.binary_last = bool(binary_last)
        # [lo, hi] original gripper range, set at train time; maps sigmoid prob → action units.
        # Registered ONLY when binary_last, so non-binary (legacy) ckpts still load strictly.
        if self.binary_last:
            self.register_buffer('gripper_range', torch.tensor([0.0, 1.0]))

        # Encoder — bit-for-bit identical to CQ-VAE up to vq3
        self.stem   = conv_same_1d(action_dim, d)
        self.f1     = ResBlock1d(d);     self.proj12 = conv_down_1d(d,    d * 2)
        self.f2     = ResBlock1d(d * 2); self.proj23 = conv_down_1d(d * 2, d * 4)
        self.f3     = ResBlock1d(d * 4)

        # Single bottleneck VQ — same shape as CQ-VAE's vq3. Pass dead_threshold
        # through to VQ1d_EMA (codebook revival cadence); FSQ1d ignores it.
        try:
            self.vq = vq_cls(k, d * 4, beta=beta, dead_threshold=dead_threshold)
        except TypeError:
            self.vq = vq_cls(k, d * 4, beta=beta)

        # Decoder — no encoder skips
        self.up2    = _UpBlockNoSkip1d(d * 4, d * 2)
        self.up1    = _UpBlockNoSkip1d(d * 2, d)
        self.out    = nn.Conv1d(d, action_dim, 1)

        # Dropout1d after each major stage (encoder + decoder). Dropout1d zeros entire
        # channels per-sample, which is more impactful for convs than per-element Dropout.
        # Zero parameters so doesn't break loading of existing dropout=0 ckpts.
        # Acts on activations between stages — forces the network to develop redundant
        # representations rather than memorizing per-sample features.
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

    # ── Uniform API for the policy + eval (mirrors ActionRQUNet1d) ──
    @property
    def vqs(self):
        return [self.vq]

    @property
    def seq_lens(self):
        return [self.bottleneck_T]

    @property
    def kind(self):
        return 'vqvae'

    # ── Encode / decode ──
    def _encode_to_bottleneck(self, x):
        h = self.stem(x)
        h = self.drop(self.f1(h))                          # dropout after stage 1
        h = self.proj12(h)
        h = self.drop(self.f2(h))                          # dropout after stage 2
        h = self.proj23(h)
        h = self.drop(self.f3(h))                          # dropout after stage 3 (pre-VQ)
        return h

    def encode(self, x):
        """Returns ([eq], commit_loss, [indices]).
        List-of-one form keeps the call signature identical to the CQ-VAE.
        """
        h = self._encode_to_bottleneck(x)
        eq, vql, idx = self.vq(h)
        return [eq], vql, [idx]

    def decode(self, embs):
        """embs: list of one tensor (B, d*4, bottleneck_T)."""
        h = embs[0]
        h = self.drop(self.up2(h))                         # dropout in decoder too
        h = self.drop(self.up1(h))
        return self.out(h)

    def forward(self, x):
        embs, vql, _ = self.encode(x)
        recon = self.decode(embs)
        recon_l = F.mse_loss(recon, x)
        return recon_l + vql, recon_l, vql

    # ── SNCE / soft-target encode (matches CQ-VAE API) ──
    @torch.no_grad()
    def encode_with_soft(self, x, tau=0.71):
        scale = 2 * tau * tau
        h = self._encode_to_bottleneck(x)
        B, C, T = h.shape
        zf = h.permute(0, 2, 1).reshape(-1, C)
        dist = _vq_distances(zf, self.vq.emb.weight)
        soft = F.softmax(-dist / scale, dim=-1)
        hard = dist.argmin(1)
        return [hard.view(B, T)], [soft.view(B, T, -1)]

    @torch.no_grad()
    def decode_from_indices(self, indices):
        """indices: list of one (B, T) tensor."""
        idx = indices[0]
        B, T = idx.shape
        emb = self.vq.emb(idx).view(B, T, self.vq.D).permute(0, 2, 1)
        return self.decode([emb])

    def recon_to_actions(self, recon, m, lam):
        """Convert raw decoder output → action units. recon: (B, A, T).
        m, lam: (B, 1, A) per-chunk precision-norm stats. Continuous dims are
        de-normalized; if binary_last, the last channel is a logit → sigmoid →
        mapped to the stored [lo, hi] gripper range. Returns (B, T, A)."""
        recon_t = recon.transpose(1, 2)                          # (B, T, A)
        act = recon_t / lam.sqrt() + m
        if self.binary_last:
            lo, hi = self.gripper_range[0], self.gripper_range[1]
            act = act.clone()
            act[..., -1] = lo + (hi - lo) * torch.sigmoid(recon_t[..., -1])
        return act
