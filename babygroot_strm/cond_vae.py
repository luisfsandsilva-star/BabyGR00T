"""Embodiment-conditioned shared VQ-VAE.

Single shared encoder + shared K-code codebook, decoder conditioned on
embodiment-id via a learned embedding added to the quantized bottleneck.

Rationale (mirrors Octo's "shared backbone + per-embodiment action head"):
  * Encoder sees per-chunk-precision-normalized actions, where similar motions
    look similar across embodiments — so a shared encoder learns motion
    semantics, not robot-specific quirks.
  * Codes therefore represent motion *primitives* shared across robots.
  * Decoder needs to know which robot to project back into — same code
    "move-forward-by-X" decodes to different joint deltas for widowx vs franka.
    A learned per-embodiment embedding added to the quantized bottleneck
    gives the decoder that signal cheaply.

API matches ActionVQVAE1d (used by the trainer + policy) — the only differences:
  * .encode/.decode/.forward take an extra `emb_id` arg (long tensor of shape (B,))
  * Single shared codebook → .vq is one module, .seq_lens is [bottleneck_T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cqvae import (GLUAct, ResBlock1d, conv_same_1d, conv_down_1d, VQ1d_EMA,
                    ACTION_DIM, D, K, BETA, CHUNK_LEN)
from .vqvae import _UpBlockNoSkip1d


class CondActionVQVAE1d(nn.Module):
    """Conditional shared VQ-VAE: emb_id conditions BOTH encoder and decoder.

    Why condition the encoder too: different embodiments use the same action
    slot to mean different things (joint angle vs ee-pose), so without the
    embodiment label the encoder would conflate incompatible inputs. With
    encoder conditioning the same code DOES mean different things per
    embodiment — that's OK; what matters is the codebook is well-used
    overall and the policy learns to predict the right code given the
    (vision, text, state, embodiment-id) prefix.

    Three injection points:
      1. Input  → emb embedding projected to action_dim, broadcast along T
      2. Bottleneck (pre-VQ) → emb embedding (d*4) added (encoder side)
      3. Bottleneck (post-VQ, pre-decode) → emb embedding (d*4) added (decoder side)

    Args:
        action_dim: max action_dim across embodiments (pad shorter ones with 0).
        n_embodiments: number of distinct embodiment IDs.
        d, k, beta, chunk_len: same as ActionVQVAE1d.
        vq_cls: VQ1d_EMA (default) or FSQ1d.
    """
    def __init__(self, action_dim=ACTION_DIM, n_embodiments=12, d=D, k=K,
                 beta=BETA, vq_cls=VQ1d_EMA, chunk_len=CHUNK_LEN):
        super().__init__()
        self.action_dim = action_dim
        self.n_embodiments = n_embodiments
        self.d = d
        self.chunk_len = chunk_len
        self.bottleneck_T = chunk_len // 4

        # Shared encoder
        self.stem   = conv_same_1d(action_dim, d)
        self.f1     = ResBlock1d(d);     self.proj12 = conv_down_1d(d,    d * 2)
        self.f2     = ResBlock1d(d * 2); self.proj23 = conv_down_1d(d * 2, d * 4)
        self.f3     = ResBlock1d(d * 4)

        # Shared codebook
        self.vq     = vq_cls(k, d * 4, beta=beta)

        # Per-embodiment conditioning at THREE points
        self.emb_input_proj  = nn.Embedding(n_embodiments, action_dim)   # added to raw action input
        self.emb_enc_proj    = nn.Embedding(n_embodiments, d * 4)        # added to encoder bottleneck (pre-VQ)
        self.emb_dec_proj    = nn.Embedding(n_embodiments, d * 4)        # added to quantized bottleneck (pre-decode)

        # Decoder — no encoder skips
        self.up2    = _UpBlockNoSkip1d(d * 4, d * 2)
        self.up1    = _UpBlockNoSkip1d(d * 2, d)
        self.out    = nn.Conv1d(d, action_dim, 1)

    @property
    def vqs(self):
        return [self.vq]

    @property
    def seq_lens(self):
        return [self.bottleneck_T]

    @property
    def kind(self):
        return 'cond_vqvae'

    def _encode_to_bottleneck(self, x, emb_id):
        """x: (B, A, T). emb_id: (B,) long. Both encoder + pre-VQ conditioning."""
        e_in = self.emb_input_proj(emb_id).unsqueeze(-1)              # (B, A, 1)
        x = x + e_in                                                   # condition input
        h = self.stem(x); h = self.f1(h); h = self.proj12(h)
        h = self.f2(h);   h = self.proj23(h); h = self.f3(h)
        e_enc = self.emb_enc_proj(emb_id).unsqueeze(-1)               # (B, d*4, 1)
        h = h + e_enc                                                  # condition encoder bottleneck
        return h

    def encode(self, x, emb_id):
        """Returns ([eq], commit_loss, [indices])."""
        h = self._encode_to_bottleneck(x, emb_id)
        eq, vql, idx = self.vq(h)
        return [eq], vql, [idx]

    def decode(self, embs, emb_id):
        """embs: [tensor (B, d*4, bottleneck_T)]. emb_id: (B,) long.

        Adds the per-embodiment decoder bias so the decoder knows which robot
        to render the code into.
        """
        h = embs[0]                                                   # (B, D, T_b)
        e = self.emb_dec_proj(emb_id).unsqueeze(-1)                   # (B, D, 1)
        h = h + e
        h = self.up2(h); h = self.up1(h)
        return self.out(h)

    def forward(self, x, emb_id):
        embs, vql, _ = self.encode(x, emb_id)
        recon = self.decode(embs, emb_id)
        recon_l = F.mse_loss(recon, x)
        return recon_l + vql, recon_l, vql

    @torch.no_grad()
    def encode_with_soft(self, x, emb_id, tau=0.71):
        """Soft-target encode. emb_id required since the encoder is conditioned."""
        scale = 2 * tau * tau
        h = self._encode_to_bottleneck(x, emb_id)
        B, C, T = h.shape
        zf = h.permute(0, 2, 1).reshape(-1, C)
        from .cqvae import _vq_distances
        dist = _vq_distances(zf, self.vq.emb.weight)
        soft = F.softmax(-dist / scale, dim=-1)
        hard = dist.argmin(1)
        return [hard.view(B, T)], [soft.view(B, T, -1)]

    @torch.no_grad()
    def decode_from_indices(self, indices, emb_id):
        idx = indices[0]
        B, T = idx.shape
        emb = self.vq.emb(idx).view(B, T, self.vq.D).permute(0, 2, 1)
        return self.decode([emb], emb_id)
