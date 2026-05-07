"""3-level convolutional CQ-VAE for action chunks.

Hierarchy (coarsest first):
  L0: 4 tokens × D*4 channels    (stride-4 downsample of CHUNK_LEN)
  L1: 8 tokens × D*2 channels
  L2: 16 tokens × D channels     (= CHUNK_LEN raw)

Each level has a K=128 codebook (EMA-updated) and contributes one of three
soft-target distributions used to train the policy. RevIN normalizes the
action chunk before the encoder and inverts after the decoder, so the model
operates in a clean per-instance unit-variance space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Defaults (per-codebook dim D, codebook size K, commitment β) ──
ACTION_DIM = 6
CHUNK_LEN  = 16
D          = 64
K          = 128
BETA       = 0.5
SEQ_LENS_1D = [4, 8, 16]


# ════════════════════════════════════════════════════════════
#  RevIN — Reversible Instance Normalization (Kim et al., ICLR 2022)
# ════════════════════════════════════════════════════════════

class RevIN(nn.Module):
    """Per-instance normalization with cached stats for invertible denorm.
    Forward modes:  'norm' caches mean/std; 'denorm' uses the cached values.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None
        self._mean = None
        self._std = None

    def forward(self, x, mode):
        if mode == 'norm':
            self._mean = x.mean(dim=-2, keepdim=True)
            self._std = (x.var(dim=-2, keepdim=True, unbiased=False) + self.eps).sqrt()
            x = (x - self._mean) / self._std
            if self.weight is not None:
                x = x * self.weight + self.bias
            return x
        elif mode == 'denorm':
            if self.weight is not None:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._std + self._mean
            return x
        raise ValueError(f"Unknown mode: {mode}")


# ════════════════════════════════════════════════════════════
#  1D building blocks (GLU-gated conv stack)
# ════════════════════════════════════════════════════════════

class GLUAct(nn.Module):
    """SwiGLU-style gated activation in conv-channel space."""
    def __init__(self, c):
        super().__init__()
        self.up = nn.Conv1d(c, 2 * c, 1, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=1)
        return a * F.silu(b)


class ResBlock1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c), GLUAct(c),
            nn.Conv1d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(8, c), GLUAct(c),
            nn.Conv1d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


def conv_same_1d(ic, oc):
    return nn.Sequential(
        nn.Conv1d(ic, oc, 3, padding=1, bias=False),
        nn.GroupNorm(8, oc), GLUAct(oc), ResBlock1d(oc),
    )


def conv_down_1d(ic, oc):
    return nn.Sequential(
        nn.Conv1d(ic, oc, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(8, oc), GLUAct(oc), ResBlock1d(oc),
    )


class UpBlock1d(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up = nn.ConvTranspose1d(ic, ic, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(ic + sc, oc, 3, padding=1, bias=False),
            nn.GroupNorm(8, oc), GLUAct(oc), ResBlock1d(oc),
        )

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


# ════════════════════════════════════════════════════════════
#  Vector quantizer (EMA codebook, dead-code revival)
# ════════════════════════════════════════════════════════════

def _vq_distances(zf, codebook):
    return torch.cdist(zf, codebook, p=2).square()


class VQ1d_EMA(nn.Module):
    """Codebook with EMA-updated centroids and Laplace smoothing.
    Dead codes (unused for `dead_threshold` consecutive batches) are revived
    by reseeding to a fresh encoder vector.
    """
    def __init__(self, k, d, beta=BETA, decay=0.99, eps=1e-5, dead_threshold=5):
        super().__init__()
        self.K, self.D, self.beta = k, d, beta
        self.decay, self.eps, self.dead_threshold = decay, eps, dead_threshold
        self.emb = nn.Embedding(k, d)
        nn.init.uniform_(self.emb.weight, -1 / k, 1 / k)
        self.register_buffer('ema_count',  torch.ones(k))
        self.register_buffer('ema_weight', self.emb.weight.data.clone())
        self.register_buffer('dead_counter', torch.zeros(k, dtype=torch.long))

    def _flatten(self, z):
        B, C, T = z.shape
        return z.permute(0, 2, 1).reshape(-1, C), B, T

    def forward(self, z):
        zf, B, T = self._flatten(z)
        dist = _vq_distances(zf, self.emb.weight)
        idx = dist.argmin(1)
        eq = self.emb(idx).view(B, T, -1).permute(0, 2, 1)

        if self.training:
            onehot = F.one_hot(idx, self.K).float()
            counts = onehot.sum(0)
            self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
            self.ema_weight.mul_(self.decay).add_(onehot.t() @ zf, alpha=1 - self.decay)
            N = self.ema_count.sum()
            n_s = (self.ema_count + self.eps) / (N + self.K * self.eps) * N
            self.emb.weight.data.copy_(self.ema_weight / n_s.unsqueeze(1))
            used = counts > 0
            self.dead_counter[used] = 0
            self.dead_counter[~used] += 1
            dead = self.dead_counter >= self.dead_threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                ri = torch.randint(0, zf.shape[0], (n_dead,), device=zf.device)
                nv = zf[ri].detach() + torch.randn(n_dead, self.D, device=zf.device) * 0.01
                self.emb.weight.data[dead] = nv
                self.ema_count[dead] = 1.0
                self.ema_weight[dead] = nv.clone()
                self.dead_counter[dead] = 0

        loss = self.beta * F.mse_loss(z, eq.detach())
        return z + (eq - z).detach(), loss, idx.view(B, T)


# ════════════════════════════════════════════════════════════
#  3-level CQ-VAE
# ════════════════════════════════════════════════════════════

class ActionRQUNet1d(nn.Module):
    """3-level residual VQ U-Net over (B, ACTION_DIM, T) action chunks.
    Coarsest first in encode_with_soft / decode: [L0=4×D*4, L1=8×D*2, L2=16×D].
    """
    def __init__(self, action_dim=ACTION_DIM, d=D, k=K, beta=BETA, vq_cls=VQ1d_EMA):
        super().__init__()
        self.action_dim = action_dim
        self.d = d
        self.stem   = conv_same_1d(action_dim, d)
        self.f1     = ResBlock1d(d);    self.proj12 = conv_down_1d(d,    d * 2)
        self.f2     = ResBlock1d(d * 2); self.proj23 = conv_down_1d(d * 2, d * 4)
        self.f3     = ResBlock1d(d * 4)
        self.vq1    = vq_cls(k, d,     beta=beta)   # finest  (16 tokens)
        self.vq2    = vq_cls(k, d * 2, beta=beta)   # mid     (8 tokens)
        self.vq3    = vq_cls(k, d * 4, beta=beta)   # coarse  (4 tokens)
        self.d2     = UpBlock1d(d * 4, d * 2, d * 2)
        self.d1     = UpBlock1d(d * 2, d,     d)
        self.out    = nn.Conv1d(d, action_dim, 1)

    def encode(self, x):
        """Returns (embs[L0,L1,L2], commit_loss, indices[L0,L1,L2])."""
        r = self.stem(x); vql = 0.0
        h1 = self.f1(r);  e1, l1, i1 = self.vq1(h1)
        r = r + (h1 - e1.detach()); r = self.proj12(r); vql += l1
        h2 = self.f2(r);  e2, l2, i2 = self.vq2(h2)
        r = r + (h2 - e2.detach()); r = self.proj23(r); vql += l2
        h3 = self.f3(r);  e3, l3, i3 = self.vq3(h3); vql += l3
        return [e3, e2, e1], vql, [i3, i2, i1]

    def decode(self, embs):
        e3, e2, e1 = embs
        return self.out(self.d1(self.d2(e3, e2), e1))

    def forward(self, x):
        embs, vql, _ = self.encode(x)
        recon = self.decode(embs)
        return F.mse_loss(recon, x) + vql, F.mse_loss(recon, x), vql

    @torch.no_grad()
    def encode_with_soft(self, x, tau=0.71):
        """SNCE-style encoding: hard indices + soft codebook distributions.
        soft_l = softmax(-||z - c_k||² / (2τ²)) per position.
        Used by the policy as soft-CE targets (or just hard CE with this method).
        """
        scale = 2 * tau * tau

        def soft_of(h, vq):
            B, C, T = h.shape
            zf = h.permute(0, 2, 1).reshape(-1, C)
            dist = _vq_distances(zf, vq.emb.weight)
            soft = F.softmax(-dist / scale, dim=-1)
            hard = dist.argmin(1)
            return hard.view(B, T), soft.view(B, T, -1)

        r = self.stem(x)
        h1 = self.f1(r);  i1, q1 = soft_of(h1, self.vq1)
        e1 = self.vq1.emb(i1).permute(0, 2, 1)
        r = r + (h1 - e1.detach()); r = self.proj12(r)
        h2 = self.f2(r);  i2, q2 = soft_of(h2, self.vq2)
        e2 = self.vq2.emb(i2).permute(0, 2, 1)
        r = r + (h2 - e2.detach()); r = self.proj23(r)
        h3 = self.f3(r);  i3, q3 = soft_of(h3, self.vq3)
        return [i3, i2, i1], [q3, q2, q1]

    @torch.no_grad()
    def decode_from_indices(self, indices):
        vqs = [self.vq3, self.vq2, self.vq1]
        B = indices[0].shape[0]
        embs = []
        for idx, vq, T_l in zip(indices, vqs, SEQ_LENS_1D):
            embs.append(vq.emb(idx).view(B, T_l, vq.D).permute(0, 2, 1))
        return self.decode(embs)


def cosine_snce_tau(step, total_steps, tau_max=2.0, tau_min=0.1, anneal_frac=0.4):
    """Cosine anneal from `tau_max` to `tau_min` over `anneal_frac` of training.
    v3 uses anneal_frac=0.4 — most training happens at low tau (sharp targets).
    """
    import math
    anneal_steps = max(1, int(total_steps * anneal_frac))
    if total_steps <= 0 or step >= anneal_steps:
        return tau_min
    progress = max(0, step) / anneal_steps
    return tau_min + (tau_max - tau_min) * 0.5 * (1 + math.cos(math.pi * progress))
