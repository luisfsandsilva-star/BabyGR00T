"""Per-image RevIN-style normalization with a global-variance floor.

The CNN's first conv layer otherwise learns to expect a specific
absolute pixel scale / mean / contrast.  At test time (sim, different
lighting, different sensor) those statistics shift and the CNN's
representations drift.  We subtract the per-image per-channel mean and
divide by an instance-specific *precision* (1 / std² with a floor),
forcing the CNN to learn from relative patterns instead.

Form (same as the action-chunk normalization):
    m   = img.mean(spatial)
    S   = ((img - m) ** 2).sum(spatial)
    λ   = HW / (S + HW · var_global)
    out = (img - m) · √λ

`var_global` (per-channel pixel variance averaged over a sample of the
training set) is the "Gamma prior" — it stops λ from exploding on
low-variance images (uniform sim backgrounds, blank patches).
"""
import torch


@torch.no_grad()
def normalize_image(img: torch.Tensor, var_global: torch.Tensor, eps: float = 1e-6):
    """Per-image RevIN.

    Args:
        img: (..., C, H, W) float tensor, any scale.
        var_global: (C,) per-channel pixel variance prior, computed once
            over the training set (e.g. ~0.02-0.06 for natural images in [0,1]).
        eps: numerical floor on the denominator (essentially unused if var_global > 0).

    Returns:
        Same shape as `img`, per-image standardized with the global-var floor.
    """
    *lead, C, H, W = img.shape
    HW = H * W
    m = img.mean(dim=(-2, -1), keepdim=True)                          # (...,C,1,1)
    S = ((img - m) ** 2).sum(dim=(-2, -1), keepdim=True)              # (...,C,1,1)
    vg = var_global.view(*([1] * len(lead)), C, 1, 1).to(img.dtype).to(img.device)
    lam = HW / (S + HW * vg + eps)                                    # (...,C,1,1) precision
    return (img - m) * lam.sqrt()


@torch.no_grad()
def compute_image_var_global(loader_or_iter, n_batches: int = 200, channels_last_chw: bool = True) -> torch.Tensor:
    """Estimate per-channel pixel variance over the training set.

    Iterates a few batches, accumulates per-image var, returns the channel-wise mean.
    Use this once at preprocessing time and persist the result.

    Args:
        loader_or_iter: iterable that yields tensors of shape (B, C, H, W) in [0,1].
        n_batches: how many batches to average over.
    Returns:
        (C,) tensor of per-channel pixel variance.
    """
    acc = None; n = 0
    it = iter(loader_or_iter)
    for _ in range(n_batches):
        try: x = next(it)
        except StopIteration: break
        if isinstance(x, (tuple, list)): x = x[0]
        if not channels_last_chw and x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)                                  # NHWC → NCHW
        # per-image per-channel variance, then average across the batch
        v = x.var(dim=(-2, -1), unbiased=False).mean(dim=0)            # (C,)
        if acc is None: acc = torch.zeros_like(v)
        acc += v; n += 1
    return acc / max(n, 1)
