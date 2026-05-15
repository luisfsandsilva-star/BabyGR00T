# Recipes

Named recipes ship with `scripts/train_policy.py`:

- **v3** — Pareto-best (current default). Plain CE, depth=2/dim=768, outer
  Parcae 1/H, mask curriculum, stochastic-H training.
- **v4** — v3 + MSE-decode auxiliary loss in *corrected* form (expectation
  mode, all H cycles, small β). The original v4 (argmax-STE, final-cycle
  only, β=0.1) actively regressed mse_pol — see "v4 lesson" below.
- **v5** — v3 recipe trained on **BridgeData V2** (OXE,
  `IPEC-COMMUNITY/bridge_orig_lerobot`). **Different embodiment** (WidowX,
  not SO-101): 7-DoF action, 8-dim state, 53k episodes. Architecture
  unchanged; the data scale and embodiment both move.
- **v5-vqvae** — v5 architecture and recipe, with the action codebook
  swapped from the 3-level CQ-VAE to the single-level VQ-VAE
  (`ActionVQVAE1d`). The encoder is identical to the CQ-VAE; only the
  codebook structure changes. The clean comparison answer to "does the
  CQ-VAE's hierarchical residual + skip-connection structure earn its
  cost?"

---

## v3 — current default

```bash
python -m scripts.train_policy \
  --steps 25000 \
  --depth 2 --dim 768 \
  --L-inner 5 --H-outer 4 --h-max 12 \
  --rho1 0.75 --rho2 0.65 --rho-H 0.85 \
  --lr 9.5e-4 \
  --no-snce \
  --tau-anneal-frac 0.4 \
  --mask-curriculum --mask-curriculum-init 0.3 --mask-curriculum-frac 0.5 \
  --ckpt-path so101_strm_v3.pt
```

Results on the 78-episode SO-101 set (cold all-masked single forward, N=64):

| Run | Probe mean | mse_pol | top-1 / 5 / 10 | Notes |
|---|---:|---:|---:|---|
| **v3** | **12.4 %** | **0.486** | **11.2 / 26.8 / 39.5** | Current best non-FiLM baseline. |
| v2 | 11.3 % | 0.50 | 8.1 / 24.3 / 36.5 | depth=1, dim=512, no outer Parcae. |
| v1 | 10.7 % | 0.66 | 9.6 / 22.2 / 33.5 | Same as v2, with old `lars_coeff`. |

## v4 — corrected MSE-decode aux

The corrected recipe adds a small action-space MSE on the decoded prediction
to v3. The auxiliary loss is computed by passing the policy's predicted
*soft mixture* `Σ_k softmax(logits)_k · codebook_k` per level through the
**frozen** CQ-VAE decoder, and comparing to the (RevIN-normalized) ground
truth action chunk:

```
β · MSE( D( E_p[code_emb] ),  RevIN(action) )
```

The frozen decoder doesn't update — its parameters are detached — but
gradients flow back through it to the policy's logits, providing an
action-space training signal that the codebook CE alone doesn't.

```bash
python -m scripts.train_policy \
  --steps 25000 \
  --depth 2 --dim 768 \
  --L-inner 5 --H-outer 4 --h-max 12 \
  --rho1 0.75 --rho2 0.65 --rho-H 0.85 \
  --lr 9.5e-4 \
  --no-snce --tau-anneal-frac 0.4 \
  --mask-curriculum --mask-curriculum-init 0.3 --mask-curriculum-frac 0.5 \
  --mse-decode-weight 0.05 \
  --mse-decode-mode expectation \
  --mse-decode-cycles all \
  --ckpt-path so101_strm_v4.pt
```

### The v4 lesson

The original v4 ran with `--mse-decode-weight 0.1 --mse-decode-mode argmax
--mse-decode-cycles final`. It actively *hurt*: mse_pol regressed
0.486 → 0.61 and best-H collapsed from 4 → 1. Two compounding issues:

1. **Single-cycle MSE biases against iterative refinement.** Strong gradient
   on the final cycle made the model commit early; H ≥ 2 stopped helping.
2. **argmax-STE is gradient-noisy early in training.** When softmax is
   diffuse, the discontinuous argmax forward + soft-mixture backward has
   little correlation between forward action MSE and backward gradient
   direction.

The corrected recipe addresses both: `expectation` (forward = backward, no
discontinuity) + `all` (deep supervision preserved across H) + smaller β
(don't dominate the codebook CE). Treat v4 as exploratory until validated
side-by-side with v3.

## v5 — OXE migration (BridgeData V2, `IPEC-COMMUNITY/bridge_orig_lerobot`)

The current bottleneck on the 78-episode SO-101 set is data, not architecture
— param:target ratio is ~920:1 (pure memorization regime). Migrating to
**BridgeData V2** is also a **different embodiment** (WidowX, not SO-101),
which is the real cross-validation we want. Action space is 7-DoF
(6 + gripper), state is 8-dim; the CQ-VAE rebuilds against the new
distribution. 53k episodes / 1.9M frames at 5 fps.

The full migration is three steps. **Steps 1 and 2 take real time** (decoder
on hours-scale, vision cache on hours-scale); only do step 3 once both have
finished. Cap with `--n-eps-cap N` for a fast overnight first pass —
expect ~24 MB per chunk on disk, plan accordingly.

```bash
# 1. Re-train the CQ-VAE on Bridge's 7-DoF action distribution.
#    The codebook is dataset-specific.
python -m scripts.train_cqvae \
  --dataset oxe \
  --oxe-dataset-id IPEC-COMMUNITY/bridge_orig_lerobot \
  --steps 8000 --batch-size 32 \
  --action-dim 7 \
  --ckpt-path oxe_vae_revin.pt

# 2. Re-build the InternVL3 vision cache for Bridge.
#    Cameras are observation.images.image_0/_1/_2/_3; image_0 is the primary.
#    --n-vis-aug 0 for a fast first pass; bump it later for more diversity.
#    --n-eps-cap to limit disk during an overnight first run.
python -m scripts.cache_vision \
  --dataset oxe \
  --oxe-dataset-id IPEC-COMMUNITY/bridge_orig_lerobot \
  --oxe-camera observation.images.image_0 \
  --cache-dir oxe_vision_cache \
  --n-vis-aug 0 \
  --n-eps-cap 200

# 3. Train v5: identical hyperparameters to v3, only the data path differs.
#    state-dim is inferred from the first episode (=8 for Bridge); override
#    with --state-dim if needed.
python -m scripts.train_policy \
  --dataset oxe \
  --oxe-dataset-id IPEC-COMMUNITY/bridge_orig_lerobot \
  --oxe-camera observation.images.image_0 \
  --steps 50000 \
  --vae-ckpt oxe_vae_revin.pt \
  --cache-dir oxe_vision_cache \
  --ckpt-path oxe_strm_v5.pt
```

`load_lerobot_episodes()` works for any LeRobot-format dataset — pass
`--oxe-dataset-id <other-id>` to swap in a different OXE subset (e.g.
`IPEC-COMMUNITY/fractal20220817_data_lerobot` for RT-1 Google Robot). For
datasets with a different action dimensionality, also pass `--action-dim <d>`
to `train_cqvae.py` (or `train_vqvae.py`).

## v5-vqvae — direct CQ-VAE vs VQ-VAE comparison

This is the bit-for-bit-fairest version of "does the CQ-VAE's hierarchical
structure earn its cost?". Same encoder, same vision cache, same policy
hyperparameters, same training steps — only the codebook structure
changes:

| | CQ-VAE (v5) | VQ-VAE (v5-vqvae) |
|---|---|---|
| Encoder | `stem → f1 → proj12 → f2 → proj23 → f3` | **identical** |
| Codebooks | 3 (at T, T/2, T/4 resolutions) | 1 (at the T/4 bottleneck) |
| Codebook shape | 4×256, 8×128, 16×64 (K=128 each) | 4×256, K=128 |
| Residual passing across levels | yes (RQ-style) | no (single level) |
| Decoder | U-Net with encoder skip connections | pure upsample, no skips |
| Info capacity per chunk | 28 tokens × 7 bits = 196 bits | 4 tokens × 7 bits = 28 bits |
| Parameter count (action_dim=7) | ~2674 K | ~2588 K |
| Policy `seq_lens` | `(4, 8, 16)` | `(4,)` |

The 86 K-param difference is the second and third codebooks plus the
skip-conv U-Net decoder. Everything else is shared.

Run after the CQ-VAE pipeline (`overnight.sh`) has completed so the
vision cache and the comparison reference are already on disk:

```bash
./overnight_vqvae.sh > overnight_vqvae.log 2>&1 &
```

Outputs:
- `oxe_vqvae.pt` — single-level VQ-VAE codebook (~10 MB)
- `oxe_strm_v5_vqvae.pt` — S-TRM policy trained on top (~640 MB)

Eval both side-by-side:

```bash
python -m scripts.eval_policy oxe_strm_v5.pt       --vae-ckpt oxe_vae_revin.pt
python -m scripts.eval_policy oxe_strm_v5_vqvae.pt --vae-ckpt oxe_vqvae.pt
```

Both `train_policy.py` and `eval_policy.py` auto-detect the VAE kind
from the checkpoint's `kind` field — no flag needed at the policy step.

### Cache-time augmentation

**Visual augmentation** (`--n-vis-aug N`, default 0 = off): per chunk, sample
N additional sets of photometric + blur + small-crop parameters and re-run
InternVL3 with each. The transform is consistent across all frames in one
chunk so the temporal coherence is preserved (no flickering inside a clip).
Augmented features land in the same per-episode file; `ChunkDataset` samples
uniformly across variants when `augment=True`. Recommended for a second pass
once the un-augmented v5 baseline is established.

**Prompt sampling** (always on, no flag): each chunk-variant draws a
paraphrase from a built-in `PARAPHRASE_BANK` so InternVL3 sees prompt-level
diversity with zero external dependency. The bank covers pick-and-place and
red-cube-to-bowl phrasings; extend `babygroot_strm/augment.py` for new tasks.

### Held-out test split

The 78-episode SO-101 set was too small to support a meaningful held-out
test split — every probe number above is on training-set chunks (= upper
bound on actual generalization). With Bridge V2 (~53k episodes) you can
finally hold out 5–10 % of episodes and report an honest test number; do
this before drawing conclusions about v3 vs v4 vs anything new on OXE.
