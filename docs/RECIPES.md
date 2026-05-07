# Recipes

Three named recipes ship with `scripts/train_policy.py`:

- **v3** — Pareto-best (current default). Plain CE, depth=2/dim=768, outer
  Parcae 1/H, mask curriculum, stochastic-H training.
- **v4** — v3 + MSE-decode auxiliary loss in *corrected* form (expectation
  mode, all H cycles, small β). The original v4 (argmax-STE, final-cycle
  only, β=0.1) actively regressed mse_pol — see "v4 lesson" below.
- **v5** — v3 recipe trained on a much larger LeRobot dataset
  (`lerobot/svla_so101_pickplace`, same SO-101 embodiment). Architecture
  unchanged; the only thing that moves is the data scale.

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

## v5 — OXE migration (lerobot/svla_so101_pickplace)

The current bottleneck on the 78-episode SO-101 set is data, not
architecture: param:target ratio is ~920:1 — pure memorization regime.
Swapping to `lerobot/svla_so101_pickplace` (same SO-101 arm, same
6-DoF action space, ~50× more episodes) drops the ratio to a healthy range
without changing anything else.

The full migration is three steps. **Steps 1 and 2 take real time** (decoder
on hours-scale, vision cache on hours-scale); only do step 3 once both have
finished.

```bash
# 1. Re-train the CQ-VAE on the OXE action distribution.
#    The codebook is dataset-specific; using the SO-101-trained one
#    leaves dead/unused codes for any new motion patterns.
python -m scripts.train_cqvae \
  --dataset oxe \
  --oxe-dataset-id lerobot/svla_so101_pickplace \
  --steps 8000 --batch-size 32 \
  --ckpt-path oxe_vae_revin.pt

# 2. Re-build the InternVL3 vision cache for OXE.
#    With visual + prompt augmentation: each chunk gets 1 original + 3
#    photometrically-jittered variants; each variant uses a different
#    paraphrased task prompt drawn from a 25-prompt LLM-generated pool.
#    Both forms of augmentation run ONCE here — training reads from disk.
#    `lerobot/svla_so101_pickplace` cameras are `observation.images.up` and
#    `observation.images.side` (not `front`); pick one.
python -m scripts.cache_vision \
  --dataset oxe \
  --oxe-dataset-id lerobot/svla_so101_pickplace \
  --oxe-camera observation.images.up \
  --cache-dir oxe_vision_cache \
  --n-vis-aug 3 \
  --llm-augment-prompts --n-prompt-paraphrases 25

# 3. Train v5: identical hyperparameters to v3, only the data path differs.
python -m scripts.train_policy \
  --dataset oxe \
  --oxe-dataset-id lerobot/svla_so101_pickplace \
  --steps 50000 \
  --vae-ckpt oxe_vae_revin.pt \
  --cache-dir oxe_vision_cache \
  --ckpt-path oxe_strm_v5.pt
```

`load_lerobot_episodes()` works for any LeRobot-format dataset — pass
`--oxe-dataset-id <other-id>` to swap in a different OXE subset (e.g.
`lerobot/bridge_data_v2` for WidowX BridgeData V2). For datasets with a
different action dimensionality, also pass `--action-dim <d>` to
`train_cqvae.py`.

### Cache-time augmentation (recommended for OXE)

Two flavors, both applied once during `cache_vision.py`:

**Visual augmentation** (`--n-vis-aug N`): per chunk, sample N additional sets
of photometric+blur+crop parameters and re-run InternVL3 with each. The
transform is consistent across all frames in one chunk so the temporal
coherence is preserved (no flickering inside a clip). Augmented features land
in the same per-episode file; `ChunkDataset` samples uniformly across
variants when `augment=True`.

**Prompt augmentation** (`--llm-augment-prompts --n-prompt-paraphrases K`):
build a K-paraphrase pool per unique task prompt by calling the Anthropic
API once per prompt (cheap — O(unique prompts), not O(chunks)). Each cached
variant draws a random paraphrase from the pool, so InternVL3 sees prompt
diversity. The script falls back to a hand-curated static bank if
`ANTHROPIC_API_KEY` is unset, so it works offline too.

The combination is what gives the cache real diversity: one chunk with
N visual variants × 1 prompt each is much weaker than N visual variants ×
N different prompts. Use both together.

For BridgeData V2 (or any non-SO-101 dataset) you'll want a richer paraphrase
bank than the built-in one — set `--llm-augment-prompts` and let the LLM
write task-specific paraphrases.

### Held-out test split

The 78-episode SO-101 set was too small to support a meaningful held-out
test split — every probe number above is on training-set chunks (= upper
bound on actual generalization). With `svla_so101_pickplace` you can finally
hold out 5–10 % of episodes and report an honest test number; do this
before drawing conclusions about v3 vs v4 vs anything new on OXE.
