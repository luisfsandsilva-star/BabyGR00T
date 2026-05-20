# S-TRM v3 — Architecture

This document describes the policy used in `babygroot_strm`. It is a
**stabilized, recursive** vision-language-action model that operates over a
discrete codebook produced by a frozen 3-level CQ-VAE. The recursion is
contractive by construction (Parcae channel-wise decay + 1/L averaging +
convex-combo gates, optionally tightened with Miyato spectral norm — see
§3.1), so the same checkpoint works at much larger H at inference time than
it was trained at.

---

## 1. The data path

A single training sample is one (episode, chunk) pair from the SO-101 dataset.
A "chunk" is a window of `T = 16` consecutive `(action, state)` frames.

| Tensor | Shape | Source |
|---|---|---|
| Action chunk      | `(T=16, A=6)` | LeRobot dataset |
| Robot state       | `(A=6,)`      | LeRobot dataset |
| Visual hidden     | `(25, N_tok, 896)` | InternVL3-1B (cached, int8) |
| Task prompt       | text          | `TASK_PROMPTS` |

Vision is run **once, offline** by `scripts/cache_vision.py`. The 25 LLM-layer
hidden states for every chunk are stored in per-channel symmetric int8
(cos > 0.99999 vs fp16). Training is purely a disk read after that.

## 2. The action codebook (frozen during policy training)

A 3-level convolutional residual VQ U-Net. Coarsest first:

| Level | Tokens | Codebook dim | Notes |
|---|---:|---:|---|
| L0 | 4  | D·4 = 256 | Coarsest; captures gross trajectory shape |
| L1 | 8  | D·2 = 128 | Mid-frequency residual |
| L2 | 16 | D = 64    | Finest; per-step residual |

Each level has a `K=128` codebook, EMA-updated, with dead-code revival.
**RevIN** (Reversible Instance Normalization, Kim et al. 2022) normalizes the
action chunk per-instance before the encoder and inverts at the output, so the
VAE works in a clean unit-variance space.

The frozen VAE provides two things to the policy:

- **Hard targets:** `target_indices[l] ∈ {0,...,K-1}^{B×T_l}`
- **Soft targets (SNCE):** `soft_l = softmax(-‖z - c_k‖² / 2τ²)` per position.
  Used as the cross-entropy distribution when `--snce` is set; v3 uses plain
  CE on hard targets (SNCE was strictly worse on this dataset).

## 3. The policy: additive closed-form-decay TRM

The policy joins three sources of information through cross-attention:

```
[ vis (128 latents) | state (1 token) ]      ←  cross-attn KV
[ L0 (4) | L1 (8) | L2 (16) ]                ←  query stream (28 action tokens)
```

It is **vanilla TRM** — two vector latents `z_H` (solution) and `z_L`
(reasoning), `z ∈ ℝ^{B × 28 × d}`, one shared tiny net `g`, L inner / H outer
recursions, deep supervision — with one change: the latents are refined
**additively with a decaying weight** instead of replaced. Convergence comes
from the *update rule*, not from constraining the network, so the recurrence
needs no contraction / spectral norm / Lipschitz bound.

### 3.1 Building blocks

- **ScaleNorm** (Nguyen & Salazar 2019): single-scalar L2 rescale, used to
  pre-norm each sub-layer inside `g`.
- **Vanilla attention** (QK-norm + plain softmax): no softmax-1 / zero-sink.
  A head that wants to abstain routes mass to the cross-attention KV prefix
  (vision + state).
- **GeGLU** (Shazeer 2020): `GeLU(W₁x) ⊙ W₂x → W₃`.
- **Shared tiny net `g`** (`depth` sub-blocks of SelfAttn → CrossAttn →
  GeGLU, each pre-ScaleNorm). `g` returns the **sum of its sub-layer
  transforms** — a *pure update direction*, no identity pass-through. The
  accumulation of `g`-outputs is itself the residual stream. One `g` is
  reused for every inner and outer step (TRM "tiny shared net").

### 3.2 Closed-form decay weights

Each loop refines its latent by a *decaying weighted sum* of the transforms.
The weights are a **closed form in the loop length** `n` (= L or H):

```
a_t = ρ^{ t / (n-1) } = ρ^{linspace(0,1,n)_t} ,    t = 0,…,n-1
```

`a_0 = 1` (full-weight first step), `a_{n-1} = ρ` exactly (last step), and the
profile in loop-fraction `t/(n-1)` is identical for any `n`. `ρ_L`, `ρ_H` are
a single learnable scalar per loop (sigmoid → (0,1)); only the *rate* is
learned. Because `a_t` is monotone-decreasing and the loop is a finite sum of
bounded transforms, the accumulation is bounded for any `n` — and the same
refinement curve is sampled at however many points the step budget gives, so
test-time L/H scaling is stable by construction (no contraction needed).

### 3.3 Inner recurrence (L steps)

```
z_L = 0
for t = 0..L-1:
    z_L = z_L + a_t^{(L)} · g(z_L + z_H + y, kv)      # a_t = ρ_L^{t/(L-1)}
```

`g`'s input is the *accumulated-so-far* reasoning latent `z_L` plus the
current solution `z_H` and the task embedding `y`; it cross-attends to the
vision/state KV. The first step (`z_L = 0`) already sees full conditioning,
so the front-loaded `a_t` schedule is coarse-to-fine, not blind. `z_L` is
re-derived fresh each outer cycle (TRM: the answer persists, the reasoning is
recomputed).

### 3.4 Outer recurrence (H cycles)

```
z_H = 0
for h = 0..H-1:
    z_L = inner_loop(z_H, y, kv, L)                  # §3.3
    z_H = z_H + a_h^{(H)} · g(z_H + z_L + y, kv)      # a_h = ρ_H^{h/(H-1)}
    logits_h = head(z_H)                             # deep supervision each h
```

`z_H` is the running solution latent — it accumulates decayed updates across
the H cycles and is read by the head at every cycle. Because the head reads
the **raw** `z_H` (no norm), the logit magnitude grows as `z_H` sharpens over
cycles: more compute → more confident predictions.

- **Stochastic-H training**: per call, sample `H ~ Uniform{1..h_max}` with
  `h_max=12`. Combined with the step-count-invariant weight profile, this
  makes test-time H scaling behave — the model isn't tied to one H.

### 3.5 Loss and training signal

Cross-entropy is applied at **every cycle** (deep supervision):

```
loss = (1/H) · Σ_h (1/n_levels) · Σ_l avg_masked_CE(logits_h^l, target^l)
```

CE only counts masked positions — the rest are teacher-forced via the input
embedding `y`. Default recipe uses plain CE on hard targets (`--no-snce`);
SNCE soft targets are also supported.

The head predicts the **K real codes** (a plain Linear, no norm). The
prediction lives in the latent `z_H` (TRM-style latent feedback), so it is
*not* re-embedded from the logits — the input `y` is a static MASK-marker /
GT embedding (MASK is one extra learned row used only to mark unknown
positions, never a prediction target). The mask-ratio curriculum (§3.7)
trains the model to fill in fully-masked sequences from vision+state context,
which is the deployment case.

### 3.6 Optional auxiliary loss: action-space MSE through the frozen decoder

An optional knob (off by default; see `docs/RECIPES.md` for the v4 recipe):

```
loss_total = loss_CE  +  β · (1/H_used) · Σ_h ‖ D(E_p_h[code_emb]) - RevIN(action) ‖²
```

where `D` is the frozen CQ-VAE decoder and `E_p_h[code_emb]` is the soft
mixture `Σ_k softmax(logits_h)_k · codebook_k` at cycle h. The decoder's
parameters are not updated — `requires_grad=False` — but gradients flow
back through it to the policy's logits, giving an action-space signal that
codebook CE alone doesn't.

The original v4 (β=0.1, argmax-STE forward, final-cycle only) was an
anti-pattern: it caused the model to over-commit at the final cycle and
abandon iterative refinement. The corrected recipe used here uses the
expectation forward (smooth, gradient agrees with forward), applies the
loss to *every* H cycle (preserves deep supervision), and keeps β small.

### 3.7 Mask-ratio curriculum

On step `t` we sample mask ratio per level uniformly in
`[1/T_l, mask_ratio_max(t)]`, where `mask_ratio_max` cosine-ramps from 0.3
to 1.0 over the first 50% of training. This gives the model easier
(low-mask) examples first, then progressively harder ones, ending at the
all-masked regime that matches deployment.

## 4. Vision pipeline

`InternVL3-1B` is run offline only. At training time the policy reads from
the int8-quantized cache and feeds the 25-layer stack through:

- **LayerAggregator**: per-token, per-channel softmax gating over the 25
  hidden states. Produces a single `(B, N_tok, 896)` tensor that is a
  learned mixture of low/mid/high LLM layers.
- **PerceiverResampler** (Flamingo, Alayrac et al. 2022): cross-attends 128
  learned latents to the visual tokens. KV concatenates the latents with
  the visual features, so latents talk to each other and to the visual
  tokens within a single attention pass. Output is `(B, 128, dim)` —
  the policy's visual KV prefix.

### 4.1 Cache-time augmentation

Because InternVL3 is run only once per (chunk, variant), augmentation
happens at *cache* time, not training time. Two complementary forms (see
`docs/RECIPES.md` for the flags):

- **Visual:** photometric jitter (brightness / contrast / saturation / hue)
  + small Gaussian blur + small centered crop-and-resize. One random draw
  per (chunk, variant) is applied identically to every frame in that chunk —
  temporal coherence inside a clip is preserved (no flickering), but
  different variants of the same chunk see different lighting/blur.
- **Prompt:** each variant draws a paraphrase of the task prompt from the
  built-in `PARAPHRASE_BANK` in `babygroot_strm/augment.py`. InternVL3 sees
  prompt-level diversity in addition to visual diversity, so the cached
  features cover a wider stretch of the (image, instruction) joint
  distribution. Extend the bank for new tasks.

`ChunkDataset(augment=True)` samples one variant per `__getitem__`; with
N visual variants × N prompts the effective dataset size is `n_chunks ×
(1 + n_vis_aug)` and the policy never sees the same (image, prompt) pair
twice in a row.

## 5. Optimizer: MuSGD + LARS

`MuSGD_LARS` is a manual implementation that combines:

- **Muon-style Newton-Schulz orthogonalization** for 2D weight matrices
  (Jordan et al. 2024) — replaces the gradient direction with the nearest
  orthogonal matrix scaled to unit operator norm.
- **Plain SGD with momentum** for 1D parameters (biases, norms).
- **LARS trust ratio** `‖w‖ / ‖update‖` per parameter (no scalar coefficient,
  removed in v3 — it was redundant with `lr`).
- **Decoupled weight decay** (`w ← w·(1 - lr·wd)`).

LARS gives well-scaled per-parameter updates without needing a per-layer lr
schedule (and pairs cleanly with either contraction mode).

## 6. Test-time scaling

Stability under more L/H steps at eval is built into the **update rule**, not
enforced on the network:

- Each loop's output is a decaying weighted sum `Σ_t a_t·g(·)` with
  `a_t = ρ^{t/(n-1)}` — a bounded sum of bounded transforms for any `n`, so
  the accumulation cannot blow up no matter how many steps you run.
- The weight *profile* in loop-fraction `t/(n-1)` is identical for every `n`,
  so running more steps samples the same refinement curve more finely rather
  than changing its character — the model trained at one (L,H) generalizes to
  larger (L,H) at eval. Stochastic-H training (`H~U{1..h_max}`) reinforces this.
- Because the head reads the raw accumulated `z_H`, more outer compute yields
  a larger-magnitude (sharper) prediction — verified `|logits|` grows
  monotonically across cycles and with total H, finite at every H.

Practically: watch the eval H-scaling curve. Accuracy should be non-decreasing
as H grows; if it *drops*, the codebook or conditioning is the bottleneck, not
the recurrence (which is bounded by construction).

---

## References

- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
- Miyato et al., "Spectral Normalization for Generative Adversarial Networks", 2018.
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", 2022.
- Miller, "Attention Is Off By One", 2023.
- Nguyen & Salazar, "Transformers without Tears", 2019.
- Shazeer, "GLU Variants Improve Transformer", 2020.
- Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting", ICLR 2022.
- Jordan et al., "Muon: Momentum-Orthogonalized Updates", 2024.
- You et al., "LARS: Large Batch Training of CNNs with Layer-wise Adaptive Rate Scaling", 2017.
