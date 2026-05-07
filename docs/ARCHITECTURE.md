# S-TRM v3 — Architecture

This document describes the policy used in `babygroot_strm`. It is a
**stabilized, recursive** vision-language-action model that operates over a
discrete codebook produced by a frozen 3-level CQ-VAE. The recursion is
contractive by construction (Miyato spectral norm + Parcae channel-wise
contraction + convex-combo residuals), so the same checkpoint works at much
larger H at inference time than it was trained at.

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

## 3. The policy: S-TRM v3

The policy joins three sources of information through cross-attention:

```
[ vis (128 latents) | state (1 token) ]      ←  cross-attn KV
[ L0 (4) | L1 (8) | L2 (16) ]                ←  query stream (28 action tokens)
```

It maintains **two state streams** `x1, x2 ∈ ℝ^{B × 28 × d}` and runs a
recursive update with deep supervision over the outer cycles.

### 3.1 Building blocks

- **ScaleNorm** (Nguyen & Salazar 2019): single-scalar L2 rescale —
  cheaper than LayerNorm and just as stable here.
- **GeGLU** (Shazeer 2020): `GeLU(W₁x) ⊙ W₂x → W₃` — used in G.
- **Miyato spectral normalization**: every linear is wrapped with
  `torch.nn.utils.parametrizations.spectral_norm`, so its operator norm
  `σ_max(W) = 1` exactly (one power-iteration step per forward, warm-started
  across calls). This is the load-bearing stability mechanism — it removes
  the σ-cap threshold that earlier versions had to tune.
- **Softmax-1 attention sink** (Miller, "Attention Is Off By One", 2023):
  one zero-key + zero-value column appended to K and V before SDPA, so the
  softmax denominator gains an `exp(0)=1` term and heads can abstain
  cleanly without spreading mass across real keys.
- **Convex-combo residual** (instead of the usual additive one):
  `h ← (1-α) h + α f(h)` with `α = sigmoid(α_logit)`, init at `α ≈ 0.9`.
  Combined with the ≤1-Lipschitz sublayers this keeps the full F/G stack
  ≤1-Lipschitz at every depth.
- **F block:** ScaleNorm → SelfAttn (over the 28 action tokens) → conv-combo,
  ScaleNorm → CrossAttn (over `[vis|state]`) → conv-combo.
- **G block:** ScaleNorm → GeGLU → conv-combo.

`depth=2` means `F = F₂ ∘ F₁` and `G = G₂ ∘ G₁`, i.e. each inner step applies
two F-blocks (for x1) and two G-blocks (for x2).

### 3.2 Parcae channel-wise contraction (Ā)

For each stream we learn a per-channel decay `Ā ∈ (0,1)^d`:

```
Ā = exp(-|Δ| · exp(Ã))
```

Initialized so that the channel-wise mean equals a target ρ:
`ρ₁=0.75` (slow stream), `ρ₂=0.65` (fast stream), `ρ_H=0.85` (outer y_embed).
This is the same parameterization Mamba uses for selective state-space
contraction (Gu & Dao 2024) — guarantees `Ā ∈ (0, 1)^d` for any value of
`Ã, Δ`, while keeping the channel-wise dynamics learnable.

### 3.3 Inner recurrence (L steps)

```
for ℓ = 1..L:
    x1 = Ā₁ ⊙ x1 + (1/L) · F(x2 + y_embed, kv)
    x2 = Ā₂ ⊙ x2 + (1/L) · G(x1 + y_embed)
```

The `1/L` scaling is the key that makes increasing L *not* blow up the
update magnitude — the residual is averaged over the L steps so the overall
contribution stays O(1). With ‖F‖, ‖G‖ ≤ 1 (spectral norm) and Ā ∈ (0,1), the
fixed-point map is a contraction, so unrolling more steps just refines the
solution. v3 trains at L=5 — and eval generalizes to higher L without
retraining (paper §8).

### 3.4 Outer recurrence (H cycles)

```
y_embed = init(target, mask)                # masked → e_init, unmasked → E[gt]
x1 = x2 = 0
for h = 1..H:
    x1, x2 = inner_loop(x1, x2, y_embed, kv, L)
    logits = head(x1)
    if h < H:
        candidate = build_y(logits, mask, target)   # soft mixture at masked
        y_embed   = Ā_H ⊙ y_embed + (1/H) · candidate
```

Two important pieces:

- **Outer Parcae** (`Ā_H`): the same channel-wise contraction applied to the
  outer `y_embed` update, paired with `1/H` scaling on the candidate. v2
  used a hard replacement `y_embed = candidate` and was unstable as H grew;
  v3's stable update lets the same model run at H=12 at eval and the
  predictions only sharpen.
- **Stochastic-H training**: per call, sample `H ~ Uniform{1..h_max}` with
  `h_max=12`. This unlocks test-time scaling — without it, the model overfits
  to the exact H it was trained at.

### 3.5 Loss and training signal

Cross-entropy is applied at **every cycle** (deep supervision):

```
loss = (1/H) · Σ_h (1/n_levels) · Σ_l avg_masked_CE(logits_h^l, target^l)
```

CE only counts masked positions — the rest are teacher-forced via
`target_embedding`. v3 uses plain CE on hard targets (`--no-snce`).

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
- **Prompt:** each variant uses a different paraphrase of the task prompt,
  drawn from a per-base-prompt paraphrase pool. The pool is generated once
  per unique prompt — either by the Anthropic API
  (`paraphrase_prompts_with_llm`, default model `claude-haiku-4-5`) or from
  the static `PARAPHRASE_BANK` if no API key is available. InternVL3 sees
  prompt-level diversity *in addition to* visual diversity, so the cached
  features cover a wider stretch of the (image, instruction) joint
  distribution.

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

Combined with Miyato spectral norm, LARS gives well-scaled per-parameter
updates without needing a per-layer lr schedule.

## 6. Test-time scaling

The model trained at `L=5, h_max=12` keeps improving up to `H=12` at eval,
typically peaking around `H=4-6` on cold all-masked queries. The contraction
guarantees this works:

- σ_max(F), σ_max(G) ≤ 1 (Miyato spec-norm)
- Ā ∈ (0,1)^d (Parcae)
- Convex-combo residual is itself ≤1-Lipschitz when the sublayer is

So unrolling at higher H is mathematically equivalent to running more
fixed-point iterations of the same map.

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
